package variate

import (
	"math"

	"gonum.org/v1/gonum/integrate/quad"
)

// A started run is a number of independent repeats of a horizon of
// periods from an empty start: nothing in flight at period 0, the
// sources in their stationary state. Its exact means are per period of
// the horizon, from the arrivals of every earlier period and the share
// of each that passes its holding time by then.

const (
	// maxHorizon bounds a started run's horizon, in periods.
	maxHorizon = 1 << 20
	// startedWork bounds the work of the exact per-period means: the
	// convolutions of each source's arrivals with each holding's
	// weights, and the node sums over them.
	startedWork = 1 << 28
	// startedNodes is the Gauss–Legendre rule on each piece of a unit
	// interval the weights integrate over.
	startedNodes = 32
)

// Start is where a run begins. A zero Horizon is today's steady state;
// a positive one is a started run of that many periods from an empty
// start, repeated.
type Start struct {
	Horizon int
}

// Started reports whether the model is a started run.
func (m *Compiled) Started() bool { return m.horizon > 0 }

// Horizon is a started run's periods per repeat; 0 in steady state.
func (m *Compiled) Horizon() int { return int(m.horizon) }

// checkStart refuses the settings a started run has no use for.
func checkStart(g Graph, cfg Config) error {
	h := g.Start.Horizon
	switch {
	case h < 0 || h > maxHorizon:
		return modelError("start", "horizon", "must be between 1 and "+itoa(maxHorizon)+" periods")
	case h == 0 && cfg.Repeats != 0:
		return modelError("config", "Repeats", "counts the repeats of a started run; a steady-state run counts periods")
	case h == 0:
		return nil
	case cfg.Repeats < 0:
		return modelError("config", "Repeats", "must be >= 0")
	case cfg.Periods != 0:
		return modelError("config", "Periods", "a started run counts repeats of its horizon, not periods: set Repeats")
	case cfg.BurnIn != 0:
		return modelError("config", "BurnIn", "a started run starts empty at period 0 and has no burn-in")
	case cfg.BlockPeriods != 0:
		return modelError("config", "BlockPeriods", "a started run simulates one repeat of its horizon per task")
	case cfg.Transform:
		return modelError("config", "Transform", "the exact distribution assumes stationary periods; it is not computed for a started run")
	case g.Phased.Index != 0:
		return modelError("start", "phase", "a started run runs its schedules from the first value of the cycle; write the values from the season to start in instead")
	}
	return nil
}

// startedMeans fills the exact means of a started run: per period of the
// horizon and per row quantity, NaN where no closed form applies, and
// over the horizon as their sums. The variances, P(T = 0) and the
// firing probabilities are NaN.
func (m *Compiled) startedMeans(e *Expectation) {
	l := newLayout(m)
	e.ByPeriod = m.periodRows(l)
	horizon := make([]float64, l.width)
	for _, row := range e.ByPeriod {
		for q, x := range row {
			horizon[q] += x
		}
	}
	m.horizonMeans(e, l, horizon)
	nanMoments(e)
}

// periodRows is the exact mean of every quantity in every period of the
// horizon: NaN for the layers and the triggers' sums, and for all of
// them when the work would exceed its bound.
func (m *Compiled) periodRows(l layout) [][]float64 {
	rows := make([][]float64, m.horizon)
	for k := range rows {
		rows[k] = make([]float64, l.width)
	}
	ok := m.fillStartedMeans(rows, l)
	for _, row := range rows {
		lo := l.layerAt
		if !ok {
			lo = 0
		}
		// bound: row width
		for q := lo; q < l.width; q++ {
			row[q] = math.NaN()
		}
	}
	return rows
}

// horizonMeans sets the expectations' means to the horizon's sums, and
// the rounds and steps to the horizon's.
func (m *Compiled) horizonMeans(e *Expectation, l layout, horizon []float64) {
	e.Mean = horizon[l.total()]
	steps := 0.0
	for v := range m.nodes {
		e.EntriesPer[v] = horizon[l.entry(index(v))]
		e.Contrib[v] = horizon[l.contrib(index(v))]
		steps += e.EntriesPer[v]
	}
	e.StepsPerPeriod = steps
	for c := range m.categories {
		e.MeanByCategory[c] = horizon[l.cat(index(c))]
	}
	rounds := 0.0
	for k := range m.srcs {
		e.MeanBySource[k] = horizon[l.src(index(k))]
		rounds += sumOf(m.startedArrivals(k, int(m.horizon)))
		if m.srcs[k].trigger != noIndex {
			rounds = math.NaN()
		}
	}
	e.RoundsPerPeriod = rounds
	e.ArrivalsPerPeriod *= float64(m.horizon)
	for j := range m.outs {
		v := m.outcomeNode[j]
		e.OutcomeRate[j] = e.EntriesPer[v] * m.outcomeMeanProb(v, index(j))
	}
}

// nanMoments sets to NaN what a started run has no closed form for: the
// variances, the draws, P(T = 0) and the triggers' P(fire).
func nanMoments(e *Expectation) {
	nan := math.NaN()
	e.DrawsMean, e.DrawsVar, e.Var, e.pzero = nan, nan, nan, nan
	for _, v := range [][]float64{e.VarByCategory, e.VarBySource, e.ContribVar} {
		for i := range v {
			v[i] = nan
		}
	}
	for i := range e.Firing {
		e.Firing[i] = Firing{Source: e.Firing[i].Source, Lo: nan, Hi: nan, Reason: "a started run's periods are not stationary", SumMean: nan}
	}
}

// startedClass is where a source's rounds stand on reaching a node: the
// mass that reached it with no holding time on the way, and per holding
// outcome the mass that passed that one holding; mixed is set when some
// mass passed two holdings or an outage, which has no closed form.
type startedClass struct {
	none  float64
	held  []float64 // [holding outcome]
	mixed bool
}

// fillStartedMeans fills rows[k][q] for every quantity but the layers
// and triggers, NaN where the quantity is not exact. It reports false
// when the work would exceed startedWork.
func (m *Compiled) fillStartedMeans(rows [][]float64, l layout) bool {
	h := len(rows)
	holds := m.holdOutcomes()
	srcs, nodes := len(m.srcs), len(m.nodes)
	work := float64(srcs) * float64(holds.len()+1) * (float64(h)*float64(h) + float64(nodes)*float64(h))
	if work > startedWork {
		return false
	}
	pass, carry := m.holdWeights(holds, h)
	for k := range m.srcs {
		arr := m.startedArrivals(k, h)
		if arr == nil {
			m.markReached(rows, l, k)
			continue
		}
		cls := m.startedClasses(k, holds)
		passed := make([][]float64, holds.len())
		carried := make([][]float64, holds.len())
		for c := range holds.outs {
			passed[c] = convolve(arr, pass[c])
			carried[c] = convolve(arr, carry[c])
		}
		for v := range m.nodes {
			m.addNode(rows, l, k, index(v), cls[v], holds, arr, passed, carried)
		}
	}
	return true
}

// holdList is the outcomes with a holding time drawn from a law, and
// each outcome's place in that list, −1 for the others.
type holdList struct {
	outs []int32
	at   []int
}

// holdOutcomes lists the outcomes with a holding time drawn from a law.
func (m *Compiled) holdOutcomes() holdList {
	hl := holdList{at: make([]int, len(m.outs))}
	for j, o := range m.outs {
		hl.at[j] = -1
		if o.hold != noIndex && !o.outage {
			hl.at[j] = len(hl.outs)
			hl.outs = append(hl.outs, index(j))
		}
	}
	return hl
}

func (hl holdList) len() int { return len(hl.outs) }

// startedArrivals is source k's mean rounds in each period of the
// horizon: a Poisson source's rate by its schedule's value in that
// period and its factor's mean, and a renewal or block source, started
// in its stationary state, its stationary rate. Nil for a trigger, whose
// firing has no closed form here.
func (m *Compiled) startedArrivals(k, h int) []float64 {
	s := m.srcs[k]
	if s.trigger != noIndex {
		return nil
	}
	arr := make([]float64, h)
	if s.chainEnd > s.chainStart || s.every > 1 || s.repair != noIndex {
		for i := range arr {
			arr[i] = m.countMean[k]
		}
		return arr
	}
	lam := m.poolMean[s.pool]
	if s.factor != noIndex {
		lam *= m.poolMean[s.factor]
	}
	var sc *schedule
	for i := range m.schedules {
		if s.schedule != noIndex && m.schedules[i].pool == s.schedule {
			sc = &m.schedules[i]
		}
	}
	for i := range arr {
		arr[i] = lam
		if sc != nil {
			arr[i] *= sc.at(int64(i))
		}
	}
	return arr
}

// startedClasses propagates source k's rounds forward in topological
// order, splitting each node's reach by the holding it passed.
func (m *Compiled) startedClasses(k int, hl holdList) []startedClass {
	cls := make([]startedClass, len(m.nodes))
	for v := range cls {
		cls[v].held = make([]float64, hl.len())
	}
	cls[m.srcs[k].to].none = 1
	for _, v := range m.topo {
		c := cls[v]
		if c.none == 0 && !c.mixed && sumOf(c.held) == 0 {
			continue
		}
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			o := m.outs[j]
			p := m.outcomeMeanProb(v, j)
			if p == 0 {
				continue
			}
			to := &cls[o.to]
			switch {
			case o.outage:
				to.mixed = true
			case o.hold == noIndex:
				to.none += p * c.none
				for i, x := range c.held {
					to.held[i] += p * x
				}
				to.mixed = to.mixed || c.mixed
			default:
				to.held[hl.at[j]] += p * c.none
				to.mixed = to.mixed || c.mixed || sumOf(c.held) > 0
			}
		}
	}
	return cls
}

func sumOf(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s
}

// holdWeights tabulates, per holding outcome and lag d = k − i in
// periods, the expected share of a round arriving in period i, at a
// uniform time within it, that passes the holding in period k, pass;
// and the expected time it is held within period k, carry.
//
// With S the holding's survival function, pass is ∫ S over [d − 1, d]
// less ∫ S over [d, d + 1], and 1 − ∫ S over [0, 1] at d = 0; carry is
// ∫ (1 − |z − d|) S(z) dz over [d − 1, d + 1] from z = 0.
func (m *Compiled) holdWeights(hl holdList, h int) (pass, carry [][]float64) {
	x := make([]float64, startedNodes)
	wt := make([]float64, startedNodes)
	quad.Legendre{}.FixedLocations(x, wt, 0, 1)
	pass = make([][]float64, hl.len())
	carry = make([][]float64, hl.len())
	for c, j := range hl.outs {
		d := m.pools[m.outs[j].hold]
		lo, hi := d.Support()
		// ∫ g(z) S(z) dz over [a, b], split at the support's ends so that
		// a constant holding's step falls on a piece's edge.
		integral := func(a, b float64, g func(float64) float64) float64 {
			cuts := [4]float64{a, math.Min(math.Max(lo, a), b), math.Min(math.Max(hi, a), b), b}
			sum := 0.0
			for p := 0; p < 3; p++ {
				u, v := cuts[p], cuts[p+1]
				if !(v > u) {
					continue
				}
				part := 0.0
				for i := range x {
					z := u + (v-u)*x[i]
					part += wt[i] * g(z) * d.Survival(z)
				}
				sum += part * (v - u)
			}
			return sum
		}
		one := func(float64) float64 { return 1 }
		pass[c] = make([]float64, h)
		carry[c] = make([]float64, h)
		for lag := range h {
			fd := float64(lag)
			tri := func(z float64) float64 { return 1 - math.Abs(z-fd) }
			if lag == 0 {
				pass[c][0] = 1 - integral(0, 1, one)
				carry[c][0] = integral(0, 1, tri)
				continue
			}
			pass[c][lag] = integral(fd-1, fd, one) - integral(fd, fd+1, one)
			carry[c][lag] = integral(fd-1, fd, tri) + integral(fd, fd+1, tri)
		}
	}
	return pass, carry
}

// convolve returns y[k] = Σ_{i ≤ k} a[i]·w[k − i] over the horizon.
func convolve(a, w []float64) []float64 {
	y := make([]float64, len(a))
	for i, ai := range a {
		if ai == 0 {
			continue
		}
		// bound: the horizon
		for k := i; k < len(a); k++ {
			y[k] += ai * w[k-i]
		}
	}
	return y
}

// addNode adds source k's mean entries into node v, and its value booked
// there, to each period's row: the total, the category, the source, the
// node's entries and contribution. A node reached through two holdings
// or an outage, and a rate booked over a holding after another, are NaN.
func (m *Compiled) addNode(rows [][]float64, l layout, k int, v int32, c startedClass, hl holdList, arr []float64, passed, carried [][]float64) {
	if c.none == 0 && !c.mixed && sumOf(c.held) == 0 {
		return
	}
	if c.mixed {
		m.markNode(rows, l, k, v)
		return
	}
	entries := make([]float64, len(rows))
	for i := range entries {
		entries[i] = c.none * arr[i]
		for h, x := range c.held {
			entries[i] += weighted(x, passed[h][i])
		}
	}
	for i, row := range rows {
		row[l.entry(v)] += entries[i]
	}
	if !m.nodes[v].hasFunctions() {
		return
	}
	held, heldOK := m.heldTime(v, c, hl, carried, len(rows))
	book := func(q int, filt valueFilter) {
		m.bookPeriods(rows, q, v, filt, entries, held, heldOK)
	}
	book(l.total(), valueFilter{kind: filterAll})
	book(l.src(index(k)), valueFilter{kind: filterAll})
	book(l.contrib(v), valueFilter{kind: filterNode, index: v})
	for cat := range m.categories {
		book(l.cat(index(cat)), valueFilter{kind: filterCategory, index: index(cat)})
	}
}

// heldTime is the mean time per period that rounds entering node v hold
// over its outcomes' holding times, from the carry weights; ok is false
// when some of it follows an earlier holding or is an outage, which have
// no closed form.
func (m *Compiled) heldTime(v int32, c startedClass, hl holdList, carried [][]float64, h int) (held []float64, ok bool) {
	held = make([]float64, h)
	n := m.nodes[v]
	delayed := sumOf(c.held) > 0
	ok = true
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		o := m.outs[j]
		p := m.outcomeMeanProb(v, j)
		if (o.hold == noIndex && !o.outage) || p == 0 {
			continue
		}
		if o.outage || delayed {
			ok = false
			continue
		}
		for i, x := range carried[hl.at[j]] {
			held[i] += p * c.none * x
		}
	}
	return held, ok
}

// bookPeriods adds to column q of each period's row node v's value under
// the filter: its entry value times the entries, and its rate times the
// time held, NaN where that has no closed form.
func (m *Compiled) bookPeriods(rows [][]float64, q int, v int32, filt valueFilter, entries, held []float64, heldOK bool) {
	n := m.nodes[v]
	nm := m.nodeMomentsFor(v, filt)
	own := nm.a
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		x, _ := m.outcomeFn(j, filt)
		own += weighted(m.outcomeMeanProb(v, j), x)
	}
	for i, row := range rows {
		x := weighted(entries[i], own)
		switch {
		case nm.rho == 0:
		case heldOK:
			x += nm.rho * held[i]
		default:
			x = math.NaN()
		}
		row[q] += x
	}
}

// markNode sets to NaN what source k's rounds book at node v: its entries
// and contribution, the source, the total, and every category.
func (m *Compiled) markNode(rows [][]float64, l layout, k int, v int32) {
	nan := math.NaN()
	for _, row := range rows {
		row[l.entry(v)] = nan
		if m.nodes[v].hasFunctions() {
			row[l.total()] = nan
			row[l.src(index(k))] = nan
			row[l.contrib(v)] = nan
			for c := range m.categories {
				if m.nodeBooks(v, index(c)) {
					row[l.cat(index(c))] = nan
				}
			}
		}
	}
}

// nodeBooks reports whether node v has a function in category c: an
// OnEntry or PerTime function, the threshold's, or a branch's half of it.
func (m *Compiled) nodeBooks(v, c int32) bool {
	n := m.nodes[v]
	in := func(f int32) bool { return f != noIndex && m.fns[f].category == c }
	// bound: functions of one node
	for f := n.fnStart; f < n.rateEnd; f++ {
		if in(f) {
			return true
		}
	}
	if in(n.thresh) {
		return true
	}
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		if in(m.outs[j].fn) {
			return true
		}
	}
	return false
}

// markReached sets to NaN everything source k's rounds reach.
func (m *Compiled) markReached(rows [][]float64, l layout, k int) {
	for v := range m.nodes {
		if m.reach[k][v] > 0 {
			m.markNode(rows, l, k, index(v))
		}
	}
}

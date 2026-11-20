package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// The between-period variance with holding times. The rounds of one
// period share its draws Θ, its rates and whatever else is drawn once a
// period, and the rounds of different periods share nothing, so
//
//	Var[T] = Σ_k λ_k·E[Q_k] + Σ_d Var_Θ(Σ_k Λ_k·W_k(d | Θ)),
//
// W_k(d) = E[∫ K(x − d) μ_k(dx)] the mean that a round of source k,
// arriving at a uniform time in its period, books into the period d
// later. Given the draws other than the rates, Λ is independent of them,
// so the second term is Σ_{k,l} (E[Λ_k Λ_l] − λ_k λ_l)·Σ_d W_k(d)·W_l(d),
// and the conditioning on the other draws adds Σ_d Var(Σ_k λ_k W_k(d))
// over their points. From node v,
//
//	W_v(t) = Σ_j p_j (E[V + x_j]·K(t) + E[R]·(E[Φ(τ_j − t)] − Φ(−t)) + E[W_to(t − τ_j)]),
//
// Φ(z) = ∫_{−∞}^z K, on a grid of lags t from −1 to the span, n cells a
// period, stored from the span down so that each expectation over a
// holding is the lag rule's E[g(s + τ)] with the law in units of the
// grid's length.

const (
	// lagTailMass is the share of each holding's mass the span leaves out.
	lagTailMass = 1e-10
	// lagCellsMax caps a profile grid's cells, within what one FFT takes.
	lagCellsMax = 1 << 21
	// lagCellsMin is the fewest cells a period a profile grid starts on.
	lagCellsMin = 4
	// lagBudget caps the profile grids' work: their cells times the passes
	// over the outcomes, summed over the grids and a mixture's points.
	lagBudget = 1 << 22
)

// lagGrid is the grid of the lag profiles: n cells a period over the
// span + 1 periods from lag −1, index i at lag span − i/n.
type lagGrid struct {
	n, span, cells int
	k, phi         []float64 // K(t) and Φ(−t) at each point
}

func newLagGrid(n, span int) lagGrid {
	cells := n * (span + 1)
	g := lagGrid{n: n, span: span, cells: cells, k: make([]float64, cells+1), phi: make([]float64, cells+1)}
	h := 1 / float64(n)
	for i := range g.k {
		t := float64(span) - float64(i)*h
		g.k[i] = math.Max(0, 1-math.Abs(t))
		g.phi[i] = triangleCDF(-t)
	}
	return g
}

// triangleCDF is Φ(z) = ∫_{−∞}^z K.
func triangleCDF(z float64) float64 {
	switch {
	case z <= -1:
		return 0
	case z <= 0:
		return (z + 1) * (z + 1) / 2
	case z < 1:
		return 1 - (1-z)*(1-z)/2
	}
	return 1
}

// at is the grid index of integer lag d.
func (g *lagGrid) at(d int) int { return (g.span - d) * g.n }

// lagSpan is the periods past a round's arrival its bookings reach but
// for lagTailMass of each holding: the longest path's sum of the
// holdings' quantiles at 1 − lagTailMass; NaN past any bound.
func (m *Compiled) lagSpan() float64 {
	reach := make([]float64, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		nd := m.nodes[v]
		// bound: outcomes of one node
		for j := nd.outStart; j < nd.outEnd; j++ {
			o := m.outs[j]
			q := 0.0
			if o.hold != noIndex {
				q = m.pools[o.hold].Quantile(1 - lagTailMass)
			}
			reach[v] = math.Max(reach[v], q+reach[o.to])
		}
	}
	span := 0.0
	for _, s := range m.srcs {
		span = math.Max(span, reach[s.to])
	}
	return span
}

// lagRules is each outcome's rule on the profile grid, nil without a
// holding: its law over the grid's length, as a member given its
// group's normal where it is one.
func (m *Compiled) lagRules(g *lagGrid) ([]*lagRule, bool) {
	out := make([]*lagRule, len(m.outs))
	length := float64(g.span + 1)
	for j, o := range m.outs {
		if o.hold == noIndex {
			continue
		}
		d := m.pools[o.hold]
		lo, hi := d.Support()
		key := ruleKey{pool: o.hold, n: g.cells, span: g.span}
		if c, ok := m.condOf[o.hold]; ok {
			key.a, key.b, key.member = c.a, c.b, true
		}
		if lo == hi {
			key.fixed, key.at = true, lo
		}
		s, err := dist.NewScaled(d, 1/length)
		if err != nil {
			return nil, false
		}
		out[j] = m.cfg.rules.rule(key, func() lagRule {
			floor := 0.0
			if key.member {
				floor = lawNegligible
			}
			r := lagRuleWith(s, g.cells, func(mu *[4][]float64) (float64, float64) {
				if f0, tail, ok := m.cfg.rules.spanOf(key, mu); ok {
					return f0, tail
				}
				f0, tail := subMomentsAbove(s, g.cells, mu, floor)
				m.cfg.rules.keepSpan(key, mu, f0, tail)
				return f0, tail
			})
			r.ePhi = phiExpect(&r, g)
			r.cum = [4][]float64{}
			return r
		})
	}
	return out, true
}

// phiExpect is E[Φ(τ − t)] at every lag t of the profile grid, exactly:
// Φ is quadratic between the kinks at τ = t − 1, t and t + 1, which are
// cell bounds, so each piece is the cells' moments up to the second,
// summed from 0, in periods.
func phiExpect(r *lagRule, g *lagGrid) []float64 {
	cells := g.cells
	length := float64(g.span + 1)
	h := 1 / float64(g.n)
	// sum is ∫ (τ − a)² dF over cells [lo, hi), τ in periods.
	sum := func(k, lo, hi int) float64 {
		lo, hi = max(0, min(lo, cells)), max(0, min(hi, cells))
		return r.cum[k][hi] - r.cum[k][lo]
	}
	sq := func(lo, hi int, a float64) float64 {
		m0, m1, m2 := sum(0, lo, hi), sum(1, lo, hi)*length, sum(2, lo, hi)*length*length
		return m2 - 2*a*m1 + a*a*m0
	}
	out := make([]float64, cells+1)
	for i := range out {
		t := float64(g.span) - float64(i)*h
		c := g.span*g.n - i // the cell at which τ = t
		v := r.f0 * triangleCDF(-t)
		v += sq(c-g.n, c, t-1) / 2
		v += sum(0, c, c+g.n) - sq(c, c+g.n, t+1)/2
		v += sum(0, c+g.n, cells) + r.tail
		out[i] = v
	}
	return out
}

// subMaxCells caps the sub-cells one cell's moments are taken on.
const subMaxCells = 1 << 12

// subMoments fills each of n cells' moments ∫ u^k dF, k ≤ 3, of law d,
// as cellMoments does, on sub-cells no wider than the law's local scale
// over lagFine: its interquartile range, growing with u past the median
// as a law's spread does, so that a narrow law needs no fine grid. It
// stops where the survival is lawNegligible, and gives the mass at 0 and
// past 1.
func subMoments(d dist.Distribution, n int, mu *[4][]float64) (f0, tail float64) {
	return subMomentsAbove(d, n, mu, 0)
}

// subMomentsAbove is subMoments with the cells whose CDF at their end is
// below floor left empty: a group member's law given its normal, narrow,
// has no mass to speak of below its own spread.
func subMomentsAbove(d dist.Distribution, n int, mu *[4][]float64, floor float64) (f0, tail float64) {
	h := 1 / float64(n)
	med := d.Quantile(0.5)
	iqr := d.Quantile(0.75) - d.Quantile(0.25)
	f0 = d.CDF(0)
	gx, gw := legendre()
	lo := lawPoint{x: 0, f: f0, s: d.Survival(0)}
	// bound: n cells
	for j := range n {
		if lo.s < lawNegligible {
			break
		}
		b := float64(j+1) * h
		hi := lawPoint{x: b, f: d.CDF(b), s: d.Survival(b)}
		if hi.f >= floor {
			scale := iqr * math.Max(1, lo.x/med)
			sub := subMaxCells
			if scale > 0 {
				sub = min(subMaxCells, max(1, int(math.Ceil(h*lagFine/scale))))
			}
			lawCell(d, lo, hi, sub, gx, gw, mu, j)
		}
		lo = hi
	}
	return f0, d.Survival(1)
}

// lawPoint is a law's CDF and survival at x.
type lawPoint struct{ x, f, s float64 }

// lawCell adds cell j's moments from lo to hi, split where the law's
// support starts or ends inside it, past which its CDF is not smooth,
// each piece on sub sub-cells.
func lawCell(d dist.Distribution, lo, hi lawPoint, sub int, gx, gw []float64, mu *[4][]float64, j int) {
	from, to := d.Support()
	a := lo
	for _, x := range [2]float64{from, to} {
		if x > a.x && x < hi.x {
			b := lawPoint{x: x, f: d.CDF(x), s: d.Survival(x)}
			cellOnSubs(d, a, b, sub, gx, gw, mu, j)
			a = b
		}
	}
	cellOnSubs(d, a, hi, sub, gx, gw, mu, j)
}

// cellOnSubs adds cell j's moments, from lo to hi, taken on sub equal
// sub-cells by the Gauss–Legendre rule gx, gw.
func cellOnSubs(d dist.Distribution, lo, hi lawPoint, sub int, gx, gw []float64, mu *[4][]float64, j int) {
	w := (hi.x - lo.x) / float64(sub)
	a := lo
	// bound: subMaxCells sub-cells
	for q := range sub {
		b := hi
		if q < sub-1 {
			x := lo.x + float64(q+1)*w
			b = lawPoint{x: x, f: d.CDF(x), s: d.Survival(x)}
		}
		var m [4]float64
		lawSpan(d, a, b, gx, gw, &m)
		for k := range m {
			mu[k][j] += m[k]
		}
		a = b
	}
}

// lawSpan is ∫_a^b u^k dF, k ≤ 3, by parts on the Gauss–Legendre rule
// gx, gw, with the survival in place of the CDF above the median.
func lawSpan(d dist.Distribution, a, b lawPoint, gx, gw []float64, m *[4]float64) {
	w := b.x - a.x
	upper := a.f > 0.5
	var in [3]float64
	for i := range gx {
		u := a.x + gx[i]*w
		v := d.CDF(u)
		if upper {
			v = d.Survival(u)
		}
		wu := gw[i] * w * v
		for k := range in {
			in[k] += wu
			wu *= u
		}
	}
	if upper {
		m[0] = a.s - b.s
	} else {
		m[0] = b.f - a.f
	}
	ak, bk := a.x, b.x
	for k := 1; k < 4; k++ {
		if upper {
			m[k] = ak*a.s - bk*b.s + float64(k)*in[k-1]
		} else {
			m[k] = bk*b.f - ak*a.f - float64(k)*in[k-1]
		}
		ak, bk = ak*a.x, bk*b.x
	}
}

// lagProfiles is W_v on the grid for every node under the filter.
func (m *Compiled) lagProfiles(filt valueFilter, rules []*lagRule, g *lagGrid) [][]float64 {
	w := make([][]float64, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		nm := m.nodeMomentsFor(v, filt)
		nd := m.nodes[v]
		wv := make([]float64, g.cells+1)
		if nd.outStart == nd.outEnd {
			for k := range wv {
				wv[k] = nm.a * g.k[k]
			}
		}
		// bound: outcomes of one node
		for j := nd.outStart; j < nd.outEnd; j++ {
			m.profileBranch(nm, v, j, filt, rules[j], g, w[m.outs[j].to], wv)
		}
		w[v] = wv
	}
	return w
}

// profileBranch adds outcome j's term to node v's profile wv, down being
// the profile of the node it leads to.
func (m *Compiled) profileBranch(nm nodeMoments, v, j int32, filt valueFilter, r *lagRule, g *lagGrid, down, wv []float64) {
	p := m.outcomeMeanProb(v, j)
	x, _ := m.outcomeFn(j, filt)
	vx := nm.a + x
	var rate []float64
	if r != nil {
		down = r.apply(down)
		if nm.rho != 0 {
			rate = r.ePhi
		}
	}
	for k := range wv {
		add := vx*g.k[k] + down[k]
		if rate != nil {
			add += nm.rho * (rate[k] - g.phi[k])
		}
		wv[k] += weighted(p, add)
	}
}

// throughProfiles is C_v on the grid for every node under the filter:
// the part of W_v that passes through outcome cut, before its holding,
// so that W_v(t | τ) = A_v(t) + C_v(t − τ) for the cut's holding τ, w
// being every node's W_v.
func (m *Compiled) throughProfiles(filt valueFilter, rules []*lagRule, g *lagGrid, cut int32, w [][]float64) [][]float64 {
	c := make([][]float64, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		nd := m.nodes[v]
		cv := make([]float64, g.cells+1)
		// bound: outcomes of one node
		for j := nd.outStart; j < nd.outEnd; j++ {
			p := m.outcomeMeanProb(v, j)
			var add []float64
			switch {
			case j == cut:
				add = slicesClone(w[m.outs[j].to])
				if rho := m.nodeMomentsFor(v, filt).rho; rho != 0 {
					for k := range add {
						add[k] += rho * g.phi[k]
					}
				}
			case c[m.outs[j].to] == nil:
				continue
			case rules[j] != nil:
				add = rules[j].apply(c[m.outs[j].to])
			default:
				add = c[m.outs[j].to]
			}
			for k := range cv {
				cv[k] += weighted(p, add[k])
			}
		}
		if !allZero(cv) {
			c[v] = cv
		}
	}
	return c
}

func slicesClone(x []float64) []float64 { return append([]float64(nil), x...) }

// periodCuts lists the outcomes whose holding is drawn once a period;
// false when a path passes two of them, whose shifts would not be
// independent of each other's.
func (m *Compiled) periodCuts() ([]int32, bool) {
	var cuts []int32
	for j, o := range m.outs {
		if o.hold != noIndex && m.poolPeriod[o.hold] != noIndex {
			if _, member := m.condOf[o.hold]; member {
				return nil, false
			}
			cuts = append(cuts, index(j))
		}
	}
	if len(cuts) < 2 {
		return cuts, true
	}
	below := make([]int, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		nd := m.nodes[v]
		// bound: outcomes of one node
		for j := nd.outStart; j < nd.outEnd; j++ {
			n := below[m.outs[j].to]
			if m.outs[j].hold != noIndex && m.poolPeriod[m.outs[j].hold] != noIndex {
				n++
			}
			below[v] = max(below[v], n)
		}
		if below[v] > 1 {
			return nil, false
		}
	}
	return cuts, true
}

// sourceProfiles is W_k(d) at the integer lags 0 to the span for each
// source under the filter, and for each cut each source's C_k on the
// whole grid, nil where it is 0.
func (m *Compiled) sourceProfiles(filt valueFilter, rules []*lagRule, g *lagGrid, cuts []int32) (mean [][]float64, through [][][]float64) {
	w := m.lagProfiles(filt, rules, g)
	mean = make([][]float64, len(m.srcs))
	for k, s := range m.srcs {
		mean[k] = make([]float64, g.span+1)
		for d := range mean[k] {
			mean[k][d] = w[s.to][g.at(d)]
		}
	}
	through = make([][][]float64, len(cuts))
	for i, cut := range cuts {
		c := m.throughProfiles(filt, rules, g, cut, w)
		through[i] = make([][]float64, len(m.srcs))
		for k, s := range m.srcs {
			through[i][k] = c[s.to]
		}
	}
	return mean, through
}

// shiftCov is Σ_d Cov_τ(C_k(d − τ), C_l(d − τ)) over the integer lags
// for a cut's holding τ by its rule r, given E[C_k(d − τ)] and
// E[C_l(d − τ)] on the grid.
func shiftCov(r *lagRule, g *lagGrid, ck, cl, ek, el []float64) float64 {
	prod := make([]float64, len(ck))
	for i := range ck {
		prod[i] = ck[i] * cl[i]
	}
	e2 := r.apply(prod)
	v := 0.0
	// bound: the span's lags
	for d := 0; d <= g.span; d++ {
		i := g.at(d)
		v += e2[i] - ek[i]*el[i]
	}
	return v
}

// cutCovs holds, for one filter, each cut's rule and the sources' parts
// through it, and their expectations over its holding and covariances
// as they are first asked for.
type cutCovs struct {
	g       *lagGrid
	rules   []*lagRule
	through [][][]float64
	mean    [][][]float64
	cov     map[[3]int]float64
}

func newCutCovs(g *lagGrid, rules []*lagRule, through [][][]float64) *cutCovs {
	c := &cutCovs{g: g, rules: rules, through: through, mean: make([][][]float64, len(rules)), cov: map[[3]int]float64{}}
	for i := range rules {
		c.mean[i] = make([][]float64, len(through[i]))
	}
	return c
}

// at is Σ_d Cov_τ(C_k, C_l) for cut i, 0 where either part is.
func (c *cutCovs) at(i, k, l int) float64 {
	if k > l {
		k, l = l, k
	}
	ck, cl := c.through[i][k], c.through[i][l]
	if ck == nil || cl == nil {
		return 0
	}
	key := [3]int{i, k, l}
	if v, ok := c.cov[key]; ok {
		return v
	}
	for _, s := range [2]int{k, l} {
		if c.mean[i][s] == nil {
			c.mean[i][s] = c.rules[i].apply(c.through[i][s])
		}
	}
	v := shiftCov(c.rules[i], c.g, ck, cl, c.mean[i][k], c.mean[i][l])
	c.cov[key] = v
	return v
}

// mixedRates reports whether a source's rate is drawn, so that E[Λ_k
// Λ_l] differs from λ_k λ_l.
func (m *Compiled) mixedRates() bool {
	for k := range m.srcs {
		if m.countN2[k] != m.countMean[k]*m.countMean[k] {
			return true
		}
	}
	return false
}

// rateExcess is E[Λ_k Λ_l] − λ_k λ_l.
func (m *Compiled) rateExcess(k, l int) float64 {
	if k == l {
		return m.countN2[k] - m.countMean[k]*m.countMean[k]
	}
	return m.countPair(k, l) - m.countMean[k]*m.countMean[l]
}

// between is the key's between-period term from the sources' mean
// profiles w and the covariances over the cuts' holdings cc,
// Σ_{k,l} [(E[Λ_k Λ_l] − λ_k λ_l)·Σ_d W_k(d)·W_l(d) + E[Λ_k Λ_l]·Σ_cuts Σ_d Cov_τ(C_k, C_l)],
// and its mean booking by lag, Σ_k λ_k W_k(d), over every source or the
// key's one.
func (m *Compiled) between(key delayKey, w [][]float64, cc *cutCovs, lam []float64) (float64, []float64) {
	prof := make([]float64, len(w[0]))
	v := 0.0
	in := func(k int) bool { return key.src < 0 || key.src == k }
	for k := range m.srcs {
		if !in(k) {
			continue
		}
		for d := range prof {
			prof[d] += lam[k] * w[k][d]
		}
		for l := range m.srcs {
			if in(l) {
				v += m.pairBetween(k, l, w, cc, lam)
			}
		}
	}
	return v, prof
}

// pairBetween is sources k and l's share of the between-period term.
func (m *Compiled) pairBetween(k, l int, w [][]float64, cc *cutCovs, lam []float64) float64 {
	c := m.rateExcess(k, l)
	v := 0.0
	if c != 0 {
		v += c * dot(w[k], w[l])
	}
	for i := range cc.rules {
		if cov := cc.at(i, k, l); cov != 0 {
			v += (c + lam[k]*lam[l]) * cov
		}
	}
	return v
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// The schedules. Given the place φ of a period in the joint cycle,
// Var(T) = E_φ Var(T | φ) + Var_φ E(T | φ). A lag's term given φ
// depends on φ only through the schedules' values at that lag, so the
// first is the variance with each schedule a rate drawn independently
// every period from its cycle's values, as the rates' term takes it;
// that counts Σ_d Var_φ(M_d(φ)) of the mean booked from each lag, M_d(φ)
// = Σ_k λ_k·(s_k(φ − d)/s̄_k)·W̄_k(d), where the schedules give one
// sequence, not independent draws, and the second is the variance of
// their sum: the term is Var_φ(Σ_d M_d(φ)) − Σ_d Var_φ(M_d(φ)).

// scheduleBetween adds the schedules' term, if any, to each key's
// between-period term bs, settled on the profile grids beside within + bs; false where
// it is not computed: with values or probabilities drawn once a period,
// or under a value factor's conditioning, whose mixtures would need the
// schedules inside each point, or past the cycle's or the span's caps.
func (m *Compiled) scheduleBetween(draws bool, models []*Compiled, weights []float64, keys []delayKey, lam, within, bs []float64) ([]float64, bool) {
	if len(m.schedules) == 0 {
		return bs, true
	}
	if draws || m.cfg.profiles || bs == nil {
		return nil, false
	}
	l := m.lagSpan()
	if !(l < float64(delayBudget)) {
		return nil, false
	}
	span := int(math.Ceil(l))
	if n, ok := jointLength(m.schedules); !ok || n*int64(len(m.schedules)) > lagBudget {
		return nil, false
	}
	whole := make([]float64, len(keys))
	for k := range keys {
		whole[k] = within[k] + bs[k]
	}
	cs, _, _ := settleTerm(span, models, weights, keys, lam, whole, lagBudget, scheduleAt)
	out := make([]float64, len(keys))
	for k := range keys {
		out[k] = bs[k] + cs[k]
	}
	return out, true
}

// jointLength is the schedules' joint cycle in periods; false past
// maxJointPeriods.
func jointLength(scs []schedule) (int64, bool) {
	n := int64(1)
	for _, sc := range scs {
		k := int64(len(sc.values))
		n = n / int64(gcd(int(n), int(k))) * k
		if n > maxJointPeriods {
			return 0, false
		}
	}
	return n, true
}

// scheduleAt adds each unsettled key's schedules' term on an n-cell a
// period profile grid to v, from the sources' mean profiles mixed over
// the models.
func scheduleAt(cs []*Compiled, ws []float64, keys []delayKey, done []bool, lam []float64, n, span int, v []float64) ([][]float64, bool) {
	g := newLagGrid(n, span)
	byFilt, _, ok := mixProfiles(cs, ws, keys, done, &g)
	if !ok {
		return nil, false
	}
	for k, key := range keys {
		if !done[k] {
			v[k] += cs[0].scheduleTerm(key, byFilt[key.filt].mean, lam)
		}
	}
	return nil, true
}

// scheduleTerm is Var_φ(Σ_d M_d(φ)) − Σ_d Var_φ(M_d(φ)) for the key from
// the sources' mean profiles w: each schedule's lags folded onto its
// cycle and its values convolved with them, the sum over the schedules
// walked over the joint cycle.
func (m *Compiled) scheduleTerm(key delayKey, w [][]float64, lam []float64) float64 {
	b := m.scheduleLags(key, w, lam)
	perLag := 0.0
	cycles := make([][]float64, len(m.schedules))
	for i, sc := range m.schedules {
		if b[i] == nil {
			continue
		}
		for j, other := range m.schedules {
			if b[j] != nil {
				perLag += meanOf(sc.values) * meanOf(other.values) * (m.scheduleRatio[i][j] - 1) * dot(b[i], b[j])
			}
		}
		cycles[i] = cycleSum(sc.values, b[i])
	}
	n, _ := jointLength(m.schedules)
	var s1, s2 float64
	// bound: the joint cycle, at most maxJointPeriods
	for p := range n {
		x := 0.0
		for _, c := range cycles {
			if c != nil {
				x += c[p%int64(len(c))]
			}
		}
		s1, s2 = s1+x, s2+x*x
	}
	mean := s1 / float64(n)
	return math.Max(0, s2/float64(n)-mean*mean) - perLag
}

// scheduleLags is, per schedule, Σ_k λ_k·W̄_k(d)/s̄_k over the key's
// sources on it; nil for a schedule with none.
func (m *Compiled) scheduleLags(key delayKey, w [][]float64, lam []float64) [][]float64 {
	b := make([][]float64, len(m.schedules))
	for k, src := range m.srcs {
		if src.schedule == noIndex || key.src >= 0 && key.src != k {
			continue
		}
		i := m.scheduleIndex(src.schedule)
		mean := meanOf(m.schedules[i].values)
		if mean == 0 {
			continue
		}
		if b[i] == nil {
			b[i] = make([]float64, len(w[k]))
		}
		for d, x := range w[k] {
			b[i][d] += lam[k] * x / mean
		}
	}
	return b
}

// cycleSum is Σ_d b(d)·v((φ − d) mod L) at each place φ of a cycle of
// values v, the lags folded onto it.
func cycleSum(v, b []float64) []float64 {
	l := len(v)
	fold := make([]float64, l)
	for d, x := range b {
		fold[d%l] += x
	}
	out := make([]float64, l)
	for phi := range out {
		for r, x := range fold {
			out[phi] += x * v[((phi-r)%l+l)%l]
		}
	}
	return out
}

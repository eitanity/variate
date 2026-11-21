package variate

import "math"

// Stat summarises one per-period quantity.
type Stat struct {
	N                        int64
	Mean, StdErr, StdDev     float64
	Min, Max                 float64
	P50, P90, P95, P99, P999 float64
	Quantiles                bool // the percentiles exist: the quantity has a histogram
	Clipped                  bool
	Share                    float64
	StdErrUnreliable         bool // the variance is infinite, so the standard error means nothing

	// Per level of QuantileLevels(): the quantile's 95% interval, from the
	// order statistics, and its standard error, the interval's width over
	// 2·1.96. The interval is asymmetric in a skewed tail; quote it rather
	// than the quantile ± 1.96 standard errors. With delays, correlated
	// periods widen it by the batch-means inflation at the quantile, and
	// it is NaN below 16 full blocks.
	QLo, QHi, QSE [5]float64
}

// QuantileLevels returns the levels of P50, P90, P95, P99 and P999.
func QuantileLevels() [5]float64 { return [5]float64{0.5, 0.9, 0.95, 0.99, 0.999} }

// quantileZ is the normal quantile of the two-sided 95% interval.
const quantileZ = 1.959963984540054

// Prop is a proportion with its binomial standard error.
type Prop struct{ P, StdErr float64 }

// ECPoint is one point of the exceedance curve: P(T ≥ X) with its
// standard error, binomial for independent periods and from batch means
// with delays.
type ECPoint struct{ X, P, StdErr float64 }

// PlanRow is one convergence criterion: planned against observed.
type PlanRow struct {
	Criterion        string
	Driver           string
	PlannedPerPeriod float64
	PeriodsNeeded    int64
	Observed         float64
	Met              bool
	NotApplicable    string // why the criterion does not apply, which then counts as met; "" where it does
}

// Summary is the report of a run, with names as keys.
type Summary struct {
	Total      Stat
	Exceedance []ECPoint
	ByCategory map[string]Stat // empty, as are BySource and ContribPer, with Config.NoBreakdown
	BySource   map[string]Stat

	ReachProb  map[string]map[string]Prop
	Terminal   map[string]map[string]Prop
	OutcomeP   map[string]Prop
	EntriesPer map[string]Stat
	ContribPer map[string]Stat // value booked per period at each node
	Layers     map[string]Stat // each layer's payout per period, its share that of the total's mean
	Totals     map[string]Stat // each named total per period; its share NaN when a category of it is outside the total
	Triggers   map[string]Stat // by the trigger's source name: the sum it tests per period
	Plan       []PlanRow
	Queues     map[string]QueueSummary // by the queue node's name
	At         []LevelStat             // Config.At in its order
	AtMostZero []LevelStat             // P(Q ≤ 0) of each named total that subtracts, in the model's order
	When       []WhenStat              // Config.When in its order
}

// LevelStat is the chance a quantity reaches one of Config.At's values:
// P is the share of the N periods (repeats, in a started run) in which
// it is at least X, and P(Q < X) is 1 − P; the standard error is
// binomial, widened as a mean's is where periods are correlated.
type LevelStat struct {
	Level
	N         int64
	P, StdErr float64
}

// QueueSummary is a queue's figures: over the rounds it served, their
// wait and stay; and per period (in a started run, per repeat of the
// horizon) the rounds that joined, those served, the line's length at
// the period's end, the share of the servers' time busy, and the rounds
// sent to the unfinished node.
type QueueSummary struct {
	Servers                    int
	Wait, Stay                 ItemStat
	Joined, Served, Unfinished Stat
	Length                     Stat // summed over the horizon's period ends in a started run
	Busy                       Stat // busy server time
	Utilisation                float64
	Line                       float64 // the mean line at a period's end
	LongRun                    QueueLongRun
	LookBack                   float64 // a carried queue's mean periods its exact start looked back in a steady-state run; NaN otherwise
}

// exceedancePoints is the number of log-spaced points of the curve.
const exceedancePoints = 64

// Summarize fills a caller-owned Summary from the workspace's result,
// reusing its slices and maps.
func (w *Workspace) Summarize(dst *Summary) {
	r := &w.result
	l := &w.l
	dst.Total = w.statAt(l.total(), &r.Total, 1)
	dst.Exceedance = w.exceedance(dst.Exceedance[:0])
	breakdown := w.m.crit.breakdown
	dst.ByCategory = ensureStats(dst.ByCategory)
	dst.BySource = ensureStats(dst.BySource)
	dst.EntriesPer = ensureStats(dst.EntriesPer)
	dst.ContribPer = ensureStats(dst.ContribPer)
	clear(dst.ByCategory)
	clear(dst.BySource)
	clear(dst.ContribPer)
	for c, name := range r.Categories {
		if breakdown {
			dst.ByCategory[name] = w.statAt(l.cat(index(c)), &r.ByCategory[c], w.shareBase(l.cat(index(c))))
		}
	}
	dst.Totals = ensureStats(dst.Totals)
	for i, t := range w.m.totals {
		dst.Totals[t.id] = w.statAt(l.named(i), &r.Totals[i], w.shareBase(l.named(i)))
	}
	for k, name := range r.Sources {
		if breakdown {
			dst.BySource[name] = w.statAt(l.src(index(k)), &r.BySource[k], r.Total.Mean())
		}
	}
	for v, name := range r.Nodes {
		dst.EntriesPer[name] = w.statAt(l.entry(index(v)), &r.EntriesPer[v], 1)
		if breakdown {
			dst.ContribPer[name] = w.statAt(l.contrib(index(v)), &r.Contrib[v], r.Total.Mean())
		}
	}
	dst.Layers = ensureStats(dst.Layers)
	for j, y := range w.m.layers {
		dst.Layers[y.ID] = w.stat(&r.Layers[j], r.Total.Mean())
	}
	dst.Triggers = ensureStats(dst.Triggers)
	for i, tr := range w.m.trigs {
		dst.Triggers[r.Sources[tr.src]] = w.stat(&r.Triggers[i], 1)
	}
	w.summarizeReach(dst)
	w.summarizeQueues(dst)
	dst.At = dst.At[:0]
	dst.AtMostZero = dst.AtMostZero[:0]
	for i, lv := range w.m.levels {
		a := &r.Levels[i]
		p := 0.0
		if n := a.N(); n > 0 {
			p = a.Sum.Value() / float64(n)
		}
		s := LevelStat{Level: lv.Level, N: a.N(), P: p, StdErr: a.StdErr(w.delays)}
		if i < w.m.userLevels {
			dst.At = append(dst.At, s)
			continue
		}
		s.X, s.P = 0, 1-p
		dst.AtMostZero = append(dst.AtMostZero, s)
	}
	w.summarizeWhen(dst)
	dst.Plan = append(dst.Plan[:0], w.rows...)
}

func ensureStats(m map[string]Stat) map[string]Stat {
	if m == nil {
		return map[string]Stat{}
	}
	return m
}

// stat converts an Accum; share is Mean/total for breakdowns.
// statAt is the stat of row quantity q, its standard error marked where
// its variance is infinite.
func (w *Workspace) statAt(q int, a *Accum, total float64) Stat {
	s := w.stat(a, total)
	s.StdErrUnreliable = w.m.StdErrUnreliable(q)
	return s
}

func (w *Workspace) stat(a *Accum, total float64) Stat {
	s := Stat{N: a.N(), Mean: a.Mean(), StdErr: a.StdErr(w.delays), StdDev: a.StdDev(), Min: a.Min, Max: a.Max}
	if total != 0 && total != 1 {
		s.Share = a.Mean() / total
	}
	if a.Hist != nil {
		s.Quantiles = true
		var q [5]float64
		for i, p := range QuantileLevels() {
			var c, ci bool
			q[i], c = a.Hist.Quantile(p)
			s.Clipped = s.Clipped || c
			d := quantileZ * a.Inflation(q[i]) * math.Sqrt(p*(1-p)/float64(a.Hist.N))
			s.QLo[i], s.QHi[i], ci = a.Hist.QuantileInterval(p, d)
			s.QSE[i] = (s.QHi[i] - s.QLo[i]) / (2 * quantileZ)
			s.Clipped = s.Clipped || ci
		}
		s.P50, s.P90, s.P95, s.P99, s.P999 = q[0], q[1], q[2], q[3], q[4]
	}
	return s
}

// exceedance returns log-spaced points of P(T ≥ x) from the total's
// histogram, from the smallest non-zero total to the largest.
func (w *Workspace) exceedance(dst []ECPoint) []ECPoint {
	return w.exceedanceOf(&w.result.Total, dst)
}

// exceedanceOf is the exceedance curve of any accumulated quantity.
func (w *Workspace) exceedanceOf(a *Accum, dst []ECPoint) []ECPoint {
	h := a.Hist
	n := float64(h.N)
	if n == 0 || !(a.Max > 0) {
		return dst
	}
	// The histogram resolves nothing below its own start, so the curve
	// begins there or at the smallest value seen, whichever is higher.
	hl, _ := h.Range()
	lo := math.Max(a.Min, hl)
	if lo > a.Max {
		// An integer histogram's log range starts above its counts; the
		// curve then starts at one.
		lo = math.Min(1, a.Max)
	}
	hi := a.Max
	if hi <= lo {
		return append(dst, ecPoint(a, lo))
	}
	step := math.Log(hi/lo) / float64(exceedancePoints-1)
	// bound: exceedancePoints
	for i := 0; i < exceedancePoints; i++ {
		dst = append(dst, ecPoint(a, lo*math.Exp(float64(i)*step)))
	}
	return dst
}

// ecPoint is P(T ≥ x) with its binomial standard error, widened by the
// batch-means inflation where periods are correlated.
func ecPoint(a *Accum, x float64) ECPoint {
	h := a.Hist
	n := float64(h.N)
	p := h.Exceedance(x)
	return ECPoint{X: x, P: p, StdErr: a.Inflation(x) * math.Sqrt(p*(1-p)/n)}
}

// summarizeReach fills the proportions: reach and terminal per source and
// node, and the share of a node's entries taking each outcome.
func (w *Workspace) summarizeReach(dst *Summary) {
	r := &w.result
	m := w.m
	if dst.ReachProb == nil {
		dst.ReachProb = map[string]map[string]Prop{}
		dst.Terminal = map[string]map[string]Prop{}
	}
	if dst.OutcomeP == nil {
		dst.OutcomeP = map[string]Prop{}
	}
	for k, src := range r.Sources {
		reach, ok := dst.ReachProb[src]
		if !ok {
			reach = map[string]Prop{}
			dst.ReachProb[src] = reach
		}
		term, ok := dst.Terminal[src]
		if !ok {
			term = map[string]Prop{}
			dst.Terminal[src] = term
		}
		for v, node := range r.Nodes {
			reach[node] = prop(r.Reach[k][v], r.Events[k])
			if m.nodes[v].outStart == m.nodes[v].outEnd {
				term[node] = prop(r.Reach[k][v], r.Events[k])
			}
		}
	}
	for j, name := range r.Outcomes {
		v := m.outcomeNode[j]
		var taken, entries uint64
		for k := range r.Sources {
			taken += r.OutcomeCount[k][j]
			entries += r.Reach[k][v]
		}
		dst.OutcomeP[name] = prop(taken, entries)
	}
}

func prop(count, n uint64) Prop {
	if n == 0 {
		return Prop{P: math.NaN(), StdErr: math.NaN()}
	}
	p := float64(count) / float64(n)
	return Prop{P: p, StdErr: math.Sqrt(p * (1 - p) / float64(n))}
}

// summarizeQueues fills each queue's figures.
func (w *Workspace) summarizeQueues(dst *Summary) {
	if dst.Queues == nil {
		dst.Queues = map[string]QueueSummary{}
	}
	clear(dst.Queues)
	r := &w.result
	periods := float64(max(w.m.horizon, 1))
	for c, q := range w.m.queues {
		col := func(i int) Stat { return w.stat(&r.QueueCols[queueCols*c+i], 1) }
		s := QueueSummary{Servers: int(q.servers), Wait: r.Queues[c].Wait(), Stay: r.Queues[c].Stay(),
			Joined: col(qJoined), Served: col(qServed), Unfinished: col(qUnfinished), Length: col(qLength), Busy: col(qBusy)}
		s.LongRun = w.m.queueLong[c]
		s.LookBack = r.Queues[c].LookBack()
		s.Wait.StdErrUnreliable = w.m.QueueStdErrUnreliable(c)
		s.Stay.StdErrUnreliable = s.Wait.StdErrUnreliable
		s.Utilisation = s.Busy.Mean / (float64(q.servers) * periods)
		s.Line = s.Length.Mean / periods
		dst.Queues[w.m.nodeIDs[q.node]] = s
	}
}

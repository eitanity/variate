package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// A factor on values is exact by conditioning: given the factors' draws
// g, the model is the same model with each scaled function's law scaled
// by its g, and each rate naming the factor by g as today, whose exact
// figures the methods for a model without value factors give. Over a
// Gauss rule for each factor the means are the weighted means of the
// conditional ones and each variance E[Var | g] + Var(E | g). The
// moments given g are polynomials of degree at most 4 in g, which the
// 3-node rule integrates exactly; a threshold or a trigger on a scaled
// value is not, so the rule doubles until every figure settles.

const (
	// valueNodesStart is the first rule's nodes per factor.
	valueNodesStart = 3
	// valueNodesMax caps the nodes per factor; a figure unsettled there
	// is NaN.
	valueNodesMax = 48
	// valuePointsMax caps the conditioned models of one rule, the
	// product of every factor's nodes.
	valuePointsMax = 1 << 12
	// valueSettle is the relative change that counts as settled.
	valueSettle = 1e-9
	// valueBase is the points of the quantile measure a rule is built
	// from for a factor without a Gauss rule of its own.
	valueBase = 1 << 10
)

// valueFactors lists the factors, by index in the graph, that scale a
// function's value.
func (m *Compiled) valueFactors() []int {
	var out []int
	for i, pool := range m.factorPools {
		for _, f := range m.fns {
			if f.factor == pool {
				out = append(out, i)
				break
			}
		}
	}
	return out
}

// factorRule is an n-node Gauss rule for factor i's law: its family's
// own, or the Gauss rule of its quantile measure; a law with no more
// than n atoms is its atoms.
func (m *Compiled) factorRule(i, n int) (x, w []float64, ok bool) {
	d := m.pools[m.factorPools[i]]
	x, w = make([]float64, n), make([]float64, n)
	if d.Nodes(x, w) {
		return x, w, true
	}
	px, pw := make([]float64, valueBase), make([]float64, valueBase)
	if !d.Nodes(px, pw) {
		quantileNodes(d, px, pw)
	}
	if dist.GaussRule(px, pw, x, w) {
		return x, w, true
	}
	var ax, aw []float64
	for j, v := range pw {
		if v > 0 {
			ax, aw = append(ax, px[j]), append(aw, v)
		}
	}
	return ax, aw, len(ax) > 0 && len(ax) <= n
}

// conditionedAnalytic is the exact figures of a model with value
// factors: the rules doubling from valueNodesStart until every figure
// settles, those that do not NaN.
func (m *Compiled) conditionedAnalytic() (*Expectation, error) {
	factors := m.valueFactors()
	// A factor scales values and rates, never a holding, so every point
	// shares the holdings' rules.
	rules := newRuleCache()
	var last, prev *Expectation
	for n := valueNodesStart; n <= valueNodesMax; n *= 2 {
		e, ok, err := m.mixedAnalytic(factors, n, rules)
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		if prev != nil && settle(prev, e) {
			return e, nil
		}
		last, prev = prev, e
	}
	switch {
	case prev == nil:
		e, err := m.analytic()
		if err != nil {
			return nil, err
		}
		unsettle(e, e)
		e.pzero = m.zeroProbability()
		return e, nil
	case last == nil:
		unsettle(prev, prev)
	default:
		unsettle(last, prev)
	}
	return prev, nil
}

// mixedAnalytic combines the conditioned models' figures over the n-node
// rules of the factors; false when a rule is missing or the points
// exceed their cap.
func (m *Compiled) mixedAnalytic(factors []int, n int, rules *ruleCache) (*Expectation, bool, error) {
	xs, ws, points := m.factorRules(factors, n)
	if points == 0 {
		return nil, false, nil
	}
	idx := make([]int, len(factors))
	var acc *mixAcc
	// bound: valuePointsMax conditioned models
	for range points {
		g := m.graph
		g.Factors = append([]Factor(nil), m.graph.Factors...)
		weight := 1.0
		for j, i := range factors {
			c, err := dist.NewConstant(xs[j][idx[j]])
			if err != nil {
				return nil, false, err
			}
			g.Factors[i].Model = c
			weight *= ws[j][idx[j]]
		}
		cfg := m.cfg
		cfg.rules, cfg.profiles = rules, m.hasDelays
		e, err := conditioned(g, cfg)
		if err != nil {
			return nil, false, err
		}
		if acc == nil {
			acc = newMixAcc(e)
		}
		acc.add(e, weight)
		for j := range idx {
			if idx[j]++; idx[j] < len(xs[j]) {
				break
			}
			idx[j] = 0
		}
	}
	e := acc.result()
	e.pzero = m.zeroProbability()
	return e, true, nil
}

// factorRules is each factor's n-node rule and the points of their
// tensor; 0 points when a rule is missing, a factor's variance is not
// finite, or the points exceed their cap.
func (m *Compiled) factorRules(factors []int, n int) (xs, ws [][]float64, points int) {
	xs, ws = make([][]float64, len(factors)), make([][]float64, len(factors))
	points = 1
	for j, i := range factors {
		x, w, ok := m.factorRule(i, n)
		if !ok || !isFinite(m.pools[m.factorPools[i]].Var()) {
			return nil, nil, 0
		}
		xs[j], ws[j] = x, w
		points *= len(x)
		if points > valuePointsMax {
			return nil, nil, 0
		}
	}
	return xs, ws, points
}

// conditioned is the exact figures of the graph with its value factors
// constant.
func conditioned(g Graph, cfg Config) (*Expectation, error) {
	cfg.Transform = false
	c, _, _, err := g.flattenChecked(cfg)
	if err != nil {
		return nil, err
	}
	return c.analytic()
}

// exactFig is one exact figure: a mean and, where it has one, its
// variance, and where holding times give that variance, its key.
type exactFig struct {
	mean, vr *float64
	key      delayKey
	keyed    bool
}

// figures lists every figure of e in a fixed order, which models of the
// same graph share.
func figures(e *Expectation) []exactFig {
	all := valueFilter{kind: filterAll}
	fs := []exactFig{{mean: &e.Mean, vr: &e.Var, key: delayKey{filt: all, src: -1}, keyed: true}, {mean: &e.DrawsMean, vr: &e.DrawsVar},
		{mean: &e.StepsPerPeriod}, {mean: &e.ArrivalsPerPeriod}, {mean: &e.RoundsPerPeriod}}
	pair := func(a, b []float64, key func(i int) (delayKey, bool)) {
		for i := range a {
			f := exactFig{mean: &a[i]}
			if len(b) == len(a) {
				f.vr = &b[i]
				if key != nil {
					f.key, f.keyed = key(i)
				}
			}
			fs = append(fs, f)
		}
	}
	by := func(kind int) func(i int) (delayKey, bool) {
		return func(i int) (delayKey, bool) {
			return delayKey{filt: valueFilter{kind: kind, index: index(i)}, src: -1}, true
		}
	}
	pair(e.MeanByCategory, e.VarByCategory, by(filterCategory))
	pair(e.MeanBySource, e.VarBySource, func(i int) (delayKey, bool) { return delayKey{filt: all, src: i}, true })
	pair(e.MeanByTotal, e.VarByTotal, by(filterTotal))
	pair(e.Contrib, e.ContribVar, by(filterNode))
	pair(e.EntriesPer, nil, nil)
	pair(e.OutcomeRate, nil, nil)
	for _, rows := range [][][]float64{e.Reach, e.Terminal, e.ByPeriod} {
		for _, r := range rows {
			pair(r, nil, nil)
		}
	}
	for i := range e.Firing {
		f := &e.Firing[i]
		fs = append(fs, exactFig{mean: &f.Lo}, exactFig{mean: &f.Hi}, exactFig{mean: &f.SumMean})
	}
	return fs
}

// mixAcc sums the conditioned figures: the weighted means, and per
// variance the weighted conditional variances and second moments; and
// for a variance with holding times, the weighted mean bookings by lag
// and their squares, and the long-run variances.
type mixAcc struct {
	e          *Expectation
	m1, m2, vr []float64
	pm         [][]float64
	pq, lr     []float64
	pn, ln     []int
	n          int
}

func newMixAcc(first *Expectation) *mixAcc {
	e := cloneExpectation(first)
	n := len(figures(e))
	return &mixAcc{e: e, m1: make([]float64, n), m2: make([]float64, n), vr: make([]float64, n),
		pm: make([][]float64, n), pq: make([]float64, n), lr: make([]float64, n), pn: make([]int, n), ln: make([]int, n)}
}

func (a *mixAcc) add(e *Expectation, w float64) {
	a.n++
	for i, f := range figures(e) {
		a.m1[i] += w * *f.mean
		a.m2[i] += w * *f.mean * *f.mean
		if f.vr != nil {
			a.vr[i] += w * *f.vr
		}
		if !f.keyed {
			continue
		}
		if p, ok := e.profile[f.key]; ok {
			for len(a.pm[i]) < len(p) {
				a.pm[i] = append(a.pm[i], 0)
			}
			for d, x := range p {
				a.pm[i][d] += w * x
			}
			a.pq[i] += w * dot(p, p)
			a.pn[i]++
		}
		if lr, ok := e.longRun[f.key]; ok {
			a.lr[i] += w * lr
			a.ln[i]++
		}
	}
	for i := range e.Firing {
		if a.e.Firing[i].Reason == "" {
			a.e.Firing[i].Reason = e.Firing[i].Reason
		}
	}
}

// result is the mixture's figures: E[mean | g] and
// E[Var | g] + Var(E | g). Where holding times spread a round over
// periods, the rounds booking into one period come from many, each with
// its own draw, so Var(E | g) is taken lag by lag, Σ_d Var(M(d) | g) of
// the mean booking by lag; the long-run variance takes it whole.
func (a *mixAcc) result() *Expectation {
	var profile map[delayKey][]float64
	var longRun map[delayKey]float64
	for i, f := range figures(a.e) {
		*f.mean = a.m1[i]
		if f.vr == nil {
			continue
		}
		whole := math.Max(0, a.m2[i]-a.m1[i]*a.m1[i])
		*f.vr = a.vr[i] + whole
		if !f.keyed {
			continue
		}
		if a.pn[i] == a.n {
			*f.vr = a.vr[i] + math.Max(0, a.pq[i]-dot(a.pm[i], a.pm[i]))
			if profile == nil {
				profile = map[delayKey][]float64{}
			}
			profile[f.key] = a.pm[i]
		}
		if a.ln[i] == a.n {
			if longRun == nil {
				longRun = map[delayKey]float64{}
			}
			longRun[f.key] = a.lr[i] + whole
		}
	}
	a.e.profile, a.e.longRun = profile, longRun
	return a.e
}

// settle reports whether every figure of e is within valueSettle of
// prev's, NaN with NaN.
func settle(prev, e *Expectation) bool {
	ok := true
	pf := figures(prev)
	for i, f := range figures(e) {
		ok = figSettled(*pf[i].mean, *f.mean) && ok
		if f.vr != nil {
			ok = figSettled(*pf[i].vr, *f.vr) && ok
		}
	}
	return ok
}

// figSettled reports whether b is within valueSettle of a.
func figSettled(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.IsNaN(a) == math.IsNaN(b) && (math.IsNaN(a) || a == b)
	}
	return math.Abs(a-b) <= valueSettle*math.Max(math.Abs(b), 1e-300)
}

// unsettle sets to NaN every figure of e that differs from prev's by
// more than valueSettle; with prev = e, every figure.
func unsettle(prev, e *Expectation) {
	pf := figures(prev)
	for i, f := range figures(e) {
		if prev == e || !figSettled(*pf[i].mean, *f.mean) {
			*f.mean = math.NaN()
		}
		if f.vr != nil && (prev == e || !figSettled(*pf[i].vr, *f.vr)) {
			*f.vr = math.NaN()
		}
	}
}

// cloneExpectation copies e's slices.
func cloneExpectation(e *Expectation) *Expectation {
	c := *e
	cp := func(x []float64) []float64 { return append([]float64(nil), x...) }
	cp2 := func(x [][]float64) [][]float64 {
		if x == nil {
			return nil
		}
		out := make([][]float64, len(x))
		for i := range x {
			out[i] = cp(x[i])
		}
		return out
	}
	c.EntriesPer, c.OutcomeRate = cp(e.EntriesPer), cp(e.OutcomeRate)
	c.MeanByCategory, c.MeanBySource, c.MeanByTotal, c.Contrib = cp(e.MeanByCategory), cp(e.MeanBySource), cp(e.MeanByTotal), cp(e.Contrib)
	c.VarByCategory, c.VarBySource, c.VarByTotal, c.ContribVar = cp(e.VarByCategory), cp(e.VarBySource), cp(e.VarByTotal), cp(e.ContribVar)
	c.Reach, c.Terminal, c.ByPeriod = cp2(e.Reach), cp2(e.Terminal), cp2(e.ByPeriod)
	c.Firing = append([]Firing(nil), e.Firing...)
	return &c
}

// isFinite reports whether x is neither infinite nor NaN.
func isFinite(x float64) bool { return !math.IsInf(x, 0) && !math.IsNaN(x) }

package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
	"gonum.org/v1/gonum/dsp/fourier"
)

// The variance of a period's total with holding times. A round arriving
// at a uniform time in its period books a measure μ over the time from
// its arrival; rounds arrive as a Poisson process of rate λ a period, so
// the period's total has variance λ·E[Q], Q = ∫∫ K(x − y) μ(dx) μ(dy),
// K(δ) = (1 − |δ|)⁺ the chance two bookings δ apart land in one period.
// From node v a round books its values at 0, its rates over its holding
// [0, τ) and the rest from τ, so with A_v(t) = E[∫ K(t + y) μ_v(dy)] on
// t in [0, 1], 0 beyond, and S_v = E[Q(μ_v)]:
//
//	S_v = Σ_j p_j (E[(V + x_j)²] + E[R²]·E[G₂(τ_j)] + 2·E[V + x_j]·E[R]·E[H(τ_j)]
//	      + S_to + 2·E[V + x_j]·E[A_to(τ_j)] + 2·E[R]·E[C_to(τ_j)])
//	A_v(t) = Σ_j p_j (E[V + x_j]·K(t) + E[R]·(E[H(t + τ_j)] − H(t)) + E[A_to(t + τ_j)])
//
// H(u) = ∫₀ᵘ K, G₂(τ) = ∫∫_{[0,τ]²} K and C(τ) = ∫₀^τ A. Every
// expectation over a holding is ∫ g(t + u) dF(u) of a function g on the
// lag grid, constant past lag 1, taken cell by cell with g linear in
// each cell against the holding's CDF; the grid doubles until every
// figure settles.

const (
	// delayCellsStart and delayCellsMax are the lag grids' cells.
	delayCellsStart = 1 << 7
	delayCellsMax   = 1 << 15
	// delaySettle is the relative change that counts as settled.
	delaySettle = 1e-9
)

// lagRule is one holding's rule on an n-cell lag grid: E[g(t_i + τ)] =
// g_i·f0 + Σ_m g_{i+m}·c_m + g_n·tail, g held at g_n past the grid.
type lagRule struct {
	f0, tail float64
	c        []float64
	ePhi     []float64    // on a profile grid, E[Φ(τ − t)] at each lag t
	cum      [4][]float64 // the cells' moments summed from 0, for ePhi
	eH       []float64    // E[H(t + τ)] on the grid
	eG2      float64      // E[G₂(τ)]
	eHtau    float64      // E[H(τ)]
	fft      *fourier.FFT
	rev      []complex128 // the rule's transform, reversed, at the FFT's size
}

// lagNodes is the Gauss–Legendre rule a cell's moments take.
const (
	lagNodes = 8
	// lagFine is the cells a holding's median spans at least on the first
	// grid, so that no coarse cell holds most of a narrow law.
	lagFine = 16
	// delayBudget caps the cells the grids and their mixtures sum to.
	delayBudget = 1 << 27
)

// newLagRule builds law d's rule on n cells from each cell's moments
// ∫ u^k dF, k ≤ 3, a constant law's exactly. The interpolation weights
// take the first two; E[H(t + τ)] and E[G₂(τ)], polynomials on each
// cell, take all four exactly.
//
// grid is the law's CDF and survival on the grid's points, or nil to
// evaluate them.
func newLagRule(d dist.Distribution, n int, grid *lawGrid) lagRule {
	return lagRuleWith(d, n, func(mu *[4][]float64) (float64, float64) {
		if grid == nil {
			g := baseGrid(d, n)
			grid = &g
		}
		return cellMoments(grid, n, mu)
	})
}

// lagRuleWith is law d's rule on n cells, the cells' moments of a law
// that is not constant from moments.
func lagRuleWith(d dist.Distribution, n int, moments func(mu *[4][]float64) (f0, tail float64)) lagRule {
	h := 1 / float64(n)
	r := lagRule{c: make([]float64, n+1)}
	var mu [4][]float64
	for k := range mu {
		mu[k] = make([]float64, n)
	}
	lo, hi := d.Support()
	if lo == hi {
		switch {
		case lo <= 0:
			r.f0 = 1
		case lo >= 1:
			r.tail = 1
		default:
			j := min(n-1, int(lo*float64(n)))
			for k := range mu {
				mu[k][j] = math.Pow(lo, float64(k))
			}
		}
	} else {
		r.f0, r.tail = moments(&mu)
	}
	// bound: n cells
	for m := range n + 1 {
		if m < n {
			r.c[m] += mu[0][m] - (mu[1][m]-float64(m)*h*mu[0][m])/h
		}
		if m >= 1 {
			r.c[m] += (mu[1][m-1] - float64(m-1)*h*mu[0][m-1]) / h
		}
	}
	var cum [4][]float64
	for k := range cum {
		cum[k] = make([]float64, n+1)
		// bound: n cells
		for j := range n {
			cum[k][j+1] = cum[k][j] + mu[k][j]
		}
	}
	r.eH = make([]float64, n+1)
	for i := range r.eH {
		t, below := float64(i)*h, n-i
		in := r.f0 + cum[0][below]
		r.eH[i] = (t-t*t/2)*in + (1-t)*cum[1][below] - cum[2][below]/2 + (1-in)/2
	}
	r.eHtau = r.eH[0]
	r.eG2 = cum[2][n] - cum[3][n]/3 + math.Max(0, d.Mean()-cum[1][n]) - r.tail/3
	r.cum = cum
	return r
}

// lawGrid is a law's CDF and survival at the lag grid's cell bounds
// and at each cell's Gauss–Legendre points.
type lawGrid struct {
	fb, sb []float64 // n+1 bounds
	fq, sq []float64 // lagNodes a cell
	// Or, for a law given its group's normal, the base law's scores on
	// the grid, the CDF and survival Φ(±(z − a)/b) taken from them.
	zb, zq []float64
	a, b   float64
}

// bound is the CDF and survival at the grid's i-th bound.
func (g *lawGrid) bound(i int) (float64, float64) {
	if g.zb != nil {
		return normalPair((g.zb[i] - g.a) / g.b)
	}
	return g.fb[i], g.sb[i]
}

// point is the CDF and survival at the grid's i-th Gauss–Legendre point.
func (g *lawGrid) point(i int) (float64, float64) {
	if g.zq != nil {
		return normalPair((g.zq[i] - g.a) / g.b)
	}
	return g.fq[i], g.sq[i]
}

// gridPoints calls at for every point of an n-cell grid: the bounds,
// then each cell's Gauss–Legendre points.
func gridPoints(n int, at func(bound bool, i int, u float64)) {
	h := 1 / float64(n)
	gx, _ := legendre()
	// bound: n + 1 bounds
	for i := range n + 1 {
		at(true, i, float64(i)*h)
	}
	// bound: n cells
	for j := range n {
		for q, x := range gx {
			at(false, j*lagNodes+q, (float64(j)+x)*h)
		}
	}
}

// legendre is the lagNodes-point Gauss–Legendre rule on [0, 1].
func legendre() ([]float64, []float64) {
	gx, gw := make([]float64, lagNodes), make([]float64, lagNodes)
	dist.LegendreRule(gx, gw)
	return gx, gw
}

// newLawGrid allocates an n-cell grid.
func newLawGrid(n int) lawGrid {
	return lawGrid{fb: make([]float64, n+1), sb: make([]float64, n+1), fq: make([]float64, n*lagNodes), sq: make([]float64, n*lagNodes)}
}

// lawNegligible is the survival past which a law's grid is taken as
// exhausted: its cells hold nothing and its CDF is 1.
const lawNegligible = 1e-15

// baseGrid evaluates law d on the grid, up to the bound where its
// survival is lawNegligible, past which every cell is exhausted.
func baseGrid(d dist.Distribution, n int) lawGrid {
	g := newLawGrid(n)
	last := n
	h := 1 / float64(n)
	// bound: n + 1 bounds
	for i := range n + 1 {
		u := float64(i) * h
		if i > last {
			g.fb[i], g.sb[i] = 1, 0
			continue
		}
		g.fb[i], g.sb[i] = d.CDF(u), d.Survival(u)
		if g.sb[i] < lawNegligible && last == n {
			last = i
		}
	}
	gx, _ := legendre()
	// bound: n cells
	for j := range n {
		for q, x := range gx {
			k := j*lagNodes + q
			if j >= last {
				g.fq[k], g.sq[k] = 1, 0
				continue
			}
			u := (float64(j) + x) * h
			g.fq[k], g.sq[k] = d.CDF(u), d.Survival(u)
		}
	}
	return g
}

// scoreGrid is law d's normal scores Φ⁻¹(F(u)) on the grid, from the
// survival above the median.
func scoreGrid(d dist.Distribution, n int) (zb, zq []float64) {
	zb, zq = make([]float64, n+1), make([]float64, n*lagNodes)
	gridPoints(n, func(bound bool, i int, u float64) {
		f := d.CDF(u)
		z := stdNormalQuantile(f)
		if f > 0.5 {
			z = -stdNormalQuantile(d.Survival(u))
		}
		if bound {
			zb[i] = z
		} else {
			zq[i] = z
		}
	})
	return zb, zq
}

// normalPair is Φ(z) and 1 − Φ(z), the smaller from the tail.
func normalPair(z float64) (float64, float64) {
	if z < 0 {
		f := stdNormalCDF(z)
		return f, 1 - f
	}
	s := stdNormalCDF(-z)
	return 1 - s, s
}

// cellMoments fills each cell's moments from the law's grid, giving its
// mass at 0 and past 1. A cell's ∫ u^k dF is b^k·F(b) − a^k·F(a) −
// k·∫ u^(k−1)·F, by parts, with the survival S in place of F above the
// median so that nothing cancels: F is bounded and smooth, and the
// Gauss–Legendre rule on each cell integrates it closely once the cells
// are narrow beside the law, as the first grid makes them.
func cellMoments(g *lawGrid, n int, mu *[4][]float64) (f0, tail float64) {
	h := 1 / float64(n)
	gx, gw := legendre()
	// bound: n cells
	for j := range n {
		a := float64(j) * h
		b := a + h
		fa, sa := g.bound(j)
		fb, sb := g.bound(j + 1)
		upper := fa > 0.5
		var in [3]float64 // ∫ u^(k−1)·F, or ·S above the median, k = 1..3
		for q := range gx {
			u := a + gx[q]*h
			v, sv := g.point(j*lagNodes + q)
			if upper {
				v = sv
			}
			wu := gw[q] * h * v
			for k := range in {
				in[k] += wu
				wu *= u
			}
		}
		if upper {
			mu[0][j] = sa - sb
		} else {
			mu[0][j] = fb - fa
		}
		ak, bk := a, b
		for k := 1; k < 4; k++ {
			if upper {
				mu[k][j] = ak*sa - bk*sb + float64(k)*in[k-1]
			} else {
				mu[k][j] = bk*fb - ak*fa - float64(k)*in[k-1]
			}
			ak, bk = ak*a, bk*b
		}
	}
	f0, _ = g.bound(0)
	_, tail = g.bound(n)
	return f0, tail
}

// at0 is E[g(τ)].
func (r *lagRule) at0(g []float64) float64 {
	n := len(g) - 1
	s := g[0]*r.f0 + g[n]*r.tail
	for m, c := range r.c {
		s += c * g[m]
	}
	return s
}

// apply is E[g(t_i + τ)] for every grid lag, by a real FFT of the
// extended g correlated with the rule.
func (r *lagRule) apply(g []float64) []float64 {
	n := len(g) - 1
	size := 1
	// bound: doublings to 3n + 2, past which the circular correlation of
	// g, extended to 2n, with the rule's n + 1 weights does not wrap
	for size < 3*n+2 {
		size *= 2
	}
	ext := make([]float64, size)
	copy(ext, g)
	// bound: the grid's extension
	for k := n + 1; k <= 2*n; k++ {
		ext[k] = g[n]
	}
	if r.fft == nil {
		rev := make([]float64, size)
		for m, c := range r.c {
			rev[(size-m)%size] = c
		}
		r.fft = fourier.NewFFT(size)
		r.rev = r.fft.Coefficients(nil, rev)
	}
	a := r.fft.Coefficients(nil, ext)
	for i := range a {
		a[i] *= r.rev[i]
	}
	conv := r.fft.Sequence(nil, a)
	out := make([]float64, n+1)
	for i := range out {
		out[i] = conv[i]/float64(size) + g[i]*r.f0 + g[n]*r.tail
	}
	return out
}

// delayExact reports whether the variance with holding times is
// computed: Poisson sources, their rates fixed, drawn once a period or
// on a schedule whose joint cycle is within its cap, and no repair,
// trigger, queue or outage holding.
func (m *Compiled) delayExact() bool {
	if m.hasRepair || m.renewal || m.HasTriggers() || len(m.queues) > 0 || m.cycle.over {
		return false
	}
	for _, s := range m.srcs {
		if s.trigger != noIndex || s.every > 1 || s.chainEnd > s.chainStart {
			return false
		}
	}
	for _, o := range m.outs {
		if o.outage {
			return false
		}
	}
	return true
}

// delayStart is the first grid: delayCellsStart cells, or more so that
// the median of each holding in a group spans lagFine of them, its law
// given the group's normal taken on the grid; another holding's cells
// take their moments on sub-cells as narrow as it is, and a fixed one is
// one point, which no cell need resolve.
func (m *Compiled) delayStart() int {
	n := delayCellsStart
	member := map[int32]bool{}
	for _, g := range m.groups {
		for _, mb := range g.members {
			if mb.fn == noIndex {
				member[mb.out] = true
			}
		}
	}
	for j, o := range m.outs {
		if o.hold == noIndex || !member[index(j)] {
			continue
		}
		if lo, hi := m.pools[o.hold].Support(); lo == hi {
			continue
		}
		med := m.pools[o.hold].Quantile(0.5)
		// bound: doublings to delayCellsMax
		for n < delayCellsMax && med > 0 && med*float64(n) < lagFine {
			n *= 2
		}
	}
	return n
}

// delayRules is each outcome's lag rule on n cells, nil without a
// holding.
func (m *Compiled) delayRules(n int) []*lagRule {
	out := make([]*lagRule, len(m.outs))
	for j, o := range m.outs {
		if o.hold == noIndex {
			continue
		}
		key := ruleKey{pool: o.hold, n: n}
		var grid *lawGrid
		if c, ok := m.condOf[o.hold]; ok && c.b > 0 {
			zb, zq := c.scores.at(c.base, o.hold, n)
			grid = &lawGrid{zb: zb, zq: zq, a: c.a, b: c.b}
			key.a, key.b, key.member = c.a, c.b, true
		} else if ok {
			key.member, key.lockstepAt = true, c.a
		}
		d := m.pools[o.hold]
		if lo, hi := d.Support(); lo == hi {
			key.fixed, key.at = true, lo
		}
		out[j] = m.cfg.rules.rule(key, func() lagRule {
			if grid != nil || key.member {
				return newLagRule(d, n, grid)
			}
			return lagRuleWith(d, n, func(mu *[4][]float64) (float64, float64) { return subMoments(d, n, mu) })
		})
	}
	return out
}

// delayScratch holds one grid's working arrays, reused by every pass
// on it: the kernel K and H on the grid, and per node its S, A, ∫A and
// whether its A is 0.
type delayScratch struct {
	kg, hg, s []float64
	a, c      [][]float64
	zero      []bool
}

// newDelayScratch is the arrays for nodes on an n-cell grid.
func newDelayScratch(nodes, n int) *delayScratch {
	h := 1 / float64(n)
	sc := &delayScratch{kg: make([]float64, n+1), hg: make([]float64, n+1), s: make([]float64, nodes),
		a: make([][]float64, nodes), c: make([][]float64, nodes), zero: make([]bool, nodes)}
	for i := range sc.kg {
		t := float64(i) * h
		sc.kg[i], sc.hg[i] = 1-t, t-t*t/2
	}
	for v := range sc.a {
		sc.a[v], sc.c[v] = make([]float64, n+1), make([]float64, n+1)
	}
	return sc
}

// delaySecond is S_v, E[Q] of the round from each node under the
// filter, on the rules' grid of n cells, in the scratch's arrays.
func (m *Compiled) delaySecond(filt valueFilter, rules []*lagRule, n int, sc *delayScratch) []float64 {
	h := 1 / float64(n)
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		nm := m.nodeMomentsFor(v, filt)
		nd := m.nodes[v]
		av := sc.a[v]
		clear(av)
		sc.s[v] = 0
		if nd.outStart == nd.outEnd {
			sc.s[v] = nm.a2
			for k := range av {
				av[k] = nm.a * sc.kg[k]
			}
		}
		// bound: outcomes of one node
		for j := nd.outStart; j < nd.outEnd; j++ {
			m.delayBranch(nm, j, filt, rules[j], sc, &sc.s[v], av)
		}
		sc.zero[v] = allZero(av)
		if !sc.zero[v] {
			cumulative(av, h, sc.c[v])
		}
	}
	return sc.s
}

// delayBranch adds outcome j's term to node v's S and A.
func (m *Compiled) delayBranch(nm nodeMoments, j int32, filt valueFilter, r *lagRule, sc *delayScratch, sv *float64, av []float64) {
	o := m.outs[j]
	p := m.outcomeMeanProb(m.outcomeNode[j], j)
	x, x2 := m.outcomeFn(j, filt)
	to := o.to
	kg, hg, s := sc.kg, sc.hg, sc.s
	eAt, eG2, eHtau, eC, eH := sc.a[to], 0.0, 0.0, 0.0, hg
	if r != nil {
		eG2, eHtau, eH = r.eG2, r.eHtau, r.eH
		if !sc.zero[to] {
			eAt, eC = r.apply(sc.a[to]), r.at0(sc.c[to])
		}
	}
	vx := nm.a + x
	v2 := nm.a2 + 2*nm.a*x + x2
	*sv += weighted(p, v2+rateSecond(nm.rho2, eG2)+2*vx*nm.rho*eHtau+s[to]+2*vx*eAt[0]+2*nm.rho*eC)
	for k := range av {
		av[k] += weighted(p, vx*kg[k]+nm.rho*(eH[k]-hg[k])+eAt[k])
	}
}

// allZero reports whether every value of g is 0.
func allZero(g []float64) bool {
	for _, x := range g {
		if x != 0 {
			return false
		}
	}
	return true
}

// cumulative fills out with ∫₀^t g on the grid by the trapezoidal rule.
func cumulative(g []float64, h float64, out []float64) {
	out[0] = 0
	// bound: the grid
	for k := 1; k < len(g); k++ {
		out[k] = out[k-1] + h*(g[k-1]+g[k])/2
	}
}

// delayKey is one variance the batch computes: a filter, over every
// source or, from 0, one.
type delayKey struct {
	filt valueFilter
	src  int
}

// delayFigures is every variance an Analytic call reports, with holding
// times, computed together so that the conditioned models and the
// holdings' rules are built once for all of them, and beside each the
// long-run variance per period the plan sizes its runs by. With drawn
// rates, or when profiles is set for a mixture over draws once a
// period, it adds the between-period term and gives each key's mean
// booking by lag; profiles is nil otherwise.
func (m *Compiled) delayFigures(lam []float64) (perPeriod, longRun map[delayKey]float64, profiles map[delayKey][]float64) {
	keys := m.delayKeys()
	perPeriod, longRun = make(map[delayKey]float64, len(keys)), make(map[delayKey]float64, len(keys))
	nan := func() (map[delayKey]float64, map[delayKey]float64, map[delayKey][]float64) {
		for _, k := range keys {
			perPeriod[k], longRun[k] = math.NaN(), math.NaN()
		}
		return perPeriod, longRun, nil
	}
	draws := m.periodDraws()
	cuts, ok := m.delayDraws(draws)
	if !ok {
		return nan()
	}
	models, weights := []*Compiled{m}, []float64{1}
	if len(m.groups) > 0 {
		var ok bool
		if models, weights, ok = m.depRule(keys, lam); !ok {
			return nan()
		}
	}
	start := m.delayStart()
	vs := settleGrids(start, models, weights, keys, lam)
	var bs []float64
	var profs [][]float64
	long := make([]float64, len(keys))
	switch {
	case len(draws) > 0:
		var ok bool
		if bs, profs, long, ok = m.drawMixture(draws, keys, lam, vs); !ok {
			return nan()
		}
	case m.mixedRates() || m.cfg.profiles || len(cuts) > 0:
		var ok bool
		if bs, profs, long, ok = m.ownBetween(models, weights, keys, lam, vs); !ok {
			return nan()
		}
	}
	if bs, ok = m.scheduleBetween(len(draws) > 0, models, weights, keys, lam, vs, bs); !ok {
		return nan()
	}
	if profs != nil {
		profiles = make(map[delayKey][]float64, len(keys))
	}
	for i, k := range keys {
		perPeriod[k] = vs[i]
		if bs != nil {
			perPeriod[k] += bs[i]
			profiles[k] = profs[i]
		}
		longRun[k] = long[i]
		for j, c := range models {
			longRun[k] += weights[j] * c.roundSecond(k, lam)
		}
	}
	return perPeriod, longRun, profiles
}

// delayDraws is the outcomes whose holding is drawn once a period;
// false where the variance with holding times is not computed: a path
// through two of them, values or probabilities drawn once a period with
// groups, or past the points' cap.
func (m *Compiled) delayDraws(draws []int32) ([]int32, bool) {
	cuts, ok := m.periodCuts()
	if !ok || len(draws) > 0 && len(m.groups) > 0 {
		return nil, false
	}
	_, _, _, ok = m.periodRules(draws)
	return cuts, ok
}

// ownBetween is the between-period term, the mean booking by lag and the
// long-run variance's share beyond λ·E[B²] for each key of a model
// without values or probabilities drawn once a period, its rounds' means
// mixed over the groups' normals; false past the span's bound.
func (m *Compiled) ownBetween(models []*Compiled, weights []float64, keys []delayKey, lam, within []float64) ([]float64, [][]float64, []float64, bool) {
	l := m.lagSpan()
	if !(l < float64(delayBudget)) {
		return nil, nil, nil, false
	}
	bs, profs, _ := settleBetween(int(math.Ceil(l)), models, weights, keys, lam, within, lagBudget)
	long := make([]float64, len(keys))
	for i, k := range keys {
		m1 := make([]float64, len(m.nodes))
		for j, c := range models {
			c1, _ := c.valueToGo(k.filt)
			for v := range m1 {
				m1[v] += weights[j] * c1[v]
			}
		}
		long[i] = m.rateLong(k, m1) + m.cutLong(k)
	}
	return bs, profs, long, true
}

// pointBetween is each key's between-period term on the n-cell grid,
// extrapolated from it and the grid of half as many cells, and its
// mean booking by lag on the n-cell grid.
func pointBetween(c *Compiled, keys []delayKey, lam []float64, n, span int) ([]float64, [][]float64, bool) {
	done := make([]bool, len(keys))
	half, v := make([]float64, len(keys)), make([]float64, len(keys))
	if _, ok := betweenAt([]*Compiled{c}, []float64{1}, keys, done, lam, n/2, span, half); !ok {
		return nil, nil, false
	}
	at, ok := betweenAt([]*Compiled{c}, []float64{1}, keys, done, lam, n, span, v)
	if !ok {
		return nil, nil, false
	}
	for k := range v {
		v[k] += (v[k] - half[k]) / 3
	}
	return v, at, true
}

// drawMixture is the between-period term, the mean booking by lag and
// the long-run variance's share beyond λ·E[B²] for each key, mixed over
// the points of the values and probabilities drawn once a period: at
// each point the model with them fixed gives its own between-period
// term, and the points add Σ_d Var(M(d)) of their mean bookings by lag,
// and to the long-run variance the variance of the period's whole mean.
// A round's within-period term is linear in each draw's moments, so the
// model's own, from their laws, is the mixture's. False when the points
// pass their cap or a point's term does not settle.
func (m *Compiled) drawMixture(draws []int32, keys []delayKey, lam []float64, within []float64) (bs []float64, profs [][]float64, long []float64, ok bool) {
	xs, ws, points, ok := m.periodRules(draws)
	if !ok {
		return nil, nil, nil, false
	}
	rules := m.cfg.rules
	if rules == nil {
		rules = newRuleCache()
	}
	l := m.lagSpan()
	if !(l < float64(delayBudget)) {
		return nil, nil, nil, false
	}
	span := int(math.Ceil(l))
	// The model as it is settles the grid, its draws at their means, where
	// a mean booking by lag multilinear in them is their mean; each point
	// takes that grid and the one of half as many cells, extrapolated.
	mm := *m
	mm.cfg.rules = rules
	settled, _, grid := settleBetween(span, []*Compiled{&mm}, []float64{1}, keys, lam, within, lagBudget/(points+1))
	for _, b := range settled {
		if math.IsNaN(b) {
			return nil, nil, nil, false
		}
	}
	if points*profileWork(m, grid*(span+1)*3/2, keys, make([]bool, len(keys))) > lagBudget {
		return nil, nil, nil, false
	}
	acc := newPointMix(len(keys))
	idx := make([]int, len(draws))
	at := make([]float64, len(draws))
	// bound: periodPointsMax points
	for range points {
		w := 1.0
		for i := range draws {
			at[i], w = xs[i][idx[i]], w*ws[i][idx[i]]
		}
		c, err := m.fixedAt(draws, at)
		if err != nil {
			return nil, nil, nil, false
		}
		c.cfg.rules = rules
		b, pr, ok := pointBetween(c, keys, lam, grid, span)
		if !ok || !acc.add(c, keys, lam, w, b, pr) {
			return nil, nil, nil, false
		}
		for i := range idx {
			if idx[i]++; idx[i] < len(xs[i]) {
				break
			}
			idx[i] = 0
		}
	}
	bs, profs, long = acc.result()
	return bs, profs, long, true
}

// pointMix sums a mixture's points for each key: the weighted
// between-period terms, mean bookings by lag and their squares, the
// rates' and holdings' long-run shares, and the period's whole mean and
// its square.
type pointMix struct {
	bs, sq, long, mw, mq []float64
	profs                [][]float64
}

func newPointMix(n int) *pointMix {
	return &pointMix{bs: make([]float64, n), sq: make([]float64, n), long: make([]float64, n),
		mw: make([]float64, n), mq: make([]float64, n), profs: make([][]float64, n)}
}

// add takes point c of weight w, its terms b and profiles pr; false when
// a term is NaN.
func (a *pointMix) add(c *Compiled, keys []delayKey, lam []float64, w float64, b []float64, pr [][]float64) bool {
	for k, key := range keys {
		if math.IsNaN(b[k]) {
			return false
		}
		a.bs[k] += w * b[k]
		for len(a.profs[k]) < len(pr[k]) {
			a.profs[k] = append(a.profs[k], 0)
		}
		for d, x := range pr[k] {
			a.profs[k][d] += w * x
		}
		a.sq[k] += w * dot(pr[k], pr[k])
		c1, _ := c.valueToGo(key.filt)
		mean := 0.0
		for s, src := range c.srcs {
			if key.src < 0 || key.src == s {
				mean += lam[s] * c1[src.to]
			}
		}
		a.long[k] += w * (c.rateLong(key, c1) + c.cutLong(key))
		a.mw[k], a.mq[k] = a.mw[k]+w*mean, a.mq[k]+w*mean*mean
	}
	return true
}

// result is each key's between-period term with Σ_d Var(M(d)) over the
// points, its mean booking by lag, and its long-run share with the
// variance of the period's whole mean.
func (a *pointMix) result() (bs []float64, profs [][]float64, long []float64) {
	for k := range a.bs {
		a.bs[k] += math.Max(0, a.sq[k]-dot(a.profs[k], a.profs[k]))
		a.long[k] += math.Max(0, a.mq[k]-a.mw[k]*a.mw[k])
	}
	return a.bs, a.profs, a.long
}

// delayKeys is every variance the batch computes.
func (m *Compiled) delayKeys() []delayKey {
	keys := []delayKey{{filt: valueFilter{kind: filterAll}, src: -1}}
	for k := range m.srcs {
		keys = append(keys, delayKey{filt: valueFilter{kind: filterAll}, src: k})
	}
	for c := range m.categories {
		keys = append(keys, delayKey{filt: valueFilter{kind: filterCategory, index: index(c)}, src: -1})
	}
	for v := range m.nodes {
		if m.nodes[v].hasFunctions() {
			keys = append(keys, delayKey{filt: valueFilter{kind: filterNode, index: index(v)}, src: -1})
		}
	}
	for i := range m.totals {
		keys = append(keys, delayKey{filt: valueFilter{kind: filterTotal, index: index(i)}, src: -1})
	}
	return keys
}

// roundSecond is Σ_k λ_k·E[B²] of source k's rounds under the key, B a
// round's whole booking: the long-run variance per period of a mean
// over many periods, which a round's holding spreads over neighbouring
// periods without changing.
func (m *Compiled) roundSecond(key delayKey, lam []float64) float64 {
	_, m2 := m.valueToGo(key.filt)
	v := 0.0
	for k, src := range m.srcs {
		if key.src < 0 || key.src == k {
			v += weighted(lam[k], m2[src.to])
		}
	}
	return v
}

// cutLong is the holdings drawn once a period's share of the long-run
// variance per period: a rate booked over such a holding books ρ·τ for
// every round of the period that takes it, so the period's whole
// booking moves with τ, by Var(τ)·(ρ·p)²·Σ_{k,l} E[Λ_k Λ_l]·r_k·r_l, r_k
// the chance a round of source k reaches the holding's node.
func (m *Compiled) cutLong(key delayKey) float64 {
	cuts, _ := m.periodCuts()
	v := 0.0
	for _, j := range cuts {
		node := m.outcomeNode[j]
		rho := m.nodeMomentsFor(node, key.filt).rho
		if rho == 0 {
			continue
		}
		b := rho * m.outcomeMeanProb(node, j)
		pair := 0.0
		for k := range m.srcs {
			for l := range m.srcs {
				if key.src >= 0 && (key.src != k || key.src != l) {
					continue
				}
				pair += (m.rateExcess(k, l) + m.countMean[k]*m.countMean[l]) * m.reach[k][node] * m.reach[l][node]
			}
		}
		v += m.pools[m.outs[j].hold].Var() * b * b * pair
	}
	return v
}

// rateLong is the drawn rates' share of the long-run variance per
// period, Σ_{k,l} (E[Λ_k Λ_l] − λ_k λ_l)·E[B_k]·E[B_l], from each
// node's mean booking m1.
func (m *Compiled) rateLong(key delayKey, m1 []float64) float64 {
	v := 0.0
	for k, s := range m.srcs {
		for l, t := range m.srcs {
			if key.src >= 0 && (key.src != k || key.src != l) {
				continue
			}
			if c := m.rateExcess(k, l); c != 0 {
				v += c * m1[s.to] * m1[t.to]
			}
		}
	}
	return v
}

// depRule chooses the rule over the groups' normals: doubled on a fixed
// grid until every key's mixture settles, the grid's error being smooth
// in the normals. A point of little weight then counts by its weight
// when the grid doubles for the mixture as a whole.
func (m *Compiled) depRule(keys []delayKey, lam []float64) ([]*Compiled, []float64, bool) {
	var prev []float64
	scores := &scoreCache{z: map[[2]int][2][]float64{}}
	if m.cfg.rules != nil {
		scores = m.cfg.rules.scores
	}
	// bound: the rules from depNodesStart to depMixMax
	for n := depNodesStart; n <= depMixMax; n *= 2 {
		cs, ws, ok := m.depPoints(n, scores)
		if !ok {
			return nil, nil, false
		}
		v := mixAt(cs, ws, keys, lam, m.delayStart())
		if prev != nil && allSettled(prev, v, valueSettle) {
			return cs, ws, true
		}
		prev = v
	}
	return nil, nil, false
}

// mixAt is each key's weighted sum over the models on an n-cell grid.
func mixAt(cs []*Compiled, ws []float64, keys []delayKey, lam []float64, n int) []float64 {
	out := make([]float64, len(keys))
	sc := newDelayScratch(len(cs[0].nodes), n)
	for i, c := range cs {
		rules := c.delayRules(n)
		for k, key := range keys {
			out[k] += ws[i] * c.delayAt(key, lam, rules, n, sc)
		}
	}
	return out
}

// settleGrids doubles the grid from delayCellsStart until each key's
// figure settles: two grids' figures agree, or two extrapolations do,
// the error falling as the square of the cell so that a doubling's
// figure v becomes v + (v − v_prev)/3. A key unsettled by delayCellsMax
// is NaN.
func settleGrids(start int, cs []*Compiled, ws []float64, keys []delayKey, lam []float64) []float64 {
	st := make([]gridSettle, len(keys))
	out := make([]float64, len(keys))
	for k := range keys {
		st[k] = gridSettle{prev: math.NaN(), prevR: math.NaN(), dv: math.NaN(), dr: math.NaN()}
		out[k] = math.NaN()
	}
	left, work := len(keys), 0
	// bound: the grids from the first to delayCellsMax
	for n := start; n <= delayCellsMax && left > 0; n *= 2 {
		if work += n * len(cs) * left; work > delayBudget {
			break
		}
		v := make([]float64, len(keys))
		sc := newDelayScratch(len(cs[0].nodes), n)
		// bound: the models of the mixture
		for i, c := range cs {
			rules := c.delayRules(n)
			for k, key := range keys {
				if !st[k].done {
					v[k] += ws[i] * c.delayAt(key, lam, rules, n, sc)
				}
			}
		}
		for k := range keys {
			switch {
			case st[k].done:
			case st[k].step(v[k]):
				out[k] = st[k].value
				left--
			case st[k].hopeless(n):
				st[k].done = true
				left--
			}
		}
	}
	return out
}

// settleBetween doubles the profile grid from profileStart until each
// key's between-period term settles beside its whole figure, within
// its within-period part, within the budget of work: two grids' terms, or two extrapolations, agree
// to delaySettle of within + between. It returns the terms and each
// key's mean booking by lag on the grid it settled on; a key unsettled
// by lagCellsMax, or whose within-period part is NaN, is NaN.
func settleBetween(span int, cs []*Compiled, ws []float64, keys []delayKey, lam, within []float64, budget int) ([]float64, [][]float64, int) {
	return settleTerm(span, cs, ws, keys, lam, within, budget, betweenAt)
}

// profileTerm adds each unsettled key's term on an n-cell a period
// profile grid to v, and may return its mean booking by lag.
type profileTerm func(cs []*Compiled, ws []float64, keys []delayKey, done []bool, lam []float64, n, span int, v []float64) ([][]float64, bool)

// settleTerm is settleBetween for any term on the profile grids.
func settleTerm(span int, cs []*Compiled, ws []float64, keys []delayKey, lam, within []float64, budget int, term profileTerm) ([]float64, [][]float64, int) {
	out := make([]float64, len(keys))
	profs := make([][]float64, len(keys))
	done := make([]bool, len(keys))
	st := make([]gridSettle, len(keys))
	left := 0
	for k := range keys {
		out[k] = math.NaN()
		st[k] = gridSettle{prev: math.NaN(), prevR: math.NaN(), dv: math.NaN(), dr: math.NaN()}
		if done[k] = math.IsNaN(within[k]); !done[k] {
			left++
		}
	}
	work, reached := 0, 0
	// bound: the profile grids from the first to lagCellsMax
	for n := profileStart(delayCellsStart, span); n*(span+1) <= lagCellsMax && left > 0; n *= 2 {
		reached = n
		if work += profileWork(cs[0], n*(span+1), keys, done) * len(cs); work > budget {
			break
		}
		v := make([]float64, len(keys))
		at, ok := term(cs, ws, keys, done, lam, n, span, v)
		if !ok {
			break
		}
		for k := range keys {
			if !done[k] && settleKey(&st[k], v[k], within[k], n*(span+1)*delayCellsMax/lagCellsMax, &out[k]) {
				done[k] = true
				left--
				if at != nil {
					profs[k] = at[k]
				}
			}
		}
	}
	return out, profs, reached
}

// settleKey steps one key's term v, settled beside the whole figure,
// the within-period part standing in the comparison as a constant: true
// once it settles, out its term, or cannot settle from a grid of the
// given equivalent cells, out left NaN.
func settleKey(st *gridSettle, v, within float64, cells int, out *float64) bool {
	if st.step(v + within) {
		*out = st.value - within
		return true
	}
	return st.hopeless(cells)
}

// profileWork is the work of one profile grid of the given cells: for
// each filter the unsettled keys take, a pass over the outcomes and, per
// cut, a pass for each source's part and each pair's product.
func profileWork(m *Compiled, cells int, keys []delayKey, done []bool) int {
	cuts, _ := m.periodCuts()
	return cells * (len(m.outs) + 1 + len(cuts)*len(m.srcs)*(len(m.srcs)+3)/2) * len(unsettledFilters(keys, done))
}

// betweenAt adds each unsettled key's between-period term on an n-cell
// a period profile grid to v, and returns its mean booking by lag: the
// sources' profiles mixed over the models, the groups' normals being
// drawn per round.
func betweenAt(cs []*Compiled, ws []float64, keys []delayKey, done []bool, lam []float64, n, span int, v []float64) ([][]float64, bool) {
	g := newLagGrid(n, span)
	byFilt, cutRules, ok := mixProfiles(cs, ws, keys, done, &g)
	if !ok {
		return nil, false
	}
	at := make([][]float64, len(keys))
	covs := map[valueFilter]*cutCovs{}
	for k, key := range keys {
		if done[k] {
			continue
		}
		acc := byFilt[key.filt]
		cc := covs[key.filt]
		if cc == nil {
			cc = newCutCovs(&g, cutRules, acc.through)
			covs[key.filt] = cc
		}
		b, prof := cs[0].between(key, acc.mean, cc, lam)
		v[k] += b
		at[k] = prof
	}
	return at, true
}

// mixedProfiles is one filter's sources' mean profiles and, per cut,
// their parts through it, mixed over the models.
type mixedProfiles struct {
	mean    [][]float64
	through [][][]float64
}

// add adds w times one model's profiles.
func (p *mixedProfiles) add(mean [][]float64, through [][][]float64, w float64) {
	if p.mean == nil {
		p.mean = make([][]float64, len(mean))
	}
	addRows(p.mean, mean, w)
	for len(p.through) < len(through) {
		p.through = append(p.through, nil)
	}
	for t, rows := range through {
		if p.through[t] == nil {
			p.through[t] = make([][]float64, len(rows))
		}
		addRows(p.through[t], rows, w)
	}
}

// mixProfiles is each unsettled key's filter's profiles mixed over the
// models, and the cuts' rules, the first model's.
func mixProfiles(cs []*Compiled, ws []float64, keys []delayKey, done []bool, g *lagGrid) (map[valueFilter]*mixedProfiles, []*lagRule, bool) {
	cuts, ok := cs[0].periodCuts()
	if !ok {
		return nil, nil, false
	}
	byFilt := map[valueFilter]*mixedProfiles{}
	var cutRules []*lagRule
	// bound: the models of the mixture
	for i, c := range cs {
		rules, ok := c.lagRules(g)
		if !ok {
			return nil, nil, false
		}
		if i == 0 {
			for _, j := range cuts {
				cutRules = append(cutRules, rules[j])
			}
		}
		// unordered: each filter's profiles are its own
		for f := range unsettledFilters(keys, done) {
			mean, through := c.sourceProfiles(f, rules, g, cuts)
			acc := byFilt[f]
			if acc == nil {
				acc = &mixedProfiles{}
				byFilt[f] = acc
			}
			acc.add(mean, through, ws[i])
		}
	}
	return byFilt, cutRules, true
}

// unsettledFilters is the filters the unsettled keys take.
func unsettledFilters(keys []delayKey, done []bool) map[valueFilter]bool {
	filters := map[valueFilter]bool{}
	for k, key := range keys {
		if !done[k] {
			filters[key.filt] = true
		}
	}
	return filters
}

// addRows adds w·x to acc row by row, a nil row of x adding nothing.
func addRows(acc, x [][]float64, w float64) {
	for r := range x {
		if x[r] == nil {
			continue
		}
		if acc[r] == nil {
			acc[r] = make([]float64, len(x[r]))
		}
		for d := range x[r] {
			acc[r][d] += w * x[r][d]
		}
	}
}

// profileStart is the first profile grid's cells a period: start over
// a span of a period, fewer over a longer one, down to lagCellsMin. A
// narrow holding needs no finer grid, its cells' moments being taken on
// sub-cells as narrow as it is.
func profileStart(start, span int) int {
	n := start
	// bound: halvings of start
	for n > lagCellsMin && n*(span+1) > 2*start {
		n /= 2
	}
	return n
}

// gridSettle is one key's doubling: its last figure and extrapolation.
type gridSettle struct {
	prev, prevR, value float64
	dv, dr             float64 // the last two grids' differences
	done               bool
}

// step takes the next grid's figure, reporting whether it settles: two
// figures agree, or two extrapolations v + (v − v_prev)/3 do.
func (g *gridSettle) step(v float64) bool {
	r := v + (v-g.prev)/3
	dv, dr := math.Abs(v-g.prev), math.Abs(r-g.prevR)
	switch {
	case dv <= delaySettle*math.Abs(v):
		g.value, g.done = v, true
	case dr <= delaySettle*math.Abs(r):
		g.value, g.done = r, true
	}
	g.prev, g.prevR, g.dv, g.dr = v, r, dv, dr
	return g.done
}

// hopeless reports whether the key cannot settle by delayCellsMax from
// an n-cell grid: its differences fall by about four a doubling, and
// neither the figures' nor the extrapolations' reach delaySettle in the
// doublings left.
func (g *gridSettle) hopeless(n int) bool {
	if math.IsNaN(g.dr) {
		return false
	}
	reach := delaySettle * math.Abs(g.prev)
	// bound: doublings to delayCellsMax
	for m := n; m < delayCellsMax; m *= 2 {
		reach *= 4
	}
	return g.dv > reach && g.dr > reach
}

// allSettled reports whether every figure of b is within tol of a's.
func allSettled(a, b []float64, tol float64) bool {
	for i := range a {
		if !(math.Abs(a[i]-b[i]) <= tol*math.Abs(b[i])) {
			return false
		}
	}
	return true
}

// delayAt is Σ_k λ_k·S of source k's first node under the key on an
// n-cell grid, over every source or the key's one.
func (m *Compiled) delayAt(key delayKey, lam []float64, rules []*lagRule, n int, sc *delayScratch) float64 {
	sv := m.delaySecond(key.filt, rules, n, sc)
	v := 0.0
	for k, src := range m.srcs {
		if key.src < 0 || key.src == k {
			v += weighted(lam[k], sv[src.to])
		}
	}
	return v
}

// ruleCacheBytes caps what a rule cache keeps; past it rules are built
// again rather than kept.
const ruleCacheBytes = 64 << 20

// ruleKey is a holding's rule: its pool, the grid and, for a member,
// its conditioning a and b, or its point in lockstep; for a pool fixed
// at a point of its once-a-period law, that point; and for a profile
// grid, its span.
type ruleKey struct {
	pool       int32
	n, span    int
	a, b       float64
	member     bool
	lockstepAt float64
	fixed      bool
	at         float64
}

// ruleCache keeps the holdings' rules and the members' scores for the
// models the conditioning on value factors compiles: a factor never
// scales a holding, so every point's rules are the same.
type ruleCache struct {
	rules  map[ruleKey]*lagRule
	scores *scoreCache
	bytes  int
}

func newRuleCache() *ruleCache {
	return &ruleCache{rules: map[ruleKey]*lagRule{}, scores: &scoreCache{z: map[[2]int][2][]float64{}}}
}

// rule is the cached rule, or build's, kept while the cache has room;
// its transform is filled in when first applied, and kept with it.
func (rc *ruleCache) rule(k ruleKey, build func() lagRule) *lagRule {
	if rc == nil {
		r := build()
		return &r
	}
	if r, ok := rc.rules[k]; ok {
		return r
	}
	r := build()
	if size := 48 * (k.n + 1); rc.bytes+size <= ruleCacheBytes {
		rc.rules[k] = &r
		rc.bytes += size
	}
	return &r
}

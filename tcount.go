package variate

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// countMinGrid is the smallest grid of the count distributions.
const countMinGrid = 1 << 10

// Counts is the distribution of each node's entries per period on the
// integer lattice, inside a bracket, or the reason the method does not
// apply.
type Counts struct {
	Applicable bool
	Reason     string
	M          int
	Theta      float64
	Passes     int
	QuadNodes  int
	Entries    []Bounds // [node]; H = 1 and no discretisation gap
}

// counter holds one TransformEntries' working state.
type counter struct {
	t     *transformer
	mix   *thetaMix
	reach [][]float64 // scratch [source][node]
	mean  []float64   // [node]
	sd    []float64   // [node], +Inf when infinite
	xStar []float64   // [node] Cantelli (or Markov) bound on the TailP quantile
	m     int
	theta float64
	r     float64
	roots []complex128
	fft   *fourier.CmplxFFT
	eps   float64
	g     [][]complex128 // [node] mixed tilted generating functions
	ec    [][]float64
	rho   []float64
	alias []float64
}

// TransformEntries computes the distribution of every node's entries per
// period: Poisson given the period draws, mixed over them.
func (m *Compiled) TransformEntries(cfg TransformConfig) (*Counts, error) {
	t := newTransformer(m, cfg)
	out := &Counts{}
	if reason := t.applicable(); reason != "" {
		out.Reason = reason
		return out, nil
	}
	c := &counter{t: t, reach: make([][]float64, len(m.srcs))}
	for k := range c.reach {
		c.reach[k] = make([]float64, len(m.nodes))
	}
	if reason := c.moments(); reason != "" {
		out.Reason = reason
		return out, nil
	}
	// bound: doublings up to MaxGrid
	for grid := c.startGrid(); grid <= t.cfg.MaxGrid; grid *= 2 {
		if reason := c.run(grid); reason != "" {
			out.Reason = reason
			return out, nil
		}
		if c.withinTol() {
			c.fill(out)
			return out, nil
		}
	}
	if c.m == 0 {
		out.Reason = "the counts do not fit the grid"
		return out, nil
	}
	out.Reason = c.dominant()
	return out, nil
}

// moments fills each node's mean and standard deviation of entries per
// period, mixing the reach over the period-scoped probabilities with
// nodes doubling until the second moments settle.
func (c *counter) moments() string {
	m := c.t.m
	var qs []thetaQuantity
	for _, q := range m.periodQuantities() {
		if q.fn == noIndex {
			qs = append(qs, q)
		}
	}
	c.mix = m.newThetaMix(qs)
	n := probNodesStart
	var prev []float64
	// bound: quadDoublings
	for i := 0; i <= quadDoublings; i++ {
		if !c.mix.setProbNodes(m, n) {
			return reasonNoProbRule
		}
		if c.mix.passes() > maxPasses {
			return reasonProbCapped
		}
		second := c.secondMoments()
		if c.mix.probabilities() == 0 || prev != nil && settled(second, prev) {
			c.finishMoments(second)
			return ""
		}
		prev = second
		n *= 2
	}
	return reasonProbNoConv
}

// settled reports whether two moment vectors agree to quadTol relatively.
func settled(a, b []float64) bool {
	for v := range a {
		if math.Abs(a[v]-b[v]) > quadTol*math.Max(1, math.Abs(a[v])) {
			return false
		}
	}
	return true
}

// secondMoments returns E[(Σ_k λ_k R_k(v | Θ))²] per node at the current
// nodes, and sets mean to E[Σ_k λ_k R_k(v | Θ)].
func (c *counter) secondMoments() []float64 {
	m := c.t.m
	second := make([]float64, len(m.nodes))
	c.mean = make([]float64, len(m.nodes))
	_ = m.countN2
	// bound: maxPasses
	for ok := c.mix.start(); ok; ok = c.mix.next() {
		w := c.mix.weight()
		m.reachWith(c.mix.drawn, c.reach)
		for v := range m.nodes {
			sum := 0.0
			sq := 0.0
			for k := range m.srcs {
				r := c.reach[k][v]
				sum += m.rateGiven(k, c.mix.drawn) * r
				sq += m.rate2Given(k, c.mix.drawn) * r * r
			}
			// E[(Σ λ_k R_k)²] = Σ_k E[λ_k²] R_k² + Σ_{k≠l} E[λ_k] E[λ_l] R_k R_l.
			c.mean[v] += w * sum
			second[v] += w * (sq + sum*sum - c.sumSquares(v, c.mix.drawn))
		}
	}
	return second
}

// sumSquares is Σ_k (E[λ_k] R_k(v))² at the current reach and draws.
func (c *counter) sumSquares(v int, drawn []float64) float64 {
	s := 0.0
	for k := range c.t.m.srcs {
		x := c.t.m.rateGiven(k, drawn) * c.reach[k][v]
		s += x * x
	}
	return s
}

// finishMoments turns the mixed moments into each node's sd and x*.
func (c *counter) finishMoments(second []float64) {
	t := c.t
	cc := math.Sqrt(t.tailP / (1 - t.tailP))
	c.sd = make([]float64, len(c.mean))
	c.xStar = make([]float64, len(c.mean))
	for v := range c.mean {
		mean := c.mean[v]
		vr := mean + second[v] - mean*mean
		if math.IsInf(vr, 1) || math.IsNaN(vr) {
			c.sd[v] = math.Inf(1)
			c.xStar[v] = mean / (1 - t.tailP)
			continue
		}
		c.sd[v] = math.Sqrt(math.Max(0, vr))
		c.xStar[v] = mean + cc*c.sd[v]
	}
}

// startGrid is the smallest power of two holding twice the largest x*.
func (c *counter) startGrid() int {
	need := 0.0
	for _, x := range c.xStar {
		need = math.Max(need, 2*x)
	}
	grid := countMinGrid
	// bound: doublings up to MaxGrid
	for float64(grid) < need && grid < c.t.cfg.MaxGrid {
		grid *= 2
	}
	return grid
}

// run evaluates every node's count distribution on a grid of m points.
func (c *counter) run(m int) string {
	c.m = m
	c.t.grid = m
	if b := 2 * int64(m) * bytesPerPoint * int64(len(c.t.m.nodes)+4); b > c.t.cfg.MaxBytes {
		return "the entries transform would hold " + itoa(int(b>>20)) + " MB at " + itoa(m) + " grid points, above " + itoa(int(c.t.cfg.MaxBytes>>20)) + " MB"
	}
	deepest := 0
	for v := range c.xStar {
		if c.xStar[v] > c.xStar[deepest] {
			deepest = v
		}
	}
	kStar := min(int(math.Ceil(c.xStar[deepest])), m-1)
	c.theta = c.t.chooseTiltFor(1, m, kStar, c.mean[deepest], c.sd[deepest])
	c.r = math.Exp(-c.theta)
	c.roots = unitRoots(m)
	for j := range c.roots {
		c.roots[j] *= complex(c.r, 0)
	}
	c.fft = fourier.NewCmplxFFT(m)
	if reason := c.mixed(); reason != "" {
		return reason
	}
	c.rho = c.t.roundoff(1, m, c.theta, c.eps)
	c.ec = make([][]float64, len(c.g))
	c.alias = make([]float64, len(c.g))
	for v := range c.g {
		c.ec[v] = c.invert(c.g[v])
		c.alias[v] = aliasFor(c.mean[v], c.sd[v], float64(m), c.theta)
	}
	return ""
}

// mixed fills g with the generating functions mixed over the
// probabilities, their nodes doubling until the result settles.
func (c *counter) mixed() string {
	n := probNodesStart
	var prev [][]complex128
	// bound: quadDoublings
	for i := 0; i <= quadDoublings; i++ {
		if !c.mix.setProbNodes(c.t.m, n) {
			return reasonNoProbRule
		}
		if c.mix.passes() > maxPasses {
			return reasonProbCapped
		}
		g, reason := c.tensor()
		if reason != "" {
			return reason
		}
		c.t.passes = c.mix.passes()
		if c.mix.probabilities() == 0 {
			c.g = g
			return ""
		}
		if prev != nil {
			diff := 0.0
			for v := range g {
				diff = math.Max(diff, maxAbsDiff(g[v], prev[v]))
			}
			if diff <= quadTol {
				c.eps = math.Max(c.eps, diff)
				c.g = g
				return ""
			}
		}
		prev = g
		n *= 2
	}
	return reasonProbNoConv
}

// tensor sums each node's conditional generating function over the
// period draws: Π_k P_{N_k}(1 − R_k(v | Θ)(1 − r·z_j)).
func (c *counter) tensor() ([][]complex128, string) {
	m := c.t.m
	g := make([][]complex128, len(m.nodes))
	for v := range g {
		g[v] = make([]complex128, c.m)
	}
	gc := make([]complex128, c.m)
	one := make([]complex128, c.m)
	prod := make([]complex128, c.m)
	// bound: maxPasses
	for ok := c.mix.start(); ok; ok = c.mix.next() {
		w := complex(c.mix.weight(), 0)
		m.reachWith(c.mix.drawn, c.reach)
		for v := range m.nodes {
			if reason := c.nodeGiven(v, prod, gc, one); reason != "" {
				return nil, reason
			}
			for j := range prod {
				g[v][j] += w * prod[j]
			}
		}
		// Every pass repeats the rate quadrature per node and source.
		if work := int64(c.mix.passes()) * int64(max(1, c.t.quadN)) * int64(c.m) * int64(len(m.nodes)) * int64(len(m.srcs)); work > quadWorkCap {
			return nil, "entries mixture of " + itoa(c.mix.passes()) + " passes with " + itoa(c.t.quadN) + " frequency quadrature nodes at " + itoa(c.m) + " grid points over " + itoa(len(m.nodes)) + " nodes is above the work cap"
		}
	}
	return g, ""
}

// nodeGiven fills prod with node v's conditional generating function at
// the current reach; gc and one are scratch.
func (c *counter) nodeGiven(v int, prod, gc, one []complex128) string {
	for j := range prod {
		prod[j] = 1
	}
	for k := range c.t.m.srcs {
		rk := complex(c.reach[k][v], 0)
		for j := range gc {
			gc[j] = 1 - rk*(1-c.roots[j])
		}
		eps, reason := c.t.compound(k, gc, one, c.t.m.factorAt(k, c.mix.drawn))
		if reason != "" {
			return reason
		}
		c.eps = math.Max(c.eps, eps)
		for j := range prod {
			prod[j] *= one[j]
		}
	}
	return ""
}

// invert turns a tilted generating function into an exceedance curve.
func (c *counter) invert(g []complex128) []float64 {
	masses := c.fft.Sequence(nil, g)
	ec := make([]float64, c.m)
	cdf := 0.0
	rk := 1.0
	for k := range ec {
		ec[k] = 1 - cdf
		cdf += real(masses[k]) / float64(c.m) / rk
		rk *= c.r
	}
	return ec
}

func (c *counter) lowerAt(v, k int) float64 { return math.Max(0, c.ec[v][k]-c.rho[k]) }

func (c *counter) upperAt(v, k int) float64 {
	return math.Min(1, c.ec[v][k]+c.rho[k]+c.alias[v])
}

// quantileIndex is the first lattice point where a node's upper curve
// falls to 1 − TailP.
func (c *counter) quantileIndex(v int) int {
	target := 1 - c.t.tailP
	// bound: grid points
	for k := 0; k < c.m; k++ {
		if c.upperAt(v, k) <= target {
			return k
		}
	}
	return c.m - 1
}

// withinTol reports whether every node's bracket is resolved up to its
// own TailP quantile.
func (c *counter) withinTol() bool {
	for v := range c.ec {
		kq := c.quantileIndex(v)
		// bound: grid points up to the quantile
		for k := 0; k <= kq; k++ {
			if !resolved(c.lowerAt(v, k), c.upperAt(v, k), c.t.cfg.Tol, c.t.tailP) {
				return false
			}
		}
	}
	return true
}

// dominant names the term that keeps a bracket open.
func (c *counter) dominant() string {
	var alias, rho float64
	for v := range c.ec {
		kq := c.quantileIndex(v)
		alias = math.Max(alias, c.alias[v])
		rho = math.Max(rho, c.rho[kq])
	}
	if alias >= rho {
		return "the wrap-around bound stays above Tol at MaxGrid: the tail is too heavy for this method"
	}
	return "the round-off estimate stays above Tol at MaxGrid: the tail is too heavy for this method"
}

// fill writes the result.
func (c *counter) fill(out *Counts) {
	out.Applicable = true
	out.M, out.Theta, out.Passes, out.QuadNodes = c.m, c.theta, c.t.passes, c.t.quadN
	out.Entries = make([]Bounds, len(c.ec))
	for v := range c.ec {
		b := Bounds{H: 1, Lower: make([]float64, c.m), Upper: make([]float64, c.m), Roundoff: c.rho, Alias: c.alias[v]}
		// bound: grid points
		for k := 0; k < c.m; k++ {
			b.Lower[k] = c.lowerAt(v, k)
			b.Upper[k] = c.upperAt(v, k)
		}
		tighten(b.Lower, b.Upper)
		out.Entries[v] = b
	}
}

package variate

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/dsp/fourier"
)

// passResult is one grid's complete answer: the lower and upper
// pipelines for the total, categories and sources.
type passResult struct {
	h     float64
	m     int
	theta float64
	alias float64
	beta  float64
	rho   []float64 // ρ(x) at each lattice point, for the chosen tilt
	pzero float64
	total brackets
	cats  []brackets
	srcs  []brackets
}

// brackets holds the two exceedance curves of one quantity.
type brackets struct {
	down, up []float64 // EC_down(x), EC_up(x)
}

// pipeline is one discretisation direction's transform state.
type pipeline struct {
	t         *transformer
	h         float64
	m         int
	r         float64 // tilt factor e^(−θh)
	up        bool
	fft       *fourier.CmplxFFT
	roots     []complex128 // e^(−2πi·j/M)
	mix       *thetaMix    // the period-scoped quantities with this rounding's atoms
	fnG       [][]complex128
	tilted    []complex128   // scratch: a function's tilted lattice masses
	gf        []complex128   // scratch: its transform
	mixBuf    []complex128   // scratch: a node's outcome mixture
	masses    []complex128   // scratch: an inversion's masses
	free      [][]complex128 // released node generating functions, for reuse
	accounted bool           // the lattice mean and dropped mass have been taken
	beta      float64        // dropped mass, total variation
	mean      float64        // exact lattice mean of T
	dropMean  float64        // bound on the mean the dropped tails carry
	eps       float64        // quadrature error
	pzero     float64        // mass at 0 of the last inverted total
}

// graphResult holds the tilted generating functions of the quantities.
type graphResult struct {
	total []complex128
	cats  [][]complex128
	srcs  [][]complex128
}

// pass evaluates both pipelines at (h, M) with the tilt chosen for the
// point that must be resolved: x* when known, M·h/2 otherwise.
func (t *transformer) pass(h float64, m int, xStar float64) (*passResult, string) {
	kStar := m / 2
	if xStar > 0 {
		kStar = min(int(math.Ceil(xStar/h)), m-1)
	}
	theta := t.chooseTilt(h, m, kStar)
	r := math.Exp(-theta * h)
	start := t.work
	t.grid = m
	defer func() { t.passWork, t.passM = t.work-start, m }()
	down, reason := t.newPipeline(h, m, r, false)
	if reason != "" {
		return nil, reason
	}
	up, reason := t.newPipeline(h, m, r, true)
	if reason != "" {
		return nil, reason
	}
	res := &passResult{h: h, m: m, theta: theta}
	gDown, reason := down.graph()
	if reason != "" {
		return nil, reason
	}
	gUp, reason := up.graph()
	if reason != "" {
		return nil, reason
	}
	t.passes = up.mix.passes()
	res.total = brackets{down: down.invert(gDown.total), up: up.invert(gUp.total)}
	res.pzero = up.pzero
	for c := range gDown.cats {
		res.cats = append(res.cats, brackets{down: down.invert(gDown.cats[c]), up: up.invert(gUp.cats[c])})
	}
	for k := range gDown.srcs {
		res.srcs = append(res.srcs, brackets{down: down.invert(gDown.srcs[k]), up: up.invert(gUp.srcs[k])})
	}
	res.beta = math.Max(down.beta, up.beta)
	res.alias = t.aliasBound(h, m, theta)
	eps := math.Max(down.eps, up.eps)
	res.rho = t.roundoff(h, m, theta, eps)
	if reason := res.selfCheck(t, up.mean, down.mean, up.dropMean); reason != "" {
		return nil, reason
	}
	return res, ""
}

// newPipeline prepares one rounding's state, with the period-scoped
// values' lattice atoms; the reason when the atoms exceed the pass cap.
func (t *transformer) newPipeline(h float64, m int, r float64, up bool) (*pipeline, string) {
	p := &pipeline{t: t, h: h, m: m, r: r, up: up, fft: fourier.NewCmplxFFT(m)}
	p.roots = unitRoots(m)
	p.tilted = make([]complex128, m)
	p.gf = make([]complex128, m)
	p.mixBuf = make([]complex128, m)
	p.masses = make([]complex128, m)
	qs := t.m.periodQuantities()
	atoms := 1
	for i := range qs {
		if qs[i].fn == noIndex {
			continue
		}
		q := &qs[i]
		lo, hi := t.m.pools[q.pool].Support()
		if (hi-lo)/h > maxAtomScan || !p.valueAtoms(q, maxPasses/atoms) {
			return nil, "period-scoped values span too many lattice points: " + t.m.fnNames[q.fn]
		}
		atoms *= len(q.nodes)
	}
	p.mix = t.m.newThetaMix(qs)
	return p, ""
}

// unitRoots is e^(−2πi·j/M) for j below M, exactly conjugate-symmetric
// about M/2 so that every generating function built from real masses
// is, and its second half can be mirrored instead of computed.
func unitRoots(m int) []complex128 {
	roots := make([]complex128, m)
	// bound: M/2 + 1 roots
	for j := 0; j <= m/2; j++ {
		roots[j] = cmplx.Rect(1, -2*math.Pi*float64(j)/float64(m))
	}
	if m%2 == 0 {
		roots[m/2] = -1
	}
	mirror(roots)
	return roots
}

// valueAtoms fills a period-scoped value's lattice atoms under this
// rounding and takes its lattice mean; false when the atoms exceed cap.
func (p *pipeline) valueAtoms(q *thetaQuantity, cap int) bool {
	d := p.t.m.pools[q.pool]
	lo, hi := d.Support()
	k := max(0, int(math.Floor(lo/p.h))-1)
	prev := cdfAt(d, float64(k)*p.h, p.h, p.up)
	lmean := 0.0
	// bound: the support in lattice steps, at most maxAtomScan
	for ; float64(k)*p.h <= hi+p.h; k++ {
		x := float64(k) * p.h
		next := cdfAt(d, x+p.h, p.h, p.up)
		f := math.Max(0, next-prev)
		prev = next
		if f > 0 {
			if len(q.nodes) >= cap {
				return false
			}
			q.nodes = append(q.nodes, x)
			q.weight = append(q.weight, f)
			lmean += f * x
		}
	}
	p.mean += p.t.draws[q.fn] * lmean
	return true
}

// graph runs the recursion for the total, each category and each source,
// mixed over the period-scoped quantities, with the probabilities' nodes
// doubling until the result settles.
func (p *pipeline) graph() (*graphResult, string) {
	if p.mix.probabilities() == 0 {
		return p.tensor()
	}
	n := probNodesStart
	var prev *graphResult
	var counts, last []int
	// bound: quadDoublings
	for i := 0; i <= quadDoublings; i++ {
		if !p.mix.setProbNodes(p.t.m, n) {
			return nil, reasonNoProbRule
		}
		if p.mix.passes() > maxPasses {
			return nil, reasonProbCapped
		}
		// A finite rule stops growing: its atoms are exact.
		counts = p.mix.probCounts(counts)
		if prev != nil && sameCounts(counts, last) {
			return prev, ""
		}
		last = append(last[:0], counts...)
		g, reason := p.tensor()
		if reason != "" {
			return nil, reason
		}
		if prev != nil {
			if diff := g.maxDiff(prev); diff <= quadTol {
				p.eps = math.Max(p.eps, diff)
				return g, ""
			}
		}
		prev = g
		n *= 2
	}
	return nil, reasonProbNoConv
}

// maxDiff is the largest pointwise difference between two results.
func (g *graphResult) maxDiff(o *graphResult) float64 {
	d := maxAbsDiff(g.total, o.total)
	for c := range g.cats {
		d = math.Max(d, maxAbsDiff(g.cats[c], o.cats[c]))
	}
	for k := range g.srcs {
		d = math.Max(d, maxAbsDiff(g.srcs[k], o.srcs[k]))
	}
	return d
}

func maxAbsDiff(a, b []complex128) float64 {
	d := 0.0
	for j := range a {
		d = math.Max(d, cmplx.Abs(a[j]-b[j]))
	}
	return d
}

// tensor sums the conditional results over the period draws. A point
// whose value atom lies beyond the grid is dropped into β.
func (p *pipeline) tensor() (*graphResult, string) {
	m := p.t.m
	out := &graphResult{total: make([]complex128, p.m), cats: make([][]complex128, len(m.categories))}
	for c := range out.cats {
		out.cats[c] = make([]complex128, p.m)
	}
	if len(m.srcs) <= maxSourceBounds {
		out.srcs = make([][]complex128, len(m.srcs))
		for k := range out.srcs {
			out.srcs[k] = make([]complex128, p.m)
		}
	}
	if p.mix.passes() > 1 && p.fnG == nil {
		p.fnG = make([][]complex128, len(m.fns))
	}
	// bound: maxPasses
	for ok := p.mix.start(); ok; ok = p.mix.next() {
		w := p.mix.weight()
		if p.beyondGrid() {
			p.beta += w
			continue
		}
		one, reason := p.graphGiven()
		if reason != "" {
			return nil, reason
		}
		if work := p.mix.passes() * max(1, p.t.quadN) * p.m * (1 + len(p.t.m.categories)); work > quadWorkCap {
			return nil, "period-scoped mixture of " + itoa(p.mix.passes()) + " passes with " + itoa(p.t.quadN) + " frequency quadrature nodes at " + itoa(p.m) + " grid points is above the work cap"
		}
		out.add(one, complex(w, 0))
	}
	return out, ""
}

// allOne reports whether a generating function is identically 1.
func allOne(g []complex128) bool {
	for _, x := range g {
		if x != 1 {
			return false
		}
	}
	return true
}

// beyondGrid reports whether a value atom of the current point lies at or
// beyond M·h.
func (p *pipeline) beyondGrid() bool {
	for _, q := range p.mix.q {
		if q.fn != noIndex && p.mix.drawn[q.pool] >= float64(p.m)*p.h {
			return true
		}
	}
	return false
}

func (g *graphResult) add(o *graphResult, w complex128) {
	for j := range g.total {
		g.total[j] += w * o.total[j]
	}
	for c := range g.cats {
		for j := range g.cats[c] {
			g.cats[c][j] += w * o.cats[c][j]
		}
	}
	for k := range g.srcs {
		for j := range g.srcs[k] {
			g.srcs[k][j] += w * o.srcs[k][j]
		}
	}
}

// graphGiven runs the recursion at the current period draws: once for
// the total, and once per category over that category's functions.
func (p *pipeline) graphGiven() (*graphResult, string) {
	m := p.t.m
	out := &graphResult{}
	// A source's own generating function is kept for its bounds when
	// there are few sources; a register of many would hold a grid each.
	if len(m.srcs) <= maxSourceBounds {
		out.srcs = make([][]complex128, len(m.srcs))
	}
	total, reason := p.recurse(-1, out.srcs)
	if reason != "" {
		return nil, reason
	}
	out.total = total
	for c := range m.categories {
		gc, reason := p.recurse(index(c), nil)
		if reason != "" {
			return nil, reason
		}
		out.cats = append(out.cats, gc)
	}
	return out, ""
}

// recurse computes the generating function of the period total booked
// in category cat (−1 for all) as the product over the sources of their
// compounded subgraphs. The nodes the sources reach are taken children
// before parents, depth first from each source's target; a source is
// compounded as soon as its target is complete, and a node's generating
// function is released once every source and outcome leading to it has
// used it, so only a handful of grids are held at once, not one per
// node. srcs, when given, receives each source's own compounded function.
func (p *pipeline) recurse(cat int32, srcs [][]complex128) ([]complex128, string) {
	m := p.t.m
	g := make([][]complex128, len(m.nodes))
	uses := make([]int32, len(m.nodes))
	copy(uses, p.t.uses)
	total := p.grid()
	for j := range total {
		total[j] = 1
	}
	scratch := p.grid()
	for _, v := range p.t.order {
		n := m.nodes[v]
		gv := p.grid()
		for j := range gv {
			gv[j] = 1
		}
		p.nodeFunctions(gv, n, cat)
		if n.outStart != n.outEnd {
			p.nodeMix(gv, v, n, g, uses)
		}
		g[v] = gv
		if reason := p.compoundAt(v, cat, gv, total, scratch, srcs, uses); reason != "" {
			return nil, reason
		}
		if uses[v] == 0 {
			p.release(g, v)
		}
	}
	p.free = append(p.free, scratch)
	if cat == -1 {
		p.accounted = true
	}
	return total, ""
}

// compoundAt compounds every source whose target is node v, now that
// its generating function gv is complete, into total, and into srcs
// when given; each source counts as one use of v.
func (p *pipeline) compoundAt(v, cat int32, gv, total, scratch []complex128, srcs [][]complex128, uses []int32) string {
	// bound: sources
	for k, s := range p.t.m.srcs {
		if s.to != v {
			continue
		}
		uses[v]--
		// A source that books nothing in the category has the
		// generating function 1 there, and compounds to 1.
		if cat != -1 && allOne(gv) {
			continue
		}
		if reason := p.compound(k, gv, scratch); reason != "" {
			return reason
		}
		for j := range total {
			total[j] *= scratch[j]
		}
		if srcs != nil {
			srcs[k] = append(p.grid()[:0], scratch...)
		}
	}
	return ""
}

// grid returns a released generating function's buffer, or a new one.
func (p *pipeline) grid() []complex128 {
	if n := len(p.free); n > 0 {
		gv := p.free[n-1]
		p.free = p.free[:n-1]
		return gv
	}
	return make([]complex128, p.m)
}

// release returns node v's generating function to the pool.
func (p *pipeline) release(g [][]complex128, v int32) {
	p.free = append(p.free, g[v])
	g[v] = nil
}

// transformOrder lists the nodes the sources reach, each after the nodes
// its outcomes lead to and as soon after them as possible: a depth-first
// post-order from each source's target. uses counts, per node, the
// outcomes and sources leading to it.
func (m *Compiled) transformOrder() (order, uses []int32) {
	uses = make([]int32, len(m.nodes))
	for _, o := range m.outs {
		uses[o.to]++
	}
	seen := make([]bool, len(m.nodes))
	order = make([]int32, 0, len(m.nodes))
	type frame struct{ v, j int32 }
	stack := make([]frame, 0, len(m.nodes))
	for _, s := range m.srcs {
		uses[s.to]++
		if seen[s.to] {
			continue
		}
		seen[s.to] = true
		stack = append(stack, frame{s.to, m.nodes[s.to].outStart})
		// bound: every reachable node is pushed once and each of its
		// outcomes looked at once
		for len(stack) > 0 {
			f := &stack[len(stack)-1]
			if f.j < m.nodes[f.v].outEnd {
				child := m.outs[f.j].to
				f.j++
				if !seen[child] {
					seen[child] = true
					stack = append(stack, frame{child, m.nodes[child].outStart})
				}
				continue
			}
			order = append(order, f.v)
			stack = stack[:len(stack)-1]
		}
	}
	return order, uses
}

// nodeFunctions multiplies in the node's functions of the category.
func (p *pipeline) nodeFunctions(gv []complex128, n cnode, cat int32) {
	m := p.t.m
	// bound: functions of one node
	for f := n.fnStart; f < n.fnEnd; f++ {
		if cat != -1 && m.fns[f].category != cat {
			continue
		}
		if m.poolPeriod[m.fns[f].pool] != noIndex {
			p.multiplyAtom(gv, int(math.Round(p.mix.drawn[m.fns[f].pool]/p.h)))
			continue
		}
		p.multiplyFunction(gv, f, cat == -1 && !p.accounted)
	}
}

// nodeMix multiplies in the mixture over the node's outcomes, releasing
// a child's generating function with its last use.
func (p *pipeline) nodeMix(gv []complex128, v int32, n cnode, g [][]complex128, uses []int32) {
	m := p.t.m
	mix := p.mixBuf
	clear(mix)
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		pj := complex(m.outcomeProbWith(v, j, p.mix.drawn), 0)
		to := m.outs[j].to
		child := g[to]
		for i := range mix {
			mix[i] += pj * child[i]
		}
		uses[to]--
		if uses[to] == 0 {
			p.release(g, to)
		}
	}
	for i := range gv {
		gv[i] *= mix[i]
	}
}

// multiplyAtom multiplies gv by the tilted generating function of a
// single atom at lattice index k: r^k·z_j^k, with z_j^k read from the
// root table at (j·k) mod M.
func (p *pipeline) multiplyAtom(gv []complex128, k int) {
	rk := complex(math.Pow(p.r, float64(k)), 0)
	for j := range gv {
		gv[j] *= rk * p.roots[(j*k)%p.m]
	}
}

// multiplyFunction multiplies gv by function f's tilted generating
// function, computed from its lattice masses with folding and cached
// when the mixture has more than one point. The lattice mean and the
// dropped mass are accounted once, in the first all-functions pass.
func (p *pipeline) multiplyFunction(gv []complex128, f int32, account bool) {
	if p.fnG != nil && p.fnG[f] != nil {
		for i := range gv {
			gv[i] *= p.fnG[f][i]
		}
		return
	}
	d := p.t.m.pools[p.t.m.fns[f].pool]
	_, hi := d.Support()
	tilted := p.tilted
	clear(tilted)
	rm := 1.0 // r^(nM)
	dropped := 1.0
	var lmean float64
	// bound: maxFolds grid lengths
	for n := 0; n < maxFolds; n++ {
		base := float64(n * p.m)
		mass := 0.0
		rk := rm
		prev := cdfAt(d, base*p.h, p.h, p.up)
		// bound: grid points up to the support's end
		for k := 0; k < p.m; k++ {
			x := (base + float64(k)) * p.h
			if x > hi+p.h {
				break
			}
			next := cdfAt(d, x+p.h, p.h, p.up)
			f := math.Max(0, next-prev)
			prev = next
			tilted[k] += complex(f*rk, 0)
			lmean += f * x
			mass += f
			rk *= p.r
		}
		rm *= math.Pow(p.r, float64(p.m))
		dropped = survival(d, (base+float64(p.m))*p.h)
		if rm*dropped < unitRoundoff || mass == 0 && n > 0 {
			// The tail beyond here is left out of the lattice function. Its
			// tilted weight is below round-off, and its values lie beyond
			// the grid, so the exceedance still counts it; only the lattice
			// mean misses it, by at most √(E[X²]·P(X ≥ cut)).
			if account {
				p.dropMean += p.t.draws[f] * math.Sqrt((d.Var()+d.Mean()*d.Mean())*dropped)
			}
			dropped = 0
			break
		}
	}
	if account {
		p.beta += p.t.draws[f] * dropped
		p.mean += p.t.draws[f] * lmean
	}
	gf := p.fft.Coefficients(p.gf, tilted)
	p.t.work += p.m
	// The masses are real, so the coefficients are conjugate-symmetric up
	// to round-off; made exactly so, every product of them is too.
	mirror(gf)
	if p.fnG != nil {
		p.fnG[f] = append([]complex128(nil), gf...)
	}
	for i := range gv {
		gv[i] *= gf[i]
	}
}

// compound applies source k's mixed-Poisson generating function to gc.
func (p *pipeline) compound(k int, gc, out []complex128) string {
	eps, reason := p.t.compound(k, gc, out, p.t.m.factorAt(k, p.mix.drawn))
	p.eps = math.Max(p.eps, eps)
	return reason
}

// invert turns a tilted generating function into an exceedance curve:
// inverse FFT, undo the tilt, sum the CDF from the bottom.
func (p *pipeline) invert(g []complex128) []float64 {
	masses := p.fft.Sequence(p.masses, g)
	ec := make([]float64, p.m)
	cdf := 0.0
	rk := 1.0
	for k := range ec {
		ec[k] = 1 - cdf
		mk := real(masses[k]) / float64(p.m) / rk
		cdf += mk
		rk *= p.r
	}
	p.pzero = real(masses[0]) / float64(p.m)
	return ec
}

// cdfAt is the cumulative function C of a discretisation, defined so that
// the mass on lattice point x is C(x + h) − C(x): P(X < y) for the lower
// rounding, where the mass of [kh, (k+1)h) lands on kh, and P(X ≤ y − h)
// for the upper, where the mass of ((k−1)h, kh] lands on kh.
func cdfAt(d distribution, y, h float64, up bool) float64 {
	if up {
		return d.CDF(y - h)
	}
	return cdfBelow(d, y)
}

// cdfBelow is P(X < x).
func cdfBelow(d distribution, x float64) float64 {
	if x <= 0 {
		return 0
	}
	return d.CDF(math.Nextafter(x, 0))
}

func survival(d distribution, x float64) float64 {
	return math.Max(0, 1-d.CDF(math.Nextafter(x, 0)))
}

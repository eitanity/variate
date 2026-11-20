package variate

import (
	"math"
	"slices"

	"code.eitanity.com/variate/dist"
)

// A round's figures given a group's normal W have a kink in W wherever
// the difference of two of its bookings' delays crosses 0 or ±1, where
// the period kernel K(δ) = (1 − |δ|)⁺ bends. Where the delays are the
// group's holdings, a delay given W is near its quantile at Φ(√ρ·W),
// and the kinks, sharp in lockstep, are bends as narrow as √(1 − ρ)
// below it. A Gauss–Hermite rule over W, made for a smooth integrand,
// settles slowly or not at all near them; split at them, a
// Gauss–Legendre rule on each piece of W, weighted by the normal
// density, settles fast. The lag profiles bend at every whole lag too,
// but beyond the kernel's the holdings' spread given W and the normal's
// tail smooth them: on the pieces, the between-period term settles with
// the within-period one.

const (
	// depEdge bounds W: the normal mass beyond ±8 is 1.2e-15.
	depEdge = 8
	// depScan is the steps over [−depEdge, depEdge] a delay's difference
	// is read on for its crossings, each then found by bisection.
	depScan = 1 << 10
	// depPiecesMax caps a group's pieces, the kinks nearest W = 0 kept.
	depPiecesMax = 32
	// depPathsMax caps the delays a group's kinks are looked for over.
	depPathsMax = 64
	// depSplitStart is the Gauss–Legendre nodes a piece starts from.
	depSplitStart = 4
	// depNarrow is the width of W below which a piece, a band about a
	// kink in lockstep, takes depSplitStart nodes: its figures are all but
	// polynomial across it.
	depNarrow = 2.0 * depEdge / depScan
	// depBisect is the bisection steps that place a kink.
	depBisect = 60
)

// delayGap is the difference of two bookings' delays past the outcomes
// their paths share: each side's holdings of the group, read at their
// quantile at Φ(√ρ·W), and the fixed holdings' sum.
type delayGap struct {
	plus, minus []int32
	fixed       float64
}

// at is the gap at W = w for a group of √ρ = sq.
func (g delayGap) at(m *Compiled, sq, w float64) float64 {
	v := g.fixed
	for _, p := range g.plus {
		v += copulaValue(m.pools[p], sq*w)
	}
	for _, p := range g.minus {
		v -= copulaValue(m.pools[p], sq*w)
	}
	return v
}

// groupPaths is each source's paths, by outcome, to the nodes it
// reaches; false past depPathsMax.
func (m *Compiled) groupPaths() ([][][]int32, bool) {
	out := make([][][]int32, len(m.srcs))
	total := 0
	for k, s := range m.srcs {
		paths := [][]int32{nil}
		ends := []int32{s.to}
		// bound: depPathsMax paths
		for len(paths) > 0 {
			p, v := paths[len(paths)-1], ends[len(ends)-1]
			paths, ends = paths[:len(paths)-1], ends[:len(ends)-1]
			if total++; total > depPathsMax {
				return nil, false
			}
			out[k] = append(out[k], p)
			nd := m.nodes[v]
			// bound: outcomes of one node
			for j := nd.outStart; j < nd.outEnd; j++ {
				if to := m.outs[j].to; to != noIndex {
					paths, ends = append(paths, append(slices.Clone(p), j)), append(ends, to)
				}
			}
		}
	}
	return out, true
}

// gapOf is the gap between paths a and b for group gi: false when a
// holding past their shared outcomes is neither fixed nor the group's,
// which blurs any kink, or when the gap does not vary with W.
func (m *Compiled) gapOf(gi int32, a, b []int32) (delayGap, bool) {
	c := 0
	// bound: the shorter path's outcomes
	for c < len(a) && c < len(b) && a[c] == b[c] {
		c++
	}
	var g delayGap
	for side, p := range [2][]int32{a[c:], b[c:]} {
		sign := 1.0 - 2*float64(side)
		for _, j := range p {
			o := m.outs[j]
			if o.hold == noIndex {
				continue
			}
			lo, hi := m.pools[o.hold].Support()
			switch {
			case lo == hi:
				g.fixed += sign * lo
			case m.outGroup != nil && m.outGroup[j] == gi && side == 0:
				g.plus = append(g.plus, o.hold)
			case m.outGroup != nil && m.outGroup[j] == gi:
				g.minus = append(g.minus, o.hold)
			default:
				return delayGap{}, false
			}
		}
	}
	return g, len(g.plus)+len(g.minus) > 0
}

// groupKinks is the W in (−depEdge, depEdge) where a gap of group gi
// crosses 0 or ±1, or, with each holding spread over band, the edges of
// the bands about them, ±band/2 for a gap of one holding and 0, ±band
// for one of two; nil without one, false past depPathsMax.
func (m *Compiled) groupKinks(gi int32, band float64) ([]float64, bool) {
	paths, ok := m.groupPaths()
	if !ok {
		return nil, false
	}
	sq := m.groups[gi].sq
	offsets := []float64{0}
	if band > 0 {
		offsets = []float64{-band, -band / 2, 0, band / 2, band}
	}
	var kinks []float64
	for _, ps := range paths {
		for i, a := range ps {
			for _, b := range ps[:i] {
				g, ok := m.gapOf(gi, a, b)
				if !ok {
					continue
				}
				for _, o := range offsets {
					at := g
					at.fixed -= o
					kinks = append(kinks, at.crossings(m, sq, 1)...)
				}
			}
		}
	}
	slices.Sort(kinks)
	return slices.Compact(kinks), true
}

// crossings is the W where the gap crosses a whole number of periods
// k, |k| ≤ top, read on depScan steps and placed by bisection.
func (g delayGap) crossings(m *Compiled, sq, top float64) []float64 {
	var out []float64
	h := 2.0 * depEdge / depScan
	w0, f0 := -float64(depEdge), g.at(m, sq, -depEdge)
	// bound: depScan steps
	for s := 1; s <= depScan; s++ {
		w1 := -float64(depEdge) + float64(s)*h
		f1 := g.at(m, sq, w1)
		lo, hi := math.Max(math.Min(f0, f1), -top-1), math.Min(math.Max(f0, f1), top)
		// bound: the whole lags from −top to top
		for k := math.Floor(lo) + 1; k <= hi; k++ {
			out = append(out, g.bisect(m, sq, w0, w1, k))
		}
		w0, f0 = w1, f1
	}
	return out
}

// bisect places the W in [a, b] where the gap is k, its sign at a and b
// apart.
func (g delayGap) bisect(m *Compiled, sq, a, b, k float64) float64 {
	below := g.at(m, sq, a) < k
	// bound: depBisect steps
	for range depBisect {
		c := (a + b) / 2
		if g.at(m, sq, c) < k == below {
			a = c
		} else {
			b = c
		}
	}
	return (a + b) / 2
}

// splitRule is a group's rule over W split at its kinks: on each piece
// of splitCuts n nodes by pieceRule.
func splitRule(kinks []float64, n int) (x, w []float64) {
	cuts := splitCuts(kinks)
	for p := range len(cuts) - 1 {
		px, pw := pieceRule(cuts[p], cuts[p+1], n)
		x, w = append(x, px...), append(w, pw...)
	}
	return x, w
}

// splitCuts is [−depEdge, depEdge] cut at the kinks, the depPiecesMax − 1
// nearest 0 kept, and where the tails start, ascending.
func splitCuts(kinks []float64) []float64 {
	if len(kinks) >= depPiecesMax {
		kinks = slices.Clone(kinks)
		slices.SortFunc(kinks, func(a, b float64) int {
			switch {
			case math.Abs(a) < math.Abs(b):
				return -1
			case math.Abs(a) > math.Abs(b):
				return 1
			}
			return 0
		})
		kinks = kinks[:depPiecesMax-1]
	}
	cuts := slices.Concat([]float64{-depEdge, -depTailFar, -depTailNear, depTailNear, depTailFar}, kinks, []float64{depEdge})
	slices.Sort(cuts)
	return slices.Compact(cuts)
}

// pieceSettle chooses the split rule of a model's one group piece by
// piece on the n-cell grid: each piece's share of every key's mixture is
// taken on its rule and one of twice the nodes, and while the pieces'
// changes, relative to each figure, sum past valueSettle, those past an
// even share of it double. A narrow piece, its rule fixed, and the far
// tails settle on few nodes, the bulk on more. The rule is each piece's
// finer one.
func (m *Compiled) pieceSettle(keys []delayKey, lam []float64, kinks []float64, n int, scores *scoreCache) ([]*Compiled, []float64, bool) {
	ps := &pieceShares{m: m, cuts: splitCuts(kinks), keys: keys, lam: lam, n: n, scores: scores, memo: map[[2]int]pieceShare{}}
	pieces := len(ps.cuts) - 1
	nodes := make([]int, pieces)
	for p := range nodes {
		nodes[p] = depSplitStart
	}
	// bound: the doublings from depSplitStart to depMixMax a piece
	for range 2 * pieces * 8 {
		change, fine, ok := ps.round(nodes)
		if !ok {
			return nil, nil, false
		}
		sum := 0.0
		for _, c := range change {
			sum += c
		}
		if sum <= valueSettle {
			var cs []*Compiled
			var cw []float64
			for _, sh := range fine {
				cs, cw = append(cs, sh.cs...), append(cw, sh.ws...)
			}
			return cs, cw, true
		}
		if !doubleMoved(nodes, change) {
			return nil, nil, false
		}
	}
	return nil, nil, false
}

// doubleMoved doubles the nodes of each piece whose change is past an
// even share of valueSettle; false when none is, or one would pass
// depMixMax.
func doubleMoved(nodes []int, change []float64) bool {
	moved := false
	for p, c := range change {
		if c > valueSettle/float64(len(nodes)) {
			if nodes[p] *= 2; 2*nodes[p] > depMixMax {
				return false
			}
			moved = true
		}
	}
	return moved
}

// pieceShare is a piece's rule's points and its share of each key's
// mixture.
type pieceShare struct {
	cs []*Compiled
	ws []float64
	v  []float64
}

// pieceShares takes the pieces' shares of a model's mixtures on an
// n-cell grid, each piece and rule once.
type pieceShares struct {
	m      *Compiled
	cuts   []float64
	keys   []delayKey
	lam    []float64
	n      int
	scores *scoreCache
	memo   map[[2]int]pieceShare
}

// at is piece p's share on its k-node rule.
func (ps *pieceShares) at(p, k int) (pieceShare, bool) {
	if sh, ok := ps.memo[[2]int{p, k}]; ok {
		return sh, true
	}
	x, w := pieceRule(ps.cuts[p], ps.cuts[p+1], k)
	cs, cw, ok := ps.m.depPoints([][]float64{x}, [][]float64{w}, ps.scores)
	if !ok {
		return pieceShare{}, false
	}
	sh := pieceShare{cs: cs, ws: cw, v: mixAt(cs, cw, ps.keys, ps.lam, ps.n)}
	ps.memo[[2]int{p, k}] = sh
	return sh, true
}

// round is each piece's change from its rule to one of twice the nodes,
// relative to each figure over the pieces on the finer rules, and the
// finer rules' shares.
func (ps *pieceShares) round(nodes []int) ([]float64, []pieceShare, bool) {
	total := make([]float64, len(ps.keys))
	fine := make([]pieceShare, len(nodes))
	for p, k := range nodes {
		sh, ok := ps.at(p, 2*k)
		if !ok {
			return nil, nil, false
		}
		fine[p] = sh
		for i, v := range sh.v {
			total[i] += v
		}
	}
	change := make([]float64, len(nodes))
	for p, k := range nodes {
		coarse, ok := ps.at(p, k)
		if !ok {
			return nil, nil, false
		}
		change[p] = shareChange(coarse.v, fine[p].v, total)
	}
	return change, fine, true
}

// shareChange is the most a piece's shares on two rules differ by, over
// the figures, relative to each figure over the pieces; infinite where a
// figure is 0 and the shares differ, or is not a number.
func shareChange(a, b, total []float64) float64 {
	worst := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		switch {
		case d == 0:
		case total[i] == 0 || math.IsNaN(d) || math.IsNaN(total[i]):
			return math.Inf(1)
		default:
			worst = math.Max(worst, d/math.Abs(total[i]))
		}
	}
	return worst
}

// depTailNear and depTailFar cut W, at ± each, where the normal's tails
// start, so that a figure that grows with a member there, its weight far
// out, has nodes of its own in each tail.
const (
	depTailNear = 3
	depTailFar  = 5
)

// depMeasure is the Gauss–Legendre points a piece's normal density is
// taken on for its Gauss rule.
const depMeasure = 512

// pieceRule is the rule over W on [a, b]: the n-node Gauss rule of the
// normal density there, scaled to its mass, or on a piece narrower than
// depNarrow depSplitStart Gauss–Legendre nodes weighted by the density.
func pieceRule(a, b float64, n int) (x, w []float64) {
	k := n
	if b-a < depNarrow {
		k = min(n, depSplitStart)
	}
	m := depMeasure
	if b-a < depNarrow {
		m = k
	}
	lx, lw := make([]float64, m), make([]float64, m)
	dist.LegendreRule(lx, lw)
	mass := 0.0
	for i := range lx {
		lx[i] = a + (b-a)*lx[i]
		lw[i] *= (b - a) * math.Exp(-lx[i]*lx[i]/2) / math.Sqrt(2*math.Pi)
		mass += lw[i]
	}
	if m == k {
		return lx, lw
	}
	x, w = make([]float64, k), make([]float64, k)
	if !dist.GaussRule(lx, lw, x, w) {
		return lx, lw
	}
	for i := range w {
		w[i] *= mass
	}
	return x, w
}

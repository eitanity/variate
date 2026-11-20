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
// crosses 0 or ±1; nil without one, false past depPathsMax.
func (m *Compiled) groupKinks(gi int32) ([]float64, bool) {
	paths, ok := m.groupPaths()
	if !ok {
		return nil, false
	}
	sq := m.groups[gi].sq
	var kinks []float64
	for _, ps := range paths {
		for i, a := range ps {
			for _, b := range ps[:i] {
				if g, ok := m.gapOf(gi, a, b); ok {
					kinks = append(kinks, g.crossings(m, sq, 1)...)
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

// splitRule is a group's rule over W split at its kinks: n
// Gauss–Legendre nodes a piece of [−depEdge, depEdge], each weighted by
// the normal density; the depPiecesMax − 1 kinks nearest 0 are kept.
func splitRule(kinks []float64, n int) (x, w []float64) {
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
		slices.Sort(kinks)
	}
	cuts := slices.Concat([]float64{-depEdge}, kinks, []float64{depEdge})
	lx, lw := make([]float64, n), make([]float64, n)
	dist.LegendreRule(lx, lw)
	for p := range len(cuts) - 1 {
		a, b := cuts[p], cuts[p+1]
		for i := range lx {
			at := a + (b-a)*lx[i]
			x = append(x, at)
			w = append(w, lw[i]*(b-a)*math.Exp(-at*at/2)/math.Sqrt(2*math.Pi))
		}
	}
	return x, w
}

package stats

import (
	"math"
	"math/bits"
)

// LogHist is a histogram with bins of fixed relative width over
// [lo, hi): bin k covers [lo·(1+res)^k, lo·(1+res)^(k+1)). Exact zeros,
// non-zero values below lo and values at or above hi are counted apart.
// An integer histogram adds exact buckets for 0 … exact−1 before the log
// bins and has no lower edge. Counts are integers, so merges are exact.
type LogHist struct {
	lo, hi    float64
	logLo     float64 // ln lo
	invLogRes float64 // 1 / ln(1+res)
	logRes    float64 // ln(1+res)
	exact     []int64 // integer buckets, nil for a continuous histogram
	bins      []int64
	Zero      int64
	Under     int64
	Over      int64
	N         int64      // every Add
	pair      *coarseMap // on a coarse copy: its bin for each bin of the fine histogram
}

// coarseMap maps the bins of a fine histogram onto its coarse copy's.
type coarseMap struct {
	exact []int32 // the coarse bin of each exact bucket
	div   int     // fine log bins per coarse bin
	inv   uint64  // ⌊2⁶⁴/div⌋ + 1, 0 when div is 1
	off   int     // the coarse bin of the first fine log bin
}

// bin is the coarse bin of fine log bin k: off + k/div, the division
// done as a multiplication by the reciprocal, exact for k below 2⁶⁴/div.
func (m *coarseMap) bin(k int) int {
	if m.inv == 0 {
		return m.off + k
	}
	q, _ := bits.Mul64(uint64(k), m.inv) //nolint:gosec // k is a bin index, not negative
	return m.off + int(q)                //nolint:gosec // q ≤ k
}

// AddPair counts x in h and in c, a coarse copy of h from NewCoarse,
// finding c's bin from h's.
func (h *LogHist) AddPair(x float64, c *LogHist) {
	h.N++
	c.N++
	switch {
	case x == 0:
		h.Zero++
		c.Zero++
	case h.exact != nil && x < float64(len(h.exact)):
		j := int(x)
		h.exact[j]++
		c.bins[c.pair.exact[j]]++
	case x < h.lo:
		h.Under++
		c.Under++
	case x >= h.hi:
		h.Over++
		c.Over++
	default:
		k := h.index(x)
		h.bins[k]++
		c.bins[c.pair.bin(k)]++
	}
}

// Add counts x.
func (h *LogHist) Add(x float64) {
	h.N++
	switch {
	case x == 0:
		h.Zero++
	case h.exact != nil && x < float64(len(h.exact)):
		h.exact[int(x)]++
	case x < h.lo:
		h.Under++
	case x >= h.hi:
		h.Over++
	default:
		h.bins[h.index(x)]++
	}
}

// index is the bin of a value in [lo, hi).
func (h *LogHist) index(x float64) int {
	k := int((math.Log(x) - h.logLo) * h.invLogRes)
	if k >= len(h.bins) {
		k = len(h.bins) - 1
	}
	if k < 0 {
		k = 0
	}
	return k
}

// Merge adds another histogram with the same layout.
func (h *LogHist) Merge(o *LogHist) {
	if o.N == 0 {
		return // every count moves with N, so o is empty
	}
	h.N += o.N
	h.Zero += o.Zero
	h.Under += o.Under
	h.Over += o.Over
	for i, c := range o.bins {
		h.bins[i] += c
	}
	for i, c := range o.exact {
		h.exact[i] += c
	}
}

// Reset clears every count.
func (h *LogHist) Reset() {
	if h.N == 0 {
		return
	}
	h.N, h.Zero, h.Under, h.Over = 0, 0, 0, 0
	for i := range h.bins {
		h.bins[i] = 0
	}
	for i := range h.exact {
		h.exact[i] = 0
	}
}

// Edge returns the lower edge of bin k, adjusted so that a value equal to
// it is counted in bin k and not the one below.
func (h *LogHist) Edge(k int) float64 {
	e := math.Exp(h.logLo + float64(k)*h.logRes)
	// bound: a few ulps at most
	for i := 0; i < 8 && h.index(e) < k; i++ {
		e = math.Nextafter(e, math.Inf(1))
	}
	return e
}

// MapBytes returns the size of a coarse copy's bin map, which every copy
// made from it by Empty shares.
func (h *LogHist) MapBytes() int64 {
	if h.pair == nil {
		return 0
	}
	return 4 * int64(len(h.pair.exact))
}

// Bins returns the number of log bins.
func (h *LogHist) Bins() int { return len(h.bins) }

// Range returns the covered interval [lo, hi).
func (h *LogHist) Range() (lo, hi float64) { return h.lo, h.hi }

// Quantile returns a value within one bin of the p-quantile, as the
// geometric midpoint of the bin the p-quantile falls in. clipped is set
// when it falls in Under or Over, and the range edge is returned.
func (h *LogHist) Quantile(p float64) (x float64, clipped bool) {
	if h.N == 0 {
		return math.NaN(), false
	}
	target := int64(math.Ceil(p * float64(h.N)))
	if target < 1 {
		target = 1
	}
	cum := h.Zero
	if cum >= target {
		return 0, false
	}
	for k, c := range h.exact {
		cum += c
		if cum >= target {
			return float64(k), false
		}
	}
	cum += h.Under
	if cum >= target {
		return h.lo, true
	}
	for k, c := range h.bins {
		cum += c
		if cum >= target {
			return math.Exp(h.logLo + (float64(k)+0.5)*h.logRes), false
		}
	}
	return h.hi, true
}

// QuantileInterval returns the order-statistic interval of the
// p-quantile: the quantiles at p ∓ d, clamped to [0, 1], where d is the
// interval's half-width in probability, z·√(p(1−p)/N) for independent
// values. It needs no density estimate, and it is asymmetric where the
// distribution is skewed, so it holds its coverage in the tail where a
// symmetric interval of the same width falls short. clipped is set when
// either end falls in Under or Over.
func (h *LogHist) QuantileInterval(p, d float64) (lo, hi float64, clipped bool) {
	if h.N == 0 || math.IsNaN(d) {
		return math.NaN(), math.NaN(), false
	}
	lo, c1 := h.Quantile(math.Max(p-d, 0))
	hi, c2 := h.Quantile(math.Min(p+d, 1))
	return lo, hi, c1 || c2
}

// AtOrAbove returns the count of values ≥ x. It is exact when x is 0, an
// exact bucket, a bin edge, or beyond the range; otherwise the bin
// holding x is counted whole, so the answer is off by at most that bin.
func (h *LogHist) AtOrAbove(x float64) int64 {
	switch {
	case x <= 0:
		return h.N
	case h.exact != nil && x < float64(len(h.exact)):
		n := h.Under + h.Over
		for k := int(math.Ceil(x)); k < len(h.exact); k++ {
			n += h.exact[k]
		}
		for _, c := range h.bins {
			n += c
		}
		return n
	case x < h.lo:
		n := h.Under + h.Over
		for _, c := range h.bins {
			n += c
		}
		return n
	case x >= h.hi:
		return h.Over
	default:
		n := h.Over
		for k := h.index(x); k < len(h.bins); k++ {
			n += h.bins[k]
		}
		return n
	}
}

// Exceedance returns P(X ≥ x).
func (h *LogHist) Exceedance(x float64) float64 {
	if h.N == 0 {
		return math.NaN()
	}
	return float64(h.AtOrAbove(x)) / float64(h.N)
}

// Bands fills dst[i] with P(edges[i] ≤ X < edges[i+1]) for each
// consecutive pair; dst must have len(edges)−1 entries.
func (h *LogHist) Bands(edges, dst []float64) {
	for i := range dst {
		dst[i] = h.Exceedance(edges[i]) - h.Exceedance(edges[i+1])
	}
}

// Min returns the smallest bin with a count, for diagnostics.
func (h *LogHist) Count(k int) int64 { return h.bins[k] }

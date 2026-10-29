package stats

import "math"

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
	N         int64 // every Add
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

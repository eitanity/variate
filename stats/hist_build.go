package stats

import (
	"math"
	"math/bits"
)

const (
	// histDecadesBelow and histDecadesAbove set the range around the mean.
	histDecadesBelow = 6
	histDecadesAbove = 9
	// exactBuckets is the number of exact buckets of an integer histogram.
	exactBuckets = 4096
)

// LogRange is the range of a log histogram around a mean.
func LogRange(mean float64) (lo, hi float64) {
	if !(mean > 0) {
		return math.Inf(1), math.Inf(1)
	}
	return mean * math.Pow(10, -histDecadesBelow), mean * math.Pow(10, histDecadesAbove)
}

// NewLogHist allocates a histogram over [1e-6·mean, 1e9·mean) with
// relative bin width res. A zero mean gives a histogram with no bins.
func NewLogHist(mean, res float64) *LogHist {
	lo, hi := LogRange(mean)
	if math.IsInf(lo, 1) {
		return &LogHist{lo: lo, hi: hi}
	}
	return newHist(lo, hi, res, nil)
}

// NewIntHist allocates an integer histogram: exact buckets for
// 0 … 4095, then log bins up to 1e9·mean.
func NewIntHist(mean, res float64) *LogHist {
	exact := make([]int64, exactBuckets)
	hi := mean * math.Pow(10, histDecadesAbove)
	if hi <= exactBuckets {
		return &LogHist{lo: exactBuckets, hi: exactBuckets, exact: exact}
	}
	return newHist(exactBuckets, hi, res, exact)
}

// NewCoarse allocates a continuous histogram over h's range whose bins
// are whole runs of h's, of relative width near res, so that AddPair
// finds its bin from h's without a second logarithm. An integer
// histogram's coarse copy reaches down below the count 1 and holds its
// exact buckets in its log bins.
func NewCoarse(h *LogHist, res float64) *LogHist {
	if h.exact == nil && (math.IsInf(h.lo, 0) || len(h.bins) == 0) {
		return &LogHist{lo: h.lo, hi: h.hi}
	}
	div := 1
	if h.logRes > 0 {
		div = max(1, int(math.Round(math.Log1p(res)/h.logRes)))
	}
	logRes := math.Log1p(res)
	if h.logRes > 0 {
		logRes = float64(div) * h.logRes
	}
	// below is the number of coarse bins under h.lo: none for a
	// continuous histogram, enough to reach 0.5 for an integer one.
	below := 0
	if h.exact != nil {
		below = int(math.Ceil(math.Log(h.lo/0.5) / logRes))
	}
	logLo := math.Log(h.lo) - float64(below)*logRes
	n := below + (len(h.bins)+div-1)/div
	c := &LogHist{
		lo: math.Exp(logLo), hi: h.hi, logLo: logLo, logRes: logRes, invLogRes: 1 / logRes,
		bins: make([]int64, n),
		pair: newCoarseMap(div, below),
	}
	if h.exact != nil {
		c.pair.exact = make([]int32, len(h.exact))
		for j := 1; j < len(h.exact); j++ {
			c.pair.exact[j] = int32(c.index(float64(j))) //nolint:gosec // a bin index of a few hundred
		}
	}
	return c
}

// Empty returns a histogram with h's layout and no counts. A coarse copy
// shares its bin map with h.
func (h *LogHist) Empty() *LogHist {
	e := *h
	e.bins = make([]int64, len(h.bins))
	if h.exact != nil {
		e.exact = make([]int64, len(h.exact))
	}
	e.N, e.Zero, e.Under, e.Over = 0, 0, 0, 0
	return &e
}

// newCoarseMap maps div fine log bins to each coarse bin, the first at
// coarse bin off.
func newCoarseMap(div, off int) *coarseMap {
	m := &coarseMap{div: div, off: off}
	if div > 1 {
		q, _ := bits.Div64(1, 0, uint64(div))
		m.inv = q + 1
	}
	return m
}

func newHist(lo, hi, res float64, exact []int64) *LogHist {
	logRes := math.Log1p(res)
	n := int(math.Ceil(math.Log(hi/lo) / logRes))
	return &LogHist{
		lo: lo, hi: hi, logLo: math.Log(lo), logRes: logRes, invLogRes: 1 / logRes,
		bins: make([]int64, n), exact: exact,
	}
}

// BinsFor returns the number of log bins a histogram with these
// parameters would allocate, for memory planning.
func BinsFor(res float64) int {
	return int(math.Ceil(float64(histDecadesBelow+histDecadesAbove) * math.Ln10 / math.Log1p(res)))
}

package stats

import "math"

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

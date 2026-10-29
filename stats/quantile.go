package stats

import "math"

// QuantileSorted returns the p-quantile of a sorted sample by the inverse
// of the empirical CDF (Hyndman–Fan type 1): the smallest value with at
// least a fraction p of the sample at or below it.
func QuantileSorted(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return math.NaN()
	}
	k := int(math.Ceil(p * float64(n)))
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	return sorted[k-1]
}

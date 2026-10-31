package variate

import (
	"math"
	"testing"
)

// TestArrivalTimeInPeriod: an arrival at the largest fraction of a
// period stays in that period, however large its number.
func TestArrivalTimeInPeriod(t *testing.T) {
	top := math.Nextafter(1, 0) // the largest Float64 draw
	for _, p := range []int64{0, 1, 1 << 20, 87788785, 1 << 30, 1 << 40, 1 << 51} {
		for _, u := range []float64{0, 0.5, top} {
			if at := arrivalTime(p, u); !(at >= float64(p) && at < float64(p)+1) {
				t.Errorf("period %d, fraction %g: time %.17g", p, u, at)
			}
		}
	}
}

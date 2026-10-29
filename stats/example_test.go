package stats_test

import (
	"fmt"

	"code.eitanity.com/variate/stats"
)

// A log histogram streams values into bins of fixed relative width; its
// quantiles are exact to that width, and bands read off the curve.
func ExampleLogHist() {
	h := stats.NewLogHist(100, 1e-3) // range sized from a mean of 100
	for i := 1; i <= 1000; i++ {
		h.Add(float64(i))
	}
	p50, clipped := h.Quantile(0.5)
	p99, _ := h.Quantile(0.99)
	fmt.Printf("p50 %.0f  p99 %.0f  clipped %v\n", p50, p99, clipped)
	bands := make([]float64, 3)
	h.Bands([]float64{0, 250, 500, 1000}, bands)
	fmt.Printf("bands %.3f %.3f %.3f\n", bands[0], bands[1], bands[2])
	fmt.Printf("P(X >= 900) = %.3f\n", h.Exceedance(900))
	// Output:
	// p50 500  p99 990  clipped false
	// bands 0.249 0.250 0.500
	// P(X >= 900) = 0.101
}

// Welford keeps a running mean and variance; Neumaier a compensated sum
// that keeps the small terms a plain sum loses.
func ExampleWelford() {
	var w stats.Welford
	var n stats.Neumaier
	plain := 0.0
	for _, x := range []float64{2, 4, 4, 4, 5, 5, 7, 9} {
		w.Add(x)
	}
	for i := 0; i < 1000; i++ {
		n.Add(1e16)
		n.Add(1)
		n.Add(-1e16)
		plain += 1e16
		plain += 1
		plain += -1e16
	}
	fmt.Printf("mean %.1f  sd %.1f\n", w.Mean, w.StdDev())
	fmt.Printf("compensated sum %.0f  plain sum %.0f\n", n.Value(), plain)
	// Output:
	// mean 5.0  sd 2.1
	// compensated sum 1000  plain sum 0
}

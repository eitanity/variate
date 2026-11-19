package dist_test

import (
	"fmt"
	"math"
	"math/rand/v2"

	"code.eitanity.com/variate/dist"
)

// A PERT is built from its low, most likely and high values; the fourth
// parameter is the shape, usually dist.DefaultPERTShape.
func ExampleNewPERT() {
	d, err := dist.NewPERT(1000, 2500, 9000, dist.DefaultPERTShape)
	if err != nil {
		fmt.Println(err)
		return
	}
	lo, hi := d.Support()
	fmt.Printf("mean %.1f  sd %.1f  support [%g, %g]\n", d.Mean(), math.Sqrt(d.Var()), lo, hi)
	fmt.Printf("P(X <= 2500) = %.4f\n", d.CDF(2500))
	// Output:
	// mean 3333.3  sd 1374.4  support [1000, 9000]
	// P(X <= 2500) = 0.3216
}

// Draws come from Fill, given a generator; the engine seeds one per
// draw from the draw's key.
func ExampleDistribution_fill() {
	d, _ := dist.NewPERT(1, 2, 4, dist.DefaultPERTShape)
	xs := make([]float64, 100000)
	d.Fill(rand.New(rand.NewPCG(7, 0)), xs)
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	fmt.Printf("sample mean %.2f, exact %.2f\n", sum/100000, d.Mean())
	// Output:
	// sample mean 2.17, exact 2.17
}

// Truncation restricts a distribution to an interval and renormalises;
// capping clamps draws at a ceiling, leaving an atom there.
func ExampleNewTruncated() {
	base, _ := dist.NewLognormalFromMedianP90(4, 24)
	t, err := dist.NewTruncated(base, 1, 48)
	if err != nil {
		fmt.Println(err)
		return
	}
	c, _ := dist.NewCapped(base, 48)
	fmt.Printf("lognormal mean %.2f\n", base.Mean())
	fmt.Printf("truncated to [1, 48]: mean %.2f, P(X < 1) = %.0f\n", t.Mean(), t.CDF(1))
	fmt.Printf("capped at 48: mean %.2f, P(X = 48) = %.3f\n", c.Mean(), 1-c.CDF(47.999999))
	// Output:
	// lognormal mean 10.63
	// truncated to [1, 48]: mean 8.48, P(X < 1) = 0
	// capped at 48: mean 8.70, P(X = 48) = 0.038
}

// Fill draws straight from a distribution when no Sampler is wanted.
func ExampleDistribution() {
	d, _ := dist.NewExponential(2)
	r := rand.New(rand.NewPCG(1, 2))
	buf := make([]float64, 4)
	d.Fill(r, buf)
	fmt.Println(len(buf), buf[0] > 0)
	// Output:
	// 4 true
}

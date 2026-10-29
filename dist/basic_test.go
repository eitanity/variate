package dist

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func TestConstant(t *testing.T) {
	d, err := NewConstant(3.5)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range draws(d, 1, 100) {
		if x != 3.5 {
			t.Fatal(x)
		}
	}
	if d.Mean() != 3.5 || d.Var() != 0 || d.CDF(3.4) != 0 || d.CDF(3.5) != 1 || d.Quantile(0.3) != 3.5 {
		t.Error("constant")
	}
	if _, err := NewConstant(math.Inf(1)); err == nil {
		t.Error("accepted Inf")
	}
}

func TestUniformSuite(t *testing.T) {
	seed := uint64(5000)
	for _, c := range [][2]float64{{0, 1}, {-3, 2}, {1e-6, 2e-6}, {1e12, 3e12}} {
		d, err := NewUniform(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("uniform(%g,%g)", c[0], c[1]), d, d, seed, false)
		seed++
	}
	for _, c := range [][2]float64{{1, 1}, {2, 1}, {-1e308, 1e308}, {math.NaN(), 1}} {
		if _, err := NewUniform(c[0], c[1]); err == nil {
			t.Errorf("NewUniform(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestLogUniformSuite(t *testing.T) {
	seed := uint64(5100)
	for _, c := range [][2]float64{{1, 10}, {1e-6, 1e12}, {5, 5.001}} {
		d, err := NewLogUniform(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("loguniform(%g,%g)", c[0], c[1]), d, d, seed, false)
		seed++
	}
	for _, c := range [][2]float64{{0, 1}, {-1, 1}, {2, 1}, {1, math.Inf(1)}} {
		if _, err := NewLogUniform(c[0], c[1]); err == nil {
			t.Errorf("NewLogUniform(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestTriangularSuite(t *testing.T) {
	seed := uint64(5200)
	for _, c := range [][3]float64{{0, 0.5, 1}, {0, 0, 1}, {0, 1, 1}, {1e3, 2e3, 1e12}, {-5, -4, 3}} {
		d, err := NewTriangular(c[0], c[1], c[2])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("triangular(%g,%g,%g)", c[0], c[1], c[2]), d, d, seed, false)
		seed++
	}
	for _, c := range [][3]float64{{1, 0, 2}, {0, 3, 2}, {1, 1, 1}, {0, math.NaN(), 1}} {
		if _, err := NewTriangular(c[0], c[1], c[2]); err == nil {
			t.Errorf("NewTriangular(%v) accepted", c)
		}
	}
}

func TestExponentialSuite(t *testing.T) {
	seed := uint64(5300)
	for _, m := range []float64{1e-6, 1, 1e12} {
		d, err := NewExponential(m)
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("exponential(%g)", m), d, d, seed, false)
		seed++
	}
	for _, m := range []float64{0, -1, math.Inf(1)} {
		if _, err := NewExponential(m); err == nil {
			t.Errorf("NewExponential(%g) accepted", m)
		}
	}
}

func TestLognormalSuite(t *testing.T) {
	seed := uint64(5400)
	for _, c := range [][2]float64{{0, 1}, {-14, 0.1}, {27, 2.5}, {5, 0.01}} {
		d, err := NewLognormal(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("lognormal(%g,%g)", c[0], c[1]), d, d, seed, false)
		seed++
	}
	for _, c := range [][2]float64{{0, 0}, {0, -1}, {math.Inf(1), 1}} {
		if _, err := NewLognormal(c[0], c[1]); err == nil {
			t.Errorf("NewLognormal(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestWeibullSuite(t *testing.T) {
	seed := uint64(5500)
	for _, c := range [][2]float64{{0.5, 1}, {1, 1e-6}, {1.5, 1e12}, {7, 3}} {
		d, err := NewWeibull(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("weibull(%g,%g)", c[0], c[1]), d, d, seed, false)
		seed++
	}
	for _, c := range [][2]float64{{0, 1}, {1, 0}, {-1, 1}} {
		if _, err := NewWeibull(c[0], c[1]); err == nil {
			t.Errorf("NewWeibull(%g, %g) accepted", c[0], c[1])
		}
	}
}

// TestZeroUniform forces r.Float64() = 0 on every sampler that takes a
// logarithm or a negative power of the uniform: no draw may be infinite.
func TestZeroUniform(t *testing.T) {
	r := rand.New(zeroSource{})
	w, _ := NewWeibull(0.5, 1)
	p, _ := NewPareto(1, 1.5)
	g, _ := NewGPD(0, 1, 0.5)
	gn, _ := NewGPD(0, 1, -0.5)
	tn, _ := NewTruncatedNormal(0, 1, 6, math.Inf(1))
	e, _ := NewExponential(1)
	for _, d := range []Distribution{w, p, g, gn, tn, e} {
		buf := make([]float64, 4)
		d.Fill(r, buf)
		lo, hi := d.Support()
		boundsTest(t, fmt.Sprintf("%T with U=0", d), buf, lo, hi)
	}
}

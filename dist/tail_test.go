package dist

import (
	"fmt"
	"math"
	"testing"
)

func TestParetoSuite(t *testing.T) {
	seed := uint64(5800)
	for _, c := range [][2]float64{{1, 0.5}, {1, 1}, {1e-6, 1.5}, {1e12, 2}, {3, 2.5}, {1, 10}} {
		d, err := NewPareto(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("pareto(%g,%g)", c[0], c[1]), d, d, seed, c[1] <= 4)
		seed++
	}
	d, _ := NewPareto(1, 1)
	if !math.IsInf(d.Mean(), 1) || !math.IsInf(d.Var(), 1) {
		t.Error("pareto(1,1) moments")
	}
	d, _ = NewPareto(1, 1.5)
	if !(d.Mean() > 0) || !math.IsInf(d.Var(), 1) {
		t.Error("pareto(1,1.5) moments")
	}
	for _, c := range [][2]float64{{0, 1}, {1, 0}, {-1, 1}} {
		if _, err := NewPareto(c[0], c[1]); err == nil {
			t.Errorf("NewPareto(%g, %g) accepted", c[0], c[1])
		}
	}
}

// TestLogLogisticSuite: the family suite at shapes with an infinite mean,
// a finite mean and infinite variance, and a finite variance, and its
// median is its scale.
func TestLogLogisticSuite(t *testing.T) {
	seed := uint64(5850)
	for _, c := range [][2]float64{{0.8, 1}, {1.5, 5}, {3, 1e-6}, {3, 1e12}, {8, 2}} {
		d, err := NewLogLogistic(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("loglogistic(%g,%g)", c[0], c[1]), d, d, seed, c[0] <= 4)
		if m := d.Quantile(0.5); math.Abs(m-c[1]) > 1e-15*c[1] {
			t.Errorf("loglogistic(%g,%g): median %g", c[0], c[1], m)
		}
		seed++
	}
	d, _ := NewLogLogistic(1, 2)
	if !math.IsInf(d.Mean(), 1) || !math.IsInf(d.Var(), 1) {
		t.Error("loglogistic(1,2) moments")
	}
	d, _ = NewLogLogistic(2, 2)
	if math.Abs(d.Mean()-math.Pi) > 1e-15 || !math.IsInf(d.Var(), 1) {
		t.Errorf("loglogistic(2,2): mean %v, var %v; want π and +Inf", d.Mean(), d.Var())
	}
	for _, c := range [][2]float64{{0, 1}, {1, 0}, {-1, 1}, {math.Inf(1), 1}} {
		if _, err := NewLogLogistic(c[0], c[1]); err == nil {
			t.Errorf("NewLogLogistic(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestGPDSuite(t *testing.T) {
	seed := uint64(5900)
	for _, c := range [][3]float64{{0, 1, 0}, {0, 1, 0.3}, {0, 1, 0.5}, {0, 1, 1}, {0, 1, 1.5}, {0, 1, -0.5}, {0, 1, -2}, {10, 1e-6, 0.1}, {0, 1e12, 0.25}, {0, 1, 1e-9}} {
		d, err := NewGPD(c[0], c[1], c[2])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("gpd(%g,%g,%g)", c[0], c[1], c[2]), d, d, seed, c[2] >= 0.25)
		seed++
	}
	d, _ := NewGPD(0, 1, -0.5)
	if _, hi := d.Support(); hi != 2 {
		t.Errorf("gpd(0,1,-0.5) upper bound %g", hi)
	}
	for _, c := range [][3]float64{{0, 0, 0}, {math.NaN(), 1, 0}, {0, 1, math.Inf(1)}} {
		if _, err := NewGPD(c[0], c[1], c[2]); err == nil {
			t.Errorf("NewGPD(%v) accepted", c)
		}
	}
}

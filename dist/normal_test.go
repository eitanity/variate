package dist

import (
	"fmt"
	"math"
	"testing"
)

func TestNormalSuite(t *testing.T) {
	seed := uint64(5600)
	for _, c := range [][2]float64{{0, 1}, {-1e6, 1e-3}, {1e12, 1e9}} {
		d, err := NewNormal(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		runSuite(t, fmt.Sprintf("normal(%g,%g)", c[0], c[1]), d, d, seed, false)
		seed++
	}
	if _, err := NewNormal(0, 0); err == nil {
		t.Error("accepted sigma 0")
	}
}

func TestTruncatedNormalSuite(t *testing.T) {
	inf := math.Inf(1)
	cases := [][4]float64{
		{0, 1, -1, 1}, {0, 1, 0, inf}, {0, 1, -inf, 0.5}, {0, 1, 3, 4}, {0, 1, 6, inf},
		{0, 1, 6, 6.5}, {0, 1, -7, -6}, {0, 1, -inf, -6}, {0, 1, 10, inf}, {0, 1, 30, inf},
		{100, 20, 50, 130}, {1e-3, 1e-4, 0, inf}, {5, 1, 4.9, 5.1},
	}
	seed := uint64(5700)
	for _, c := range cases {
		name := fmt.Sprintf("truncnormal(%g,%g,[%g,%g])", c[0], c[1], c[2], c[3])
		d, err := NewTruncatedNormal(c[0], c[1], c[2], c[3])
		if err != nil {
			t.Fatal(name, err)
		}
		runSuite(t, name, d, d, seed, false)
		seed++
	}
	for _, c := range [][4]float64{{0, 1, 1, 1}, {0, 1, 2, 1}, {0, 1, 40, inf}, {0, 1, -inf, -40}, {0, 0, 0, 1}} {
		if _, err := NewTruncatedNormal(c[0], c[1], c[2], c[3]); err == nil {
			t.Errorf("NewTruncatedNormal(%v) accepted", c)
		}
	}
}

// TestTruncatedNormalMomentsAgainstFullNormal checks the closed-form moments
// on an interval wide enough to hold all the mass.
func TestTruncatedNormalMomentsAgainstFullNormal(t *testing.T) {
	d, err := NewTruncatedNormal(2, 3, -100, 100)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(d.Mean()-2) > 1e-12 || math.Abs(d.Var()-9) > 1e-10 {
		t.Errorf("mean %g var %g", d.Mean(), d.Var())
	}
}

package dist

import (
	"fmt"
	"math"
	"testing"
)

func TestGammaSuite(t *testing.T) {
	shapes := []float64{0.3, 0.9, 1, 1.5, 7.5, 100}
	scales := []float64{1e-6, 1, 1e12}
	seed := uint64(1)
	for _, k := range shapes {
		for _, th := range scales {
			name := fmt.Sprintf("gamma(%g,%g)", k, th)
			d, err := NewGamma(k, th)
			if err != nil {
				t.Fatal(err)
			}
			xs := draws(d, seed, nDraws)
			seed++
			boundsTest(t, name, xs, 0, math.Inf(1))
			ksTest(t, name, xs, d.CDF)
			momentsTest(t, name, xs, d.Mean(), d.Var())
			inverseTest(t, name, d, d, 1e-12, 0)
		}
	}
}

func TestGammaRejectsParams(t *testing.T) {
	for _, c := range [][2]float64{{0, 1}, {-1, 1}, {1, 0}, {math.NaN(), 1}, {1, math.Inf(1)}} {
		if _, err := NewGamma(c[0], c[1]); err == nil {
			t.Errorf("NewGamma(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestBetaSuite(t *testing.T) {
	cases := [][2]float64{{0.3, 0.5}, {0.5, 3}, {1, 0.7}, {1, 1}, {2, 5}, {50, 50}, {0.2, 0.2}}
	seed := uint64(100)
	for _, c := range cases {
		name := fmt.Sprintf("beta(%g,%g)", c[0], c[1])
		d, err := NewBeta(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		boundsTest(t, name, xs, 0, 1)
		ksTest(t, name, xs, d.CDF)
		momentsTest(t, name, xs, d.Mean(), d.Var())
		inverseTest(t, name, d, d, 1e-12, 0)
	}
}

// TestBetaMassAtOne covers shapes far below 1, where a sizeable share of the
// mass lies within one ulp of 1 and rounds to exactly 1.0. The KS test
// cannot express that, so the check is that exact ones occur with the
// frequency the survival function gives at the last representable value.
func TestBetaMassAtOne(t *testing.T) {
	d, err := NewBeta(0.05, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	xs := draws(d, 7, nDraws)
	ones := 0
	for _, x := range xs {
		if x == 1 {
			ones++
		}
	}
	p := d.Survival(math.Nextafter(1, 0))
	n := float64(len(xs))
	if z := (float64(ones) - n*p) / math.Sqrt(n*p*(1-p)); math.Abs(z) > zLimit {
		t.Errorf("exact ones %d, expected %.0f, z=%.2f", ones, n*p, z)
	}
}

func TestBetaRejectsParams(t *testing.T) {
	for _, c := range [][2]float64{{0, 1}, {1, -1}, {math.Inf(1), 1}, {1, math.NaN()}} {
		if _, err := NewBeta(c[0], c[1]); err == nil {
			t.Errorf("NewBeta(%g, %g) accepted", c[0], c[1])
		}
	}
}

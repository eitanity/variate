package dist

import (
	"fmt"
	"math"
	"testing"
)

var pertShapes = []float64{1, 1.5, 3, 5}

var pertScales = [][2]float64{{0, 1}, {1e3, 1e12}, {1e15, 1e15 + 1e3}}

// pertFor builds the PERT with Beta shapes (α, β) on [low, high]: t = α+β−2
// and the peak at the matching position.
func pertFor(t *testing.T, alpha, beta, low, high float64) PERT {
	t.Helper()
	tt := alpha + beta - 2
	peak := low
	if tt > 0 {
		peak = low + (high-low)*(alpha-1)/tt
	}
	d, err := NewPERT(low, peak, high, tt)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestPERTSuite(t *testing.T) {
	seed := uint64(1000)
	for _, a := range pertShapes {
		for _, b := range pertShapes {
			for _, sc := range pertScales {
				name := fmt.Sprintf("pert(%g,%g)[%g,%g]", a, b, sc[0], sc[1])
				d := pertFor(t, a, b, sc[0], sc[1])
				xs := draws(d, seed, nDraws)
				seed++
				boundsTest(t, name, xs, sc[0], sc[1])
				// At magnitude 1e15 a span of 1e3 leaves 13 bits of resolution,
				// so the fit is tested on the Beta variate recovered from the draw.
				if sc[0] >= 1e15 {
					for i, x := range xs {
						xs[i] = (x - sc[0]) / (sc[1] - sc[0])
					}
					ksTest(t, name, xs, d.beta.CDF)
					continue
				}
				ksTest(t, name, xs, d.CDF)
				momentsTest(t, name, xs, d.Mean(), d.Var())
				inverseTest(t, name, d, d, 1e-12, 0)
			}
		}
	}
}

func TestPERTDefaultShapeMoments(t *testing.T) {
	d, err := NewPERT(10, 20, 100, DefaultPERTShape)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := d.Mean(), (10+4*20+100.0)/6; math.Abs(got-want) > 1e-12*want {
		t.Errorf("mean %g want %g", got, want)
	}
	a, b := d.Shape()
	if a+b != 6 {
		t.Errorf("shapes %g %g", a, b)
	}
}

func TestPERTRejectsParams(t *testing.T) {
	cases := [][4]float64{
		{1, 0, 2, 4}, {0, 3, 2, 4}, {1, 1, 1, 4}, {-1e308, 0, 1e308, 4},
		{math.NaN(), 1, 2, 4}, {0, 1, math.Inf(1), 4}, {0, 1, 2, -1}, {0, 1, 2, math.NaN()},
	}
	for _, c := range cases {
		if _, err := NewPERT(c[0], c[1], c[2], c[3]); err == nil {
			t.Errorf("NewPERT(%v) accepted", c)
		}
	}
}

func TestLogPERTSuite(t *testing.T) {
	cases := [][4]float64{
		{1e3, 1e4, 1e9, 4}, {1, 2, 10, 4}, {1e-6, 1e-3, 1e-2, 4}, {1e3, 1e4, 1e9, 1}, {5, 5.5, 6, 4},
	}
	seed := uint64(2000)
	for _, c := range cases {
		name := fmt.Sprintf("logpert(%g,%g,%g,t=%g)", c[0], c[1], c[2], c[3])
		d, err := NewLogPERT(c[0], c[1], c[2], c[3])
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		boundsTest(t, name, xs, c[0], c[2])
		momentsTest(t, name, xs, d.Mean(), d.Var())
		inverseTest(t, name, d, d, 1e-12, 0)
		logs := make([]float64, len(xs))
		for i, x := range xs {
			logs[i] = math.Log10(x)
		}
		ksTest(t, name+" log10", logs, d.lp.CDF)
	}
}

func TestLogPERTRejectsParams(t *testing.T) {
	for _, c := range [][4]float64{{0, 1, 2, 4}, {-1, 1, 2, 4}, {1, 3, 2, 4}, {1, 1, 1, 4}} {
		if _, err := NewLogPERT(c[0], c[1], c[2], c[3]); err == nil {
			t.Errorf("NewLogPERT(%v) accepted", c)
		}
	}
}

// TestLogPERTQuadratureAgainstClosedForm checks the quadrature on a case
// with a known answer: with t = 0 the log10 value is uniform on [l, h], so
// E[10^X] = (10^h − 10^l)/((h − l)·ln 10).
func TestLogPERTQuadratureAgainstClosedForm(t *testing.T) {
	d, err := NewLogPERT(10, 100, 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := (1000 - 10.0) / (2 * math.Ln10)
	if got := d.Mean(); math.Abs(got-want) > 1e-12*want {
		t.Errorf("mean %.17g want %.17g", got, want)
	}
	want2 := (1e6 - 100.0) / (2 * 2 * math.Ln10)
	if got := d.Var() + d.Mean()*d.Mean(); math.Abs(got-want2) > 1e-11*want2 {
		t.Errorf("second moment %.17g want %.17g", got, want2)
	}
}

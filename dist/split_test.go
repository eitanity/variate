package dist

import (
	"math"
	"testing"
)

// TestSplitMixesBack: for continuous, capped, finite, integer and
// constant families, the halves' masses add to 1, and their means and
// second moments mix back to the whole law's; a continuous half's
// moments are the truncation's, a discrete one's the direct sum over its
// side, and an atom at the threshold goes below.
func TestSplitMixesBack(t *testing.T) {
	ln, _ := NewLognormalFromMedianP90(2000, 8000)
	ga, _ := NewGamma(2.5, 3)
	capped, _ := NewCapped(ln, 6000)
	cat, _ := NewCategorical([]float64{1, 5, 9}, []float64{0.5, 0.3, 0.2})
	po, _ := NewPoisson(4)
	ge, _ := NewGeometric(0.3)
	co, _ := NewConstant(3)
	for _, tc := range []struct {
		name string
		d    Distribution
		t    float64
	}{
		{"lognormal", ln, 5000},
		{"gamma below its median", ga, 2},
		{"capped, threshold below the cap", capped, 5000},
		{"capped, threshold at the cap", capped, 6000},
		{"categorical, atom at t", cat, 5},
		{"poisson", po, 4},
		{"poisson, t between integers", po, 2.5},
		{"geometric", ge, 0},
		{"constant below t", co, 5},
		{"constant above t", co, 1},
	} {
		hi, s, err := NewSplit(tc.d, tc.t, true)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		lo, r, err := NewSplit(tc.d, tc.t, false)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !(math.Abs(s+r-1) <= 1e-12) {
			t.Errorf("%s: masses %g + %g", tc.name, s, r)
		}
		if want := tc.d.Survival(tc.t); !(math.Abs(s-want) <= 1e-12) {
			t.Errorf("%s: P(above) %g, want P(X > t) %g", tc.name, s, want)
		}
		var m1, m2 float64
		for _, h := range []struct {
			d    Truncated
			mass float64
		}{{hi, s}, {lo, r}} {
			if h.mass == 0 {
				continue
			}
			m1 += h.mass * h.d.Mean()
			m2 += h.mass * (h.d.Var() + h.d.Mean()*h.d.Mean())
		}
		mean := tc.d.Mean()
		if !(math.Abs(m1-mean) <= 1e-9*mean) {
			t.Errorf("%s: halves' means mix to %g, want %g", tc.name, m1, mean)
		}
		if second := tc.d.Var() + mean*mean; !(math.Abs(m2-second) <= 1e-8*second) {
			t.Errorf("%s: halves' second moments mix to %g, want %g", tc.name, m2, second)
		}
	}
	// A continuous half is the truncation to its side.
	tr, _ := NewTruncated(ln, 5000, math.Inf(1))
	hi, _, _ := NewSplit(ln, 5000, true)
	if !(math.Abs(hi.Mean()-tr.Mean()) <= 1e-9*tr.Mean()) || !(math.Abs(hi.Var()-tr.Var()) <= 1e-8*tr.Var()) {
		t.Errorf("lognormal above: %g, %g; truncation %g, %g", hi.Mean(), hi.Var(), tr.Mean(), tr.Var())
	}
	// The atom at 5 goes below: the lower half is {1, 5} at 5:3, the upper {9}.
	lo, _, _ := NewSplit(cat, 5, false)
	if want := (0.5*1 + 0.3*5) / 0.8; !(math.Abs(lo.Mean()-want) <= 1e-12) || lo.CDF(5) != 1 || lo.CDF(1) != 0.625 {
		t.Errorf("categorical below: mean %g, want %g; CDF(1) %g, CDF(5) %g", lo.Mean(), want, lo.CDF(1), lo.CDF(5))
	}
	up, _, _ := NewSplit(cat, 5, true)
	if up.Mean() != 9 || up.Var() != 0 {
		t.Errorf("categorical above: mean %g var %g, want 9, 0", up.Mean(), up.Var())
	}
	// Poisson above 4: the direct sum over 5, 6, ….
	pu, s, _ := NewSplit(po, 4, true)
	sum := 0.0
	for k := 5.0; k < 60; k++ {
		sum += k * po.PMF(k)
	}
	if !(math.Abs(pu.Mean()-sum/s) <= 1e-12*pu.Mean()) {
		t.Errorf("poisson above 4: mean %g, direct %g", pu.Mean(), sum/s)
	}
	// A constant leaves one half empty.
	if _, m, _ := NewSplit(co, 5, true); m != 0 {
		t.Errorf("constant 3 above 5: mass %g", m)
	}
}

// TestSplitHeavyTail: a Pareto's upper half keeps its infinite variance
// and a finite mean.
func TestSplitHeavyTail(t *testing.T) {
	pa, _ := NewPareto(1, 1.5)
	hi, s, err := NewSplit(pa, 4, true)
	if err != nil {
		t.Fatal(err)
	}
	// Given X > 4 a Pareto(1, α) is Pareto(4, α): mean 4α/(α − 1).
	if want := 4 * 1.5 / 0.5; !(math.Abs(hi.Mean()-want) <= 1e-9*want) || !math.IsInf(hi.Var(), 1) {
		t.Errorf("mean %g var %g, want %g, +Inf", hi.Mean(), hi.Var(), want)
	}
	if want := math.Pow(4, -1.5); !(math.Abs(s-want) <= 1e-12) {
		t.Errorf("mass %g, want %g", s, want)
	}
	if _, _, err := NewSplit(pa, math.NaN(), true); err == nil {
		t.Error("a NaN threshold is accepted")
	}
}

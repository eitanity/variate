package dist

import (
	"math"
	"testing"
)

func TestTruncatedSuite(t *testing.T) {
	inf := math.Inf(1)
	n01, _ := NewNormal(0, 1)
	ln, _ := NewLognormal(0, 1)
	g, _ := NewGamma(2.5, 3)
	pe, _ := NewPERT(0, 2, 10, 4)
	pa, _ := NewPareto(1, 1.5)
	cases := []struct {
		name   string
		d      Continuous
		lo, hi float64
	}{
		{"normal[0,inf]", n01, 0, inf},
		{"normal[-1,2]", n01, -1, 2},
		{"normal[3,inf]", n01, 3, inf},
		{"lognormal beyond 1-1e-8", ln, ln.Quantile(1 - 1e-8), inf},
		{"lognormal[0.5,2]", ln, 0.5, 2},
		{"gamma[1,4]", g, 1, 4},
		{"gamma[40,inf]", g, 40, inf},
		{"pert[1,3]", pe, 1, 3},
		{"pert[9,10]", pe, 9, 10},
		{"pareto[2,5]", pa, 2, 5},
		{"pareto[1e4,inf]", pa, 1e4, inf},
	}
	seed := uint64(7000)
	for _, c := range cases {
		d, err := NewTruncated(c.d, c.lo, c.hi)
		if err != nil {
			t.Fatal(c.name, err)
		}
		// The probability line F(lo) + p·mass resolves p no finer than an
		// ulp of F(lo) or S(hi) over the mass.
		absTol := ulps * math.Max(d.pr.pLo, d.pr.sLo) / d.mass
		runSuiteAbs(t, "truncated "+c.name, d, d, seed, false, absTol)
		seed++
	}
}

// TestTruncatedMomentsAgainstClosedForm compares the quadrature moments of
// a truncated Normal with TruncatedNormal's closed forms.
func TestTruncatedMomentsAgainstClosedForm(t *testing.T) {
	inf := math.Inf(1)
	n, _ := NewNormal(2, 3)
	for _, c := range [][2]float64{{-1, 1}, {0, inf}, {-inf, 1}, {8, inf}, {-5, 20}, {5, 5.5}} {
		tr, err := NewTruncated(n, c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		tn, err := NewTruncatedNormal(2, 3, c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		if !near(tr.Mean(), tn.Mean(), 1e-11) || !near(tr.Var(), tn.Var(), 1e-10) {
			t.Errorf("[%g,%g]: quadrature mean %.15g var %.15g, closed form %.15g %.15g", c[0], c[1], tr.Mean(), tr.Var(), tn.Mean(), tn.Var())
		}
	}
}

func TestTruncatedRejects(t *testing.T) {
	n, _ := NewNormal(0, 1)
	if _, err := NewTruncated(n, 1, 1); err == nil {
		t.Error("accepted lo == hi")
	}
	if _, err := NewTruncated(n, 7, 8); err == nil {
		t.Error("accepted a range with mass below 1e-9")
	}
	pe, _ := NewPERT(0, 1, 2, 4)
	if _, err := NewTruncated(pe, 3, 4); err == nil {
		t.Error("accepted a range outside the support")
	}
}

func TestCappedSuite(t *testing.T) {
	ln, _ := NewLognormal(0, 1)
	pa, _ := NewPareto(1, 0.8)
	g, _ := NewGamma(0.5, 2)
	pe, _ := NewPERT(0, 2, 10, 4)
	cases := []struct {
		name string
		d    Continuous
		c    float64
	}{
		{"lognormal cap 2", ln, 2}, {"lognormal cap 50", ln, 50},
		{"pareto(0.8) cap 1e3", pa, 1e3}, {"gamma cap 0.5", g, 0.5}, {"pert cap 20", pe, 20},
	}
	seed := uint64(7100)
	for _, c := range cases {
		d, err := NewCapped(c.d, c.c)
		if err != nil {
			t.Fatal(c.name, err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		lo, hi := d.Support()
		boundsTest(t, c.name, xs, lo, hi)
		momentsTest(t, c.name, xs, d.Mean(), d.Var())
		// Fit below the cap and the atom's frequency at it.
		below := make([]float64, 0, len(xs))
		atoms := 0
		for _, x := range xs {
			if x >= c.c {
				atoms++
			} else {
				below = append(below, x)
			}
		}
		if d.Atom() > 0 && d.Atom() < 1 {
			n := float64(len(xs))
			if z := (float64(atoms) - n*d.Atom()) / math.Sqrt(n*d.Atom()*(1-d.Atom())); math.Abs(z) > zLimit {
				t.Errorf("%s: atom frequency %d vs %.0f, z=%.2f", c.name, atoms, n*d.Atom(), z)
			}
			ksTest(t, c.name+" below cap", below, func(x float64) float64 { return d.CDF(x) / (1 - d.Atom()) })
		}
	}
	// A capped Pareto with alpha < 1 has a finite mean.
	d, _ := NewCapped(pa, 1e3)
	if math.IsInf(d.Mean(), 0) || !(d.Mean() > 1) {
		t.Errorf("capped pareto mean %g", d.Mean())
	}
	if _, err := NewCapped(pe, 0); err == nil {
		t.Error("accepted a cap at the lower bound")
	}
}

// TestCappedExponentialClosedForm checks the quadrature against the
// closed form E[min(X, c)] = m(1 − e^(−c/m)) for an exponential.
func TestCappedExponentialClosedForm(t *testing.T) {
	e, _ := NewExponential(2)
	d, err := NewCapped(e, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := 2 * (1 - math.Exp(-1.5))
	if !near(d.Mean(), want, 1e-12) {
		t.Errorf("mean %.17g want %.17g", d.Mean(), want)
	}
	// E[min(X,c)²] = 2m² − 2m e^(−c/m)(c + m).
	want2 := 2*4 - 2*2*math.Exp(-1.5)*(3+2)
	if got := d.Var() + d.Mean()*d.Mean(); !near(got, want2, 1e-12) {
		t.Errorf("second moment %.17g want %.17g", got, want2)
	}
}

func BenchmarkWrappers(b *testing.B) {
	ln, _ := NewLognormal(0, 1)
	pe, _ := NewPERT(0, 2, 10, 4)
	tr, _ := NewTruncated(ln, 0.5, 2)
	tp, _ := NewTruncated(pe, 1, 3)
	ca, _ := NewCapped(ln, 2)
	for _, c := range []struct {
		name string
		d    Distribution
	}{{"truncated_lognormal", tr}, {"truncated_pert", tp}, {"capped_lognormal", ca}} {
		b.Run(c.name, func(b *testing.B) { benchFill(b, c.d) })
	}
}

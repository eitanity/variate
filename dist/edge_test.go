package dist

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestScaledCDFAtOwnAtoms uses one fixed seed and is deterministic: every
// value f·k of a scaled count must see the whole of the parent's mass at k.
func TestScaledCDFAtOwnAtoms(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	parent, _ := NewPoisson(5)
	for range 200000 {
		f := 0.01 + r.Float64()*100
		k := float64(1 + r.IntN(12))
		s, _ := NewScaled(parent, f)
		if got, want := s.CDF(f*k), parent.CDF(k); got != want {
			t.Fatalf("CDF(%v·%v) = %v, want %v", f, k, got, want)
		}
		if got, want := s.Survival(f*k), parent.Survival(k); got != want {
			t.Fatalf("Survival(%v·%v) = %v, want %v", f, k, got, want)
		}
		// Just below the atom the quotient can round up onto it.
		below := math.Nextafter(f*k, math.Inf(-1))
		if got, want := s.CDF(below), parent.CDF(k-1); got != want {
			t.Fatalf("CDF just below %v·%v = %v, want %v", f, k, got, want)
		}
	}
}

func TestScaledCDFAtOwnCap(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	ln, _ := NewLognormal(0, 1)
	parent, err := NewCapped(ln, 50)
	if err != nil {
		t.Fatal(err)
	}
	for range 200000 {
		f := 0.01 + r.Float64()*100
		s, _ := NewScaled(parent, f)
		if got := s.CDF(f * 50); got != 1 {
			t.Fatalf("CDF at the scaled cap with f=%v is %v", f, got)
		}
		if got := s.Survival(f * 50); got != 0 {
			t.Fatalf("Survival at the scaled cap with f=%v is %v", f, got)
		}
	}
}

func TestScaledContinuousCDFUnmoved(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	parent, _ := NewGamma(2, 3)
	for range 20000 {
		f := 0.01 + r.Float64()*100
		x := f * parent.Quantile(r.Float64())
		s, _ := NewScaled(parent, f)
		want := parent.CDF(x / f)
		if got := s.CDF(x); math.Abs(got-want) > 1e-12*want+1e-300 {
			t.Fatalf("CDF(%v) with f=%v is %v, want %v", x, f, got, want)
		}
	}
}

func TestScaledInfiniteArguments(t *testing.T) {
	parent, _ := NewExponential(1)
	s, _ := NewScaled(parent, 0.5)
	if s.CDF(math.Inf(1)) != 1 || s.Survival(math.Inf(1)) != 0 || s.CDF(math.Inf(-1)) != 0 || s.Survival(math.Inf(-1)) != 1 {
		t.Fatal("wrong value at an infinite argument")
	}
}

func TestDiscreteAtPlusInfinity(t *testing.T) {
	p5, _ := NewPoisson(5)
	p0, _ := NewPoisson(0)
	nb, _ := NewNegativeBinomial(5, 0.3)
	inf := math.Inf(1)
	for name, d := range map[string]Distribution{"poisson 5": p5, "poisson 0": p0, "negbin": nb} { // unordered: each entry is checked alone
		if got := d.CDF(inf); got != 1 {
			t.Errorf("%s: CDF(+Inf) = %v", name, got)
		}
		if got := d.Survival(inf); got != 0 {
			t.Errorf("%s: Survival(+Inf) = %v", name, got)
		}
	}
	if got, want := p5.CDF(3), 0.2650259152973615; math.Abs(got-want) > 1e-14 {
		t.Errorf("Poisson(5).CDF(3) = %v, want %v", got, want)
	}
}

func TestLogUniformSurvivalBelowSupport(t *testing.T) {
	d, _ := NewLogUniform(1, 10)
	for _, x := range []float64{math.Inf(-1), -5, 0, 0.5, 1} {
		if got := d.Survival(x); got != 1 {
			t.Errorf("Survival(%v) = %v", x, got)
		}
	}
	if got, want := d.Survival(math.Sqrt(10)), 0.5; math.Abs(got-want) > 1e-15 {
		t.Errorf("Survival at the geometric midpoint = %v", got)
	}
}

func TestGeometricCertainPMF(t *testing.T) {
	d, _ := NewGeometric(1)
	if got := d.PMF(0); got != 1 {
		t.Errorf("PMF(0) = %v", got)
	}
	if got := d.PMF(1); got != 0 {
		t.Errorf("PMF(1) = %v", got)
	}
	half, _ := NewGeometric(0.5)
	if got := half.PMF(2); got != 0.125 {
		t.Errorf("Geometric(0.5).PMF(2) = %v", got)
	}
}

package dist

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/integrate/quad"
)

// must unwraps a constructor whose parameters the test knows are valid.
func must[T any](v T, err error) func(*testing.T) T {
	return func(t *testing.T) T {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
}

func closeTo(got, want, tol float64) bool {
	if math.IsInf(want, 0) || math.IsNaN(want) {
		return got == want || math.IsNaN(got) && math.IsNaN(want)
	}
	return math.Abs(got-want) <= tol*(1+math.Abs(want))
}

type namedDist struct {
	name string
	d    Distribution
}

// continuousCases have no atoms inside the probability intervals the
// integral test uses, and no kink inside them.
func continuousCases(t *testing.T) []namedDist {
	t.Helper()
	ln := must(NewLognormal(0, 1))(t)
	ga := must(NewGamma(0.5, 2))(t)
	return []namedDist{
		{"uniform", must(NewUniform(1, 5))(t)},
		{"loguniform", must(NewLogUniform(1, 100))(t)},
		{"triangular", must(NewTriangular(0, 1, 2))(t)},
		{"exponential", must(NewExponential(2))(t)},
		{"lognormal", ln},
		{"weibull below 1", must(NewWeibull(0.7, 3))(t)},
		{"weibull above 1", must(NewWeibull(2, 3))(t)},
		{"gamma below 1", ga},
		{"gamma above 1", must(NewGamma(3, 2))(t)},
		{"beta below 1", must(NewBeta(0.5, 0.5))(t)},
		{"beta above 1", must(NewBeta(2, 3))(t)},
		{"pert", must(NewPERT(1, 2, 5, 4))(t)},
		{"logpert", must(NewLogPERT(1, 10, 1000, 4))(t)},
		{"normal", must(NewNormal(-1, 2))(t)},
		{"truncnormal", must(NewTruncatedNormal(1, 1, 0, 10))(t)},
		{"pareto", must(NewPareto(2, 3))(t)},
		{"gpd positive", must(NewGPD(1, 2, 0.2))(t)},
		{"gpd zero", must(NewGPD(1, 2, 0))(t)},
		{"gpd negative", must(NewGPD(1, 2, -0.3))(t)},
		{"scaled", must(NewScaled(ga, 7))(t)},
		{"truncated", must(NewTruncated(ln, 0.5, 50))(t)},
		{"truncated whole tail", must(NewTruncated(ln, 0.5, math.Inf(1)))(t)},
		{"capped", must(NewCapped(ln, 50))(t)},
		{"scaled capped", must(NewScaled(must(NewCapped(ln, 50))(t), 3))(t)},
		{"truncated density", must(NewTruncatedDensity(ln, 0.5, 50))(t)},
		{"capped density", must(NewCappedDensity(ln, 50))(t)},
		{"logpert density", must(NewLogPERTDensity(1, 10, 1000, 4))(t)},
	}
}

func TestLogDensityIntegratesToCDF(t *testing.T) {
	for _, c := range continuousCases(t) {
		for _, band := range [][2]float64{{0.1, 0.2}, {0.6, 0.7}, {0.85, 0.95}} {
			a, b := c.d.Quantile(band[0]), c.d.Quantile(band[1])
			got := quad.Fixed(func(x float64) float64 { return math.Exp(c.d.LogDensity(x)) }, a, b, 400, nil, 0)
			if want := c.d.CDF(b) - c.d.CDF(a); !closeTo(got, want, 1e-10) {
				t.Errorf("%s on [%v, %v]: integral %.15g, CDF difference %.15g", c.name, a, b, got, want)
			}
		}
	}
}

func TestContinuousHasNoAtoms(t *testing.T) {
	for _, c := range continuousCases(t) {
		x := c.d.Quantile(0.5)
		if c.d.IsAtom(x) {
			t.Errorf("%s: atom at its median %v", c.name, x)
		}
	}
}

func TestLogSurvivalMatchesSurvival(t *testing.T) {
	for _, c := range continuousCases(t) {
		for _, p := range []float64{0.01, 0.5, 0.9} {
			x := c.d.Quantile(p)
			if got, want := c.d.LogSurvival(x), math.Log(c.d.Survival(x)); !closeTo(got, want, 1e-12) {
				t.Errorf("%s at %v: %v, want %v", c.name, x, got, want)
			}
		}
	}
}

func TestLogDensityHandValues(t *testing.T) {
	e := math.E
	root2Pi := math.Sqrt(2 * math.Pi)
	phi1 := math.Exp(-0.5) / root2Pi
	cases := []struct {
		name string
		d    Distribution
		x    float64
		pdf  float64
	}{
		{"uniform", must(NewUniform(1, 5))(t), 2, 0.25},
		{"loguniform", must(NewLogUniform(1, e*e))(t), e, 1 / (2 * e)},
		{"triangular left", must(NewTriangular(0, 1, 4))(t), 0.5, 0.25},
		{"triangular right", must(NewTriangular(0, 1, 4))(t), 3, 1.0 / 6},
		{"triangular mode", must(NewTriangular(0, 1, 4))(t), 1, 0.5},
		{"exponential", must(NewExponential(2))(t), 3, 0.5 * math.Exp(-1.5)},
		{"lognormal", must(NewLognormal(0, 1))(t), e, math.Exp(-0.5) / (e * root2Pi)},
		{"weibull", must(NewWeibull(2, 3))(t), 3, 2.0 / 3 / e},
		{"gamma", must(NewGamma(2, 3))(t), 6, 6 * math.Exp(-2) / 9},
		{"beta", must(NewBeta(2, 3))(t), 0.5, 1.5},
		{"pert", must(NewPERT(0, 1, 2, 4))(t), 1, 0.9375},
		{"logpert", must(NewLogPERT(1, 10, 100, 4))(t), 10, 0.9375 / (10 * math.Ln10)},
		{"normal", must(NewNormal(1, 2))(t), 3, phi1 / 2},
		{"truncnormal", must(NewTruncatedNormal(0, 1, 0, math.Inf(1)))(t), 1, 2 * phi1},
		{"pareto", must(NewPareto(2, 3))(t), 4, 24.0 / 256},
		{"gpd", must(NewGPD(1, 2, 0.5))(t), 3, 0.5 * math.Pow(1.5, -3)},
		{"gpd exponential", must(NewGPD(1, 2, 0))(t), 3, 0.5 / e},
		{"bernoulli one", must(NewBernoulli(0.3))(t), 1, 0.3},
		{"bernoulli zero", must(NewBernoulli(0.3))(t), 0, 0.7},
		{"binomial", must(NewBinomial(4, 0.5))(t), 2, 0.375},
		{"poisson", must(NewPoisson(3))(t), 2, math.Exp(-3) * 4.5},
		{"negbin", must(NewNegativeBinomial(2, 0.5))(t), 1, 0.25},
		{"geometric", must(NewGeometric(0.25))(t), 2, 0.25 * 0.5625},
		{"constant", must(NewConstant(3))(t), 3, 1},
	}
	for _, c := range cases {
		if got, want := c.d.LogDensity(c.x), math.Log(c.pdf); !closeTo(got, want, 1e-13) {
			t.Errorf("%s at %v: %v, want %v", c.name, c.x, got, want)
		}
	}
}

func TestLogDensityAtSupportEnds(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name string
		d    Distribution
		x    float64
		want float64
	}{
		{"weibull shape below 1", must(NewWeibull(0.5, 2))(t), 0, inf},
		{"weibull shape 1", must(NewWeibull(1, 2))(t), 0, -math.Ln2},
		{"weibull shape above 1", must(NewWeibull(2, 2))(t), 0, -inf},
		{"gamma shape below 1", must(NewGamma(0.5, 2))(t), 0, inf},
		{"gamma shape 1", must(NewGamma(1, 2))(t), 0, -math.Ln2},
		{"gamma shape above 1", must(NewGamma(2, 2))(t), 0, -inf},
		{"beta lower below 1", must(NewBeta(0.5, 2))(t), 0, inf},
		{"beta lower 1", must(NewBeta(1, 3))(t), 0, math.Log(3)},
		{"beta lower above 1", must(NewBeta(2, 2))(t), 0, -inf},
		{"beta upper below 1", must(NewBeta(2, 0.5))(t), 1, inf},
		{"beta upper 1", must(NewBeta(3, 1))(t), 1, math.Log(3)},
		{"beta upper above 1", must(NewBeta(2, 2))(t), 1, -inf},
		{"gpd end vanishing", must(NewGPD(0, 1, -0.5))(t), 2, -inf},
		{"gpd end uniform", must(NewGPD(0, 2, -1))(t), 2, -math.Ln2},
		{"gpd end diverging", must(NewGPD(0, 1, -2))(t), 0.5, inf},
		{"triangular mode at low", must(NewTriangular(0, 0, 2))(t), 0, 0},
		{"triangular mode at high", must(NewTriangular(0, 2, 2))(t), 2, 0},
		{"triangular vanishing end", must(NewTriangular(0, 1, 2))(t), 2, -inf},
		{"uniform lower", must(NewUniform(0, 4))(t), 0, -math.Log(4)},
		{"uniform upper", must(NewUniform(0, 4))(t), 4, -math.Log(4)},
		{"lognormal at zero", must(NewLognormal(0, 1))(t), 0, -inf},
		{"pareto at minimum", must(NewPareto(2, 3))(t), 2, math.Log(1.5)},
		{"pert uniform end", must(NewPERT(0, 1, 2, 0))(t), 2, -math.Ln2},
		{"logpert lower", must(NewLogPERT(1, 10, 100, 0))(t), 1, -math.Log(2 * math.Ln10)},
	}
	for _, c := range cases {
		if got := c.d.LogDensity(c.x); !closeTo(got, c.want, 1e-13) {
			t.Errorf("%s at %v: %v, want %v", c.name, c.x, got, c.want)
		}
	}
}

func TestLogDensityOffSupport(t *testing.T) {
	for _, c := range continuousCases(t) {
		lo, hi := c.d.Support()
		for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), lo - 1, hi + 1} {
			if got := c.d.LogDensity(x); !math.IsInf(got, -1) {
				t.Errorf("%s at %v: %v", c.name, x, got)
			}
		}
	}
}

func TestDiscreteLogDensity(t *testing.T) {
	count := []float64{0, 1, 2, 3, 5, 8, 13}
	cases := []struct {
		name   string
		d      Distribution
		points []float64
	}{
		{"bernoulli", must(NewBernoulli(0.3))(t), []float64{0, 1}},
		{"binomial", must(NewBinomial(20, 0.3))(t), count},
		{"binomial certain", must(NewBinomial(5, 1))(t), []float64{5}},
		{"binomial impossible", must(NewBinomial(5, 0))(t), []float64{0}},
		{"geometric", must(NewGeometric(0.3))(t), count},
		{"geometric certain", must(NewGeometric(1))(t), []float64{0}},
		{"poisson", must(NewPoisson(5))(t), count},
		{"poisson large", must(NewPoisson(200))(t), []float64{150, 200, 250}},
		{"poisson zero rate", must(NewPoisson(0))(t), []float64{0}},
		{"negbin", must(NewNegativeBinomial(5, 0.3))(t), count},
		{"categorical", must(NewCategorical([]float64{1, 2, 7.25, 2}, []float64{0.1, 0.2, 0.3, 0.4}))(t), []float64{1, 2, 7.25}},
		{"empirical", must(NewEmpirical([]float64{4, 4, 9}))(t), []float64{4, 9}},
		{"constant", must(NewConstant(3))(t), []float64{3}},
	}
	for _, c := range cases {
		for _, x := range c.points {
			jump := c.d.CDF(x) - c.d.CDF(math.Nextafter(x, math.Inf(-1)))
			ld := c.d.LogDensity(x)
			if !closeTo(math.Exp(ld), jump, 1e-9) {
				t.Errorf("%s at %v: mass %v, CDF jump %v", c.name, x, math.Exp(ld), jump)
			}
			if !c.d.IsAtom(x) {
				t.Errorf("%s: %v is not an atom", c.name, x)
			}
			if pm, ok := c.d.(PMFer); ok && !closeTo(pm.PMF(x), math.Exp(ld), 1e-14) {
				t.Errorf("%s at %v: PMF %v, exp(LogDensity) %v", c.name, x, pm.PMF(x), math.Exp(ld))
			}
			off := x + 0.125
			if got := c.d.LogDensity(off); !math.IsInf(got, -1) || c.d.IsAtom(off) {
				t.Errorf("%s at %v off the lattice: %v", c.name, off, got)
			}
		}
		for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1} {
			if got := c.d.LogDensity(x); !math.IsInf(got, -1) || c.d.IsAtom(x) {
				t.Errorf("%s at %v: %v", c.name, x, got)
			}
		}
	}
}

func TestCategoricalSmallMassKeepsDigits(t *testing.T) {
	d := must(NewCategorical([]float64{1, 2, 3}, []float64{0.75, 1e-13, 0.25 - 1e-13}))(t)
	if got, want := d.LogDensity(2), math.Log(1e-13); !closeTo(got, want, 1e-15) {
		t.Errorf("%v, want %v", got, want)
	}
}

func TestCappedAtom(t *testing.T) {
	ln := must(NewLognormal(0, 1))(t)
	d := must(NewCapped(ln, 3))(t)
	if got, want := d.LogDensity(3), math.Log(ln.Survival(3)); got != want {
		t.Errorf("at the cap: %v, want %v", got, want)
	}
	if got, want := d.LogDensity(2), ln.LogDensity(2); got != want {
		t.Errorf("below the cap: %v, want %v", got, want)
	}
	if got := d.LogDensity(3.5); !math.IsInf(got, -1) {
		t.Errorf("above the cap: %v", got)
	}
	if !d.IsAtom(3) || d.IsAtom(2) || d.IsAtom(3.5) {
		t.Error("atom only at the cap")
	}
	if got := d.LogSurvival(3); !math.IsInf(got, -1) {
		t.Errorf("survival at the cap: %v", got)
	}
	// A cap at or past the top of the support collects nothing.
	u := must(NewUniform(0, 1))(t)
	k := must(NewCapped(u, 1))(t)
	if k.IsAtom(1) || k.LogDensity(1) != 0 {
		t.Errorf("empty cap: atom %v, density %v", k.IsAtom(1), k.LogDensity(1))
	}
}

func TestCappedDiscreteParentMassAtCap(t *testing.T) {
	p := must(NewPoisson(5))(t)
	d := must(NewCappedDensity(p, 3))(t)
	if got, want := d.LogDensity(3), math.Log(p.Survival(2)); !closeTo(got, want, 1e-13) {
		t.Errorf("%v, want P(X ≥ 3) = %v", got, want)
	}
	if !d.IsAtom(2) || !d.IsAtom(3) || d.IsAtom(4) {
		t.Error("atoms at the counts up to the cap")
	}
}

func TestTruncatedLogDensity(t *testing.T) {
	ln := must(NewLognormal(0, 1))(t)
	d := must(NewTruncated(ln, 0.5, 50))(t)
	mass := ln.CDF(50) - ln.CDF(0.5)
	for _, x := range []float64{0.5, 1, 50} {
		if got, want := d.LogDensity(x), ln.LogDensity(x)-math.Log(mass); !closeTo(got, want, 1e-13) {
			t.Errorf("at %v: %v, want %v", x, got, want)
		}
	}
	if !math.IsInf(d.LogDensity(0.4), -1) || !math.IsInf(d.LogDensity(51), -1) {
		t.Error("off the range is -Inf")
	}
}

func TestTruncatedDiscreteEdges(t *testing.T) {
	p := must(NewPoisson(5))(t)
	above, mass, err := NewSplit(p, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(above.LogDensity(2), -1) || above.IsAtom(2) {
		t.Errorf("the split point belongs below: %v", above.LogDensity(2))
	}
	if got, want := above.LogDensity(3), p.LogDensity(3)-math.Log(mass); !closeTo(got, want, 1e-13) || !above.IsAtom(3) {
		t.Errorf("above the split: %v, want %v", got, want)
	}
	below, lowMass, err := NewSplit(p, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := below.LogDensity(0), p.LogDensity(0)-math.Log(lowMass); !closeTo(got, want, 1e-13) || !below.IsAtom(0) {
		t.Errorf("the lower atom stays: %v, want %v", got, want)
	}
}

// TestScaledLogDensity checks every value of a scaled atom over random
// factors with one fixed seed; it is deterministic.
func TestScaledLogDensity(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	poisson := must(NewPoisson(5))(t)
	ln := must(NewLognormal(0, 1))(t)
	capped := must(NewCapped(ln, 50))(t)
	for range 200000 {
		f := 0.01 + r.Float64()*100
		k := float64(1 + r.IntN(12))
		s := must(NewScaled(poisson, f))(t)
		if !s.IsAtom(f*k) || s.LogDensity(f*k) != poisson.LogDensity(k) {
			t.Fatalf("count %v·%v: atom %v, value %v", f, k, s.IsAtom(f*k), s.LogDensity(f*k))
		}
		c := must(NewScaled(capped, f))(t)
		if !c.IsAtom(f*50) || c.LogDensity(f*50) != capped.LogDensity(50) {
			t.Fatalf("cap at %v·50: atom %v, value %v", f, c.IsAtom(f*50), c.LogDensity(f*50))
		}
	}
	c := must(NewScaled(capped, 2))(t)
	if got, want := c.LogDensity(60), ln.LogDensity(30)-math.Ln2; !closeTo(got, want, 1e-14) {
		t.Errorf("below the scaled cap: %v, want %v", got, want)
	}
	if c.IsAtom(60) {
		t.Error("60 is not an atom")
	}
	if got := c.LogSurvival(60); !closeTo(got, ln.LogSurvival(30), 1e-14) {
		t.Errorf("scaled survival: %v", got)
	}
}

func TestLogSurvivalPastUnderflow(t *testing.T) {
	cases := []struct {
		name string
		d    Distribution
		x    float64
		want float64
	}{
		{"exponential", must(NewExponential(1))(t), 1000, -1000},
		{"weibull", must(NewWeibull(2, 1))(t), 30, -900},
		{"pareto", must(NewPareto(1, 2000))(t), math.E, -2000},
		{"gpd exponential", must(NewGPD(0, 1, 0))(t), 1000, -1000},
		{"gpd", must(NewGPD(0, 1, 0.001))(t), 1e9, -math.Log1p(1e6) / 0.001},
		{"geometric", must(NewGeometric(0.5))(t), 2000, 2001 * math.Log(0.5)},
		{"scaled weibull", must(NewScaled(must(NewWeibull(2, 1))(t), 2))(t), 60, -900},
		{"truncated whole tail", must(NewTruncated(must(NewWeibull(2, 1))(t), 1, math.Inf(1)))(t), 30, -899},
		{"capped", must(NewCappedDensity(must(NewWeibull(2, 1))(t), 100))(t), 30, -900},
	}
	for _, c := range cases {
		if got := c.d.LogSurvival(c.x); !closeTo(got, c.want, 1e-13) {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

func TestLogSurvivalEnds(t *testing.T) {
	for _, c := range continuousCases(t) {
		lo, hi := c.d.Support()
		if got := c.d.LogSurvival(lo - 1); got != 0 {
			t.Errorf("%s below the support: %v", c.name, got)
		}
		if got := c.d.LogSurvival(hi); !math.IsInf(got, -1) {
			t.Errorf("%s at the top: %v", c.name, got)
		}
	}
	g := must(NewGeometric(1))(t)
	if got := g.LogSurvival(0); !math.IsInf(got, -1) {
		t.Errorf("certain geometric: %v", got)
	}
}

func TestDensityConstructorsMatchFull(t *testing.T) {
	ln := must(NewLognormal(0, 1))(t)
	pairs := []struct {
		name       string
		full, fast Distribution
	}{
		{"truncated", must(NewTruncated(ln, 0.5, 50))(t), must(NewTruncatedDensity(ln, 0.5, 50))(t)},
		{"capped", must(NewCapped(ln, 50))(t), must(NewCappedDensity(ln, 50))(t)},
		{"logpert", must(NewLogPERT(1, 10, 1000, 4))(t), must(NewLogPERTDensity(1, 10, 1000, 4))(t)},
	}
	for _, c := range pairs {
		if !math.IsNaN(c.fast.Mean()) || !math.IsNaN(c.fast.Var()) {
			t.Errorf("%s: moments %v, %v, want NaN", c.name, c.fast.Mean(), c.fast.Var())
		}
		if math.IsNaN(c.full.Mean()) || math.IsNaN(c.full.Var()) {
			t.Errorf("%s: the full constructor lost its moments", c.name)
		}
		for _, p := range []float64{0.01, 0.5, 0.99} {
			x := c.full.Quantile(p)
			if c.full.LogDensity(x) != c.fast.LogDensity(x) || c.full.CDF(x) != c.fast.CDF(x) ||
				c.full.Survival(x) != c.fast.Survival(x) || c.full.Quantile(p) != c.fast.Quantile(p) ||
				c.full.QuantileUpper(p) != c.fast.QuantileUpper(p) {
				t.Errorf("%s differs at p=%v", c.name, p)
			}
		}
		a, b := make([]float64, 256), make([]float64, 256)
		c.full.Fill(newRand(1), a)
		c.fast.Fill(newRand(1), b)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s: draw %d is %v and %v", c.name, i, a[i], b[i])
			}
		}
	}
}

func TestDensityConstructorsValidate(t *testing.T) {
	ln := must(NewLognormal(0, 1))(t)
	if _, err := NewTruncatedDensity(ln, 5, 1); err == nil {
		t.Error("lo above hi accepted")
	}
	if _, err := NewCappedDensity(ln, -1); err == nil {
		t.Error("cap below the support accepted")
	}
	if _, err := NewLogPERTDensity(-1, 10, 100, 4); err == nil {
		t.Error("non-positive low accepted")
	}
}

func BenchmarkConstructTruncated(b *testing.B) {
	ln, _ := NewLognormal(0, 1)
	for i := 0; b.Loop(); i++ {
		if _, err := NewTruncated(ln, 0.5, 50+float64(i%100)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConstructTruncatedDensity(b *testing.B) {
	ln, _ := NewLognormal(0, 1)
	for i := 0; b.Loop(); i++ {
		if _, err := NewTruncatedDensity(ln, 0.5, 50+float64(i%100)); err != nil {
			b.Fatal(err)
		}
	}
}

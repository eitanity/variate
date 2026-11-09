package dist

import (
	"math"
	"testing"
)

func TestLogContributionForms(t *testing.T) {
	inf := math.Inf(1)
	ln := must(NewLognormal(0, 1))(t)
	normal := must(NewNormal(0, 1))(t)
	poisson := must(NewPoisson(2))(t)
	if got := LogContribution(ln, 2, 2); got != ln.LogDensity(2) {
		t.Errorf("exact: %v", got)
	}
	if got := LogContribution(poisson, 2, 2); got != poisson.LogDensity(2) {
		t.Errorf("exact count: %v", got)
	}
	if got := LogContribution(ln, 2, inf); got != ln.LogSurvival(2) {
		t.Errorf("right: %v", got)
	}
	if got := LogContribution(ln, -inf, 2); got != math.Log(ln.CDF(2)) {
		t.Errorf("left: %v", got)
	}
	if got := LogContribution(normal, -inf, 1); got != math.Log(normal.CDF(1)) {
		t.Errorf("left on the whole line: %v", got)
	}
	if got := LogContribution(normal, -inf, inf); got != 0 {
		t.Errorf("whole line: %v", got)
	}
	if got, want := LogContribution(ln, 1, 2), math.Log(ln.CDF(2)-ln.CDF(1)); !closeTo(got, want, 1e-15) {
		t.Errorf("interval: %v, want %v", got, want)
	}
}

func TestLogContributionLeftKeepsAtomAtZero(t *testing.T) {
	d := must(NewPoisson(2))(t)
	if got, want := LogContribution(d, math.Inf(-1), 2), math.Log(d.CDF(2)); got != want {
		t.Errorf("%v, want %v", got, want)
	}
	if got, want := LogContribution(d, -1, 2), math.Log(d.CDF(2)); got != want {
		t.Errorf("below the support: %v, want %v", got, want)
	}
	if got, want := LogContribution(d, 0, 2), math.Log(d.CDF(2)-d.CDF(0)); !closeTo(got, want, 1e-15) {
		t.Errorf("(0, 2] excludes the atom at 0: %v, want %v", got, want)
	}
}

func TestLogContributionAtomOnEitherEnd(t *testing.T) {
	d := must(NewPoisson(5))(t)
	if got, want := LogContribution(d, 2, 3), d.LogDensity(3); !closeTo(got, want, 1e-12) {
		t.Errorf("(2, 3]: %v, want %v", got, want)
	}
	ln := must(NewLognormal(0, 1))(t)
	k := must(NewCapped(ln, 50))(t)
	if got, want := LogContribution(k, 49, 50), math.Log(ln.Survival(49)); !closeTo(got, want, 1e-12) {
		t.Errorf("an interval ending at the cap holds the atom: %v, want %v", got, want)
	}
	if got := LogContribution(k, 50, math.Inf(1)); !math.IsInf(got, -1) {
		t.Errorf("nothing exceeds the cap: %v", got)
	}
	s := must(NewScaled(k, 3))(t)
	if got, want := LogContribution(s, 3*49, 3*50), math.Log(ln.Survival(49)); !closeTo(got, want, 1e-12) {
		t.Errorf("scaled: %v, want %v", got, want)
	}
	if got, want := LogContribution(s, 150, 150), k.LogDensity(50); got != want {
		t.Errorf("scaled exact at the cap: %v, want %v", got, want)
	}
}

func TestLogContributionUpperTailDoesNotCancel(t *testing.T) {
	d := must(NewExponential(1))(t)
	a, b := 40.0, 40.0000001
	want := -a + math.Log(-math.Expm1(a-b))
	if got := LogContribution(d, a, b); !closeTo(got, want, 1e-9) {
		t.Errorf("%v, want %v", got, want)
	}
}

func TestLogContributionPastUnderflow(t *testing.T) {
	d := must(NewWeibull(2, 1))(t)
	got := LogContribution(d, 30, 31)
	want := -900 + math.Log(-math.Expm1(900-961))
	if !closeTo(got, want, 1e-13) {
		t.Errorf("%v, want %v", got, want)
	}
	if got := LogContribution(must(NewLognormal(0, 0.01))(t), 1e300, 1e301); !math.IsInf(got, -1) {
		t.Errorf("no closed form and nothing left: %v", got)
	}
	e := must(NewExponential(1))(t)
	got = LogContribution(e, 1000, 1000.1)
	want = -1000 + math.Log(-math.Expm1(-0.1))
	if !closeTo(got, want, 1e-12) {
		t.Errorf("near end of the log form: %v, want %v", got, want)
	}
}

func TestLogContributionMalformed(t *testing.T) {
	d := must(NewExponential(1))(t)
	nan := math.NaN()
	for _, r := range [][2]float64{{nan, 1}, {1, nan}, {2, 1}} {
		if got := LogContribution(d, r[0], r[1]); !math.IsNaN(got) {
			t.Errorf("(%v, %v): %v", r[0], r[1], got)
		}
	}
}

func TestLogContributionOutsideTheSupport(t *testing.T) {
	d := must(NewUniform(0, 1))(t)
	if got := LogContribution(d, 2, 3); !math.IsInf(got, -1) {
		t.Errorf("above: %v", got)
	}
	if got := LogContribution(d, -3, -2); !math.IsInf(got, -1) {
		t.Errorf("below: %v", got)
	}
	if got := LogContribution(d, -3, 3); got != 0 {
		t.Errorf("covering: %v", got)
	}
}

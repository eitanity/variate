package variate

import (
	"math"
	"testing"

	"code.eitanity.com/variate/dist"
)

// TestPassesSaturates: the count of tensor points stops at the ceiling
// instead of wrapping, however many quantities there are, and is still
// 0 when a quantity has no nodes.
func TestPassesSaturates(t *testing.T) {
	for _, k := range []int{1, 15, 16, 31, 32, 64} {
		qs := make([]thetaQuantity, k)
		for i := range qs {
			qs[i].nodes = make([]float64, 4)
		}
		want := passesCeiling
		if k < 15 {
			want = 1 << (2 * k)
		}
		if got := (&thetaMix{q: qs}).passes(); got != want {
			t.Errorf("%d quantities of 4 nodes: %d passes, want %d", k, got, want)
		}
		qs[k-1].nodes = nil
		if got := (&thetaMix{q: qs}).passes(); got != 0 {
			t.Errorf("%d quantities, one without nodes: %d passes", k, got)
		}
	}
}

// TestZeroManyPeriodValues: forty period-scoped values on one node, a
// tensor of 2^40 points if each kept both atoms. Values that are never 0
// are one point each and P(T = 0) is exact; values that are 0 half the
// time leave a mixture above the cap even at the means, and P(T = 0) is
// NaN instead of the run hanging.
func TestZeroManyPeriodValues(t *testing.T) {
	half, err := dist.NewBernoulli(0.5)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		value dist.Distribution
		want  float64
	}{
		{"never 0", pert(1, 2, 4), math.Exp(-2)},
		{"0 half the time", half, math.NaN()},
	} {
		fns := make([]Function, 40)
		for i := range fns {
			fns[i] = Function{Name: "f" + itoa(i), Category: "k", Model: tc.value, Scope: PerPeriod}
		}
		g := Graph{
			Sources: []Source{{ID: "s", Frequency: c(2), To: "a"}},
			Nodes:   []Node{{ID: "a", Functions: fns}},
		}
		m, _ := mustCompile(t, g, Config{})
		e, err := m.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		if got := e.PZero(); !(near(got, tc.want, 1e-12) || math.IsNaN(got) && math.IsNaN(tc.want)) {
			t.Errorf("%s: P(T = 0) %.17g, want %.17g", tc.name, got, tc.want)
		}
	}
}

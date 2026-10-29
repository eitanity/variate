// The long transform tests run only when VARIATE_STAT=1 asks for them.
package suite

import (
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// TestTransformRandomDAGsLong: more seeds, up to three period-scoped
// probabilities per model and the full grid.
func TestTransformRandomDAGsLong(t *testing.T) {
	skipUnlessLong(t)
	seeds := make([]uint64, 0, 16)
	for s := uint64(1); s <= 16; s++ {
		seeds = append(seeds, s)
	}
	checkRandomDAGs(t, seeds, 3, 1<<20)
}

// TestTransformLognormalFrequencyLong: a lognormal frequency with σ = 4
// is refused, by the node cap or the work cap.
func TestTransformLognormalFrequencyLong(t *testing.T) {
	skipUnlessLong(t)
	ln, _ := dist.NewLognormal(-8, 4)
	g := compoundExponential(1, 1)
	g.Sources[0].Frequency = ln
	_, tr := transformOf(t, g, variate.TransformConfig{})
	if tr.Applicable || !strings.Contains(tr.Reason, "frequency quadrature") {
		t.Errorf("applicable=%v reason %q", tr.Applicable, tr.Reason)
	}
}

// TestTransformWidePERTFrequencyLong: a PERT frequency with span 2000
// converges within the doubling cap.
func TestTransformWidePERTFrequencyLong(t *testing.T) {
	skipUnlessLong(t)
	p, err := dist.NewPERT(0, 500, 2000, dist.DefaultPERTShape)
	if err != nil {
		t.Fatal(err)
	}
	g := compoundExponential(1, 1)
	g.Sources[0].Frequency = p
	_, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
	if !tr.Applicable {
		t.Fatalf("%s", tr.Reason)
	}
	if tr.QuadNodes < 1024 {
		t.Errorf("quad nodes %d", tr.QuadNodes)
	}
}

// TestInvariantNoSelfCheckRefusalLong: the self-check invariant over
// eight random and eight heavy-tailed graphs.
func TestInvariantNoSelfCheckRefusalLong(t *testing.T) {
	skipUnlessLong(t)
	checkNoSelfCheckRefusal(t, invariantCorpus(t, 8))
}

// skipUnlessLong skips a test of the long suite in a fast run.
func skipUnlessLong(t *testing.T) {
	t.Helper()
	if !longSuite() {
		t.Skip("long suite only: VARIATE_STAT=1 selects it")
	}
}

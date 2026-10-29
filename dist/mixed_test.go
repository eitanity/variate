package dist

import (
	"fmt"
	"testing"
)

// TestMixedPoisson: counts drawn as Poisson(λ) with λ from a frequency
// PERT have E[k] = E[λ] and Var[k] = E[λ] + Var[λ]. The rows are the
// frequency PERTs the Python undercounted.
func TestMixedPoisson(t *testing.T) {
	rows := [][3]float64{{3, 5, 8}, {0.1, 1, 10}, {0.01, 0.1, 10}, {0.001, 0.05, 20}, {0, 0.1, 10}}
	seed := uint64(8000)
	for _, row := range rows {
		name := fmt.Sprintf("mixed pert%v", row)
		f, err := NewPERT(row[0], row[1], row[2], DefaultPERTShape)
		if err != nil {
			t.Fatal(err)
		}
		ss := NewSamplerSet([]Distribution{f}, 1024)
		ss.Reset(seed, 0)
		seed++
		ks := make([]float64, nDraws)
		for i := range ks {
			ks[i] = float64(PoissonCount(ss.Control(), ss.Sampler(0).Next()))
		}
		momentsTest(t, name, ks, f.Mean(), f.Mean()+f.Var())
	}
}

package dist

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// TestMixedPoisson: counts drawn as Poisson(λ) with λ from a frequency
// PERT have E[k] = E[λ] and Var[k] = E[λ] + Var[λ]. The rows are the
// frequency PERTs the Python undercounted; the counts by inversion at
// a uniform likewise.
func TestMixedPoisson(t *testing.T) {
	rows := [][3]float64{{3, 5, 8}, {0.1, 1, 10}, {0.01, 0.1, 10}, {0.001, 0.05, 20}, {0, 0.1, 10}}
	seed := uint64(8000)
	for _, row := range rows {
		name := fmt.Sprintf("mixed pert%v", row)
		f, err := NewPERT(row[0], row[1], row[2], DefaultPERTShape)
		if err != nil {
			t.Fatal(err)
		}
		r := rand.New(rand.NewPCG(seed, 0))
		seed++
		lam := make([]float64, nDraws)
		f.Fill(r, lam)
		ks, at := make([]float64, nDraws), make([]float64, nDraws)
		for i := range ks {
			ks[i] = float64(PoissonCount(r, lam[i]))
			at[i] = float64(PoissonCountAt(lam[i], (float64(r.Uint64()>>11)+0.5)/(1<<53)))
		}
		momentsTest(t, name, ks, f.Mean(), f.Mean()+f.Var())
		momentsTest(t, name+" at u", at, f.Mean(), f.Mean()+f.Var())
	}
}

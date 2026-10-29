package dist

import (
	"math"
	"math/rand/v2"
	"sort"
)

const (
	familyCategorical = "categorical"
	familyEmpirical   = "empirical"
	// probSumTol is the tolerance on Σp = 1.
	probSumTol = 1e-12
)

// Categorical draws from a finite set of values with given probabilities,
// by Vose's alias method in O(1) per draw. Values may repeat.
type Categorical struct {
	values []float64
	probs  []float64
	prob   []float64 // alias table: probability of the own column
	alias  []int32
	mean   float64
	vr     float64
	lo, hi float64
	// sorted copies for CDF and PMF
	sortedV []float64
	cumP    []float64 // cumulative probability by sorted value, merged on ties
}

func (d Categorical) Fill(r *rand.Rand, dst []float64) {
	n := len(d.prob)
	for i := range dst {
		u := r.Float64() * float64(n)
		j := int(u)
		if u-float64(j) < d.prob[j] {
			dst[i] = d.values[j]
		} else {
			dst[i] = d.values[d.alias[j]]
		}
	}
}

func (d Categorical) Mean() float64 { return d.mean }

func (d Categorical) Var() float64 { return d.vr }

func (d Categorical) Support() (lo, hi float64) { return d.lo, d.hi }

func (d Categorical) CDF(x float64) float64 {
	i := sort.SearchFloat64s(d.sortedV, x)
	if i < len(d.sortedV) && d.sortedV[i] == x {
		return math.Min(d.cumP[i], 1)
	}
	if i == 0 {
		return 0
	}
	return math.Min(d.cumP[i-1], 1)
}

func (d Categorical) PMF(x float64) float64 {
	i := sort.SearchFloat64s(d.sortedV, x)
	if i == len(d.sortedV) || d.sortedV[i] != x {
		return 0
	}
	if i == 0 {
		return d.cumP[0]
	}
	return d.cumP[i] - d.cumP[i-1]
}

// Len returns the number of entries.
func (d Categorical) Len() int { return len(d.values) }

// Survival is P(X > x).
func (d Categorical) Survival(x float64) float64 { return 1 - d.CDF(x) }

// Quantile is the smallest value with at least a fraction p of the mass
// at or below it.
func (d Categorical) Quantile(p float64) float64 {
	for i, cp := range d.cumP {
		if cp >= p {
			return d.sortedV[i]
		}
	}
	return d.hi
}

// QuantileUpper is the smallest value with at most q of the mass above it.
func (d Categorical) QuantileUpper(q float64) float64 {
	for i, cp := range d.cumP {
		if 1-cp <= q {
			return d.sortedV[i]
		}
	}
	return d.hi
}

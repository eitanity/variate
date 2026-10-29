package dist

import (
	"math"
	"sort"
)

// NewCategorical validates the lengths match, values are finite,
// probabilities are non-negative and sum to 1 within 1e-12.
func NewCategorical(values, probs []float64) (Categorical, error) {
	if len(values) == 0 || len(values) != len(probs) {
		return Categorical{}, relationError(familyCategorical, "values and p must have the same non-zero length")
	}
	if len(values) >= math.MaxInt32 {
		return Categorical{}, relationError(familyCategorical, "too many values")
	}
	sum := 0.0
	for i, p := range probs {
		if !finite(values[i]) {
			return Categorical{}, paramError(familyCategorical, paramValue, values[i])
		}
		if !(p >= 0) || math.IsInf(p, 0) {
			return Categorical{}, paramError(familyCategorical, paramP, p)
		}
		sum += p
	}
	if math.Abs(sum-1) > probSumTol {
		return Categorical{}, relationError(familyCategorical, "p must sum to 1")
	}
	d := Categorical{values: append([]float64(nil), values...), probs: append([]float64(nil), probs...)}
	d.buildAlias()
	d.buildOrder()
	return d, nil
}

// NewEmpirical is a Categorical with equal weights: resampling of observed data.
func NewEmpirical(values []float64) (Categorical, error) {
	if len(values) == 0 {
		return Categorical{}, relationError(familyEmpirical, "no values")
	}
	probs := make([]float64, len(values))
	for i := range probs {
		probs[i] = 1 / float64(len(values))
	}
	d, err := NewCategorical(values, probs)
	if err != nil {
		return Categorical{}, relationError(familyEmpirical, err.Error())
	}
	return d, nil
}

// buildAlias is Vose's construction with the small/large work lists.
func (d *Categorical) buildAlias() {
	n := len(d.probs)
	d.prob = make([]float64, n)
	d.alias = make([]int32, n)
	scaled := make([]float64, n)
	small := make([]int32, 0, n)
	large := make([]int32, 0, n)
	for i, p := range d.probs {
		scaled[i] = p * float64(n)
		if scaled[i] < 1 {
			small = append(small, int32(i))
		} else {
			large = append(large, int32(i))
		}
	}
	// bound: each iteration retires one entry; at most n iterations
	for len(small) > 0 && len(large) > 0 {
		s := small[len(small)-1]
		l := large[len(large)-1]
		small = small[:len(small)-1]
		large = large[:len(large)-1]
		d.prob[s] = scaled[s]
		d.alias[s] = l
		scaled[l] = scaled[l] + scaled[s] - 1
		if scaled[l] < 1 {
			small = append(small, l)
		} else {
			large = append(large, l)
		}
	}
	for _, i := range large {
		d.prob[i] = 1
	}
	for _, i := range small {
		d.prob[i] = 1
	}
}

// buildOrder prepares the sorted value list and cumulative masses, and the
// moments.
func (d *Categorical) buildOrder() {
	n := len(d.values)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return d.values[idx[a]] < d.values[idx[b]] })
	d.sortedV = make([]float64, 0, n)
	d.cumP = make([]float64, 0, n)
	cum := 0.0
	for _, i := range idx {
		v, p := d.values[i], d.probs[i]
		cum += p
		if k := len(d.sortedV); k > 0 && d.sortedV[k-1] == v {
			d.cumP[k-1] = cum
			continue
		}
		d.sortedV = append(d.sortedV, v)
		d.cumP = append(d.cumP, cum)
	}
	d.lo, d.hi = d.sortedV[0], d.sortedV[len(d.sortedV)-1]
	var m, m2 float64
	for i, v := range d.values {
		m += d.probs[i] * v
	}
	for i, v := range d.values {
		m2 += d.probs[i] * (v - m) * (v - m)
	}
	d.mean, d.vr = m, m2
}

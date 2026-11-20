package variate

import (
	"math"
	"slices"

	"code.eitanity.com/variate/dist"
)

// The variance with holding times of a model that draws values or
// probabilities once a period conditions on those draws: given them,
// the period's rounds are independent but for their rates and the
// holdings drawn once a period, and the mixture over the draws' points
// adds Σ_d Var(M(d)) of the mean booking by lag. A round's mean booking
// by lag is multilinear in the values and probabilities, each appearing
// at most once in a product, so their second moments are all the
// variance needs, and a rule of two points, the mean ± one standard
// deviation, is exact for them.

// periodDraws lists the pools of the values and probabilities drawn
// once a period.
func (m *Compiled) periodDraws() []int32 {
	var out []int32
	add := func(pool int32) {
		if pool != noIndex && m.poolPeriod[pool] != noIndex && !slices.Contains(out, pool) {
			out = append(out, pool)
		}
	}
	for _, f := range m.fns {
		add(f.pool)
	}
	for _, o := range m.outs {
		add(o.probPool)
	}
	return out
}

// periodPointsMax caps the mixture's points, each a model's
// between-period term computed again.
const periodPointsMax = 1 << 6

// periodRules is each draw's two points, one where it does not vary;
// false when a draw's variance is not finite or the points pass
// periodPointsMax.
func (m *Compiled) periodRules(draws []int32) (xs, ws [][]float64, points int, ok bool) {
	xs, ws = make([][]float64, len(draws)), make([][]float64, len(draws))
	points = 1
	for i, pool := range draws {
		d := m.pools[pool]
		mean, sd := d.Mean(), math.Sqrt(d.Var())
		switch {
		case !isFinite(sd):
			return nil, nil, 0, false
		case sd == 0:
			xs[i], ws[i] = []float64{mean}, []float64{1}
		default:
			xs[i], ws[i] = []float64{mean - sd, mean + sd}, []float64{0.5, 0.5}
		}
		if points *= len(xs[i]); points > periodPointsMax {
			return nil, nil, 0, false
		}
	}
	return xs, ws, points, true
}

// fixedAt is the model with each draw's pool fixed at its point.
func (m *Compiled) fixedAt(draws []int32, at []float64) (*Compiled, error) {
	c := *m
	c.pools = slices.Clone(m.pools)
	c.poolMean = slices.Clone(m.poolMean)
	c.poolPeriod = slices.Clone(m.poolPeriod)
	for i, pool := range draws {
		k, err := dist.NewConstant(at[i])
		if err != nil {
			return nil, err
		}
		c.pools[pool], c.poolMean[pool], c.poolPeriod[pool] = k, at[i], noIndex
	}
	c.expect = nil
	return &c, nil
}

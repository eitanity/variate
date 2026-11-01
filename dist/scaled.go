package dist

import "math/rand/v2"

const (
	familyScaled = "scale"
	paramFactor  = "factor"
)

// Scaled is f·X for a factor f > 0: every draw of X multiplied by f, from
// the same stream, so a scaled model draws what the unscaled one does,
// value for value, times f.
type Scaled struct {
	d Distribution
	f float64
}

// NewScaled validates that f is finite and positive.
func NewScaled(d Distribution, f float64) (Scaled, error) {
	if !finite(f) || !(f > 0) {
		return Scaled{}, paramError(familyScaled, paramFactor, f)
	}
	return Scaled{d: d, f: f}, nil
}

func (d Scaled) Fill(r *rand.Rand, dst []float64) {
	d.d.Fill(r, dst)
	for i := range dst {
		dst[i] *= d.f
	}
}

func (d Scaled) Mean() float64 { return d.f * d.d.Mean() }

func (d Scaled) Var() float64 { return d.f * d.f * d.d.Var() }

func (d Scaled) Support() (lo, hi float64) {
	lo, hi = d.d.Support()
	return d.f * lo, d.f * hi
}

func (d Scaled) CDF(x float64) float64 { return d.d.CDF(x / d.f) }

func (d Scaled) Survival(x float64) float64 { return d.d.Survival(x / d.f) }

func (d Scaled) Quantile(p float64) float64 { return d.f * d.d.Quantile(p) }

func (d Scaled) QuantileUpper(q float64) float64 { return d.f * d.d.QuantileUpper(q) }

// Laplace is E[e^{−s·f·X}], X's transform at f·s.
func (d Scaled) Laplace(s complex128) (complex128, bool) {
	return d.d.Laplace(s * complex(d.f, 0))
}

// Nodes are X's nodes times f, with X's weights.
func (d Scaled) Nodes(x, w []float64) bool {
	if !d.d.Nodes(x, w) {
		return false
	}
	for i := range x {
		x[i] *= d.f
	}
	return true
}

// Factor is f.
func (d Scaled) Factor() float64 { return d.f }

// Base is X.
func (d Scaled) Base() Distribution { return d.d }

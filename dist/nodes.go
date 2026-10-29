package dist

import "gonum.org/v1/gonum/integrate/quad"

// Nodes fills x and w with len(x) quadrature nodes and weights such that
// Σ w_i·g(x_i) approximates E[g(X)] with the family's own density as the
// weight function, exact for polynomials g of degree below 2·len(x). The
// weights sum to 1. It returns false for a family without such a rule,
// which the caller then integrates through the quantile function.

func (d Beta) Nodes(x, w []float64) bool { return betaNodes(d.alpha, d.beta, x, w) }

func (d PERT) Nodes(x, w []float64) bool {
	if !d.beta.Nodes(x, w) {
		return false
	}
	for i := range x {
		x[i] = d.low + d.span*x[i]
	}
	return true
}

func (d Uniform) Nodes(x, w []float64) bool {
	quad.Legendre{}.FixedLocations(x, w, d.a, d.b)
	for i := range w {
		w[i] /= d.span
	}
	return true
}

// Nodes of a Categorical are its atoms, exact for every g, when they fit.
func (d Categorical) Nodes(x, w []float64) bool {
	if len(x) < len(d.values) {
		return false
	}
	for i := range x {
		x[i], w[i] = 0, 0
	}
	copy(x, d.values)
	copy(w, d.probs)
	return true
}

func (d Constant) Nodes(x, w []float64) bool {
	for i := range x {
		x[i], w[i] = d.c, 0
	}
	if len(w) > 0 {
		w[0] = 1
	}
	return true
}

func (Gamma) Nodes(_, _ []float64) bool            { return false }
func (LogPERT) Nodes(_, _ []float64) bool          { return false }
func (Poisson) Nodes(_, _ []float64) bool          { return false }
func (LogUniform) Nodes(_, _ []float64) bool       { return false }
func (Triangular) Nodes(_, _ []float64) bool       { return false }
func (Exponential) Nodes(_, _ []float64) bool      { return false }
func (Lognormal) Nodes(_, _ []float64) bool        { return false }
func (Weibull) Nodes(_, _ []float64) bool          { return false }
func (Normal) Nodes(_, _ []float64) bool           { return false }
func (TruncatedNormal) Nodes(_, _ []float64) bool  { return false }
func (Pareto) Nodes(_, _ []float64) bool           { return false }
func (GPD) Nodes(_, _ []float64) bool              { return false }
func (Bernoulli) Nodes(_, _ []float64) bool        { return false }
func (Binomial) Nodes(_, _ []float64) bool         { return false }
func (Geometric) Nodes(_, _ []float64) bool        { return false }
func (NegativeBinomial) Nodes(_, _ []float64) bool { return false }
func (Truncated) Nodes(_, _ []float64) bool        { return false }
func (Capped) Nodes(_, _ []float64) bool           { return false }

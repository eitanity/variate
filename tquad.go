package variate

import (
	"math/cmplx"

	"code.eitanity.com/variate/dist"
	"gonum.org/v1/gonum/integrate/quad"
)

// compound applies source k's mixed-Poisson generating function P_N to
// gc, writing out; eps is the quadrature error, 0 for a closed form, and
// reason is non-empty when the quadrature is refused.
func (t *transformer) compound(k int, gc, out []complex128, f float64) (eps float64, reason string) {
	d := t.m.pools[t.m.srcs[k].pool]
	if pmf := t.m.countPMF[k]; pmf != nil {
		if t.work+len(pmf)*len(gc) > t.cfg.MaxWork {
			return 0, t.budgetReason(t.grid)
		}
		for i, w := range gc {
			out[i] = polyAt(pmf, w)
		}
		t.work += len(pmf) * len(gc)
		return t.m.countTail[k], ""
	}
	if _, ok := d.Laplace(0); ok {
		if t.work+len(gc) > t.cfg.MaxWork {
			return 0, t.budgetReason(t.grid)
		}
		for i, w := range gc {
			out[i], _ = d.Laplace(complex(f, 0) * (1 - w))
		}
		t.work += len(gc)
		return 0, ""
	}
	eps, reason = t.quadratureCompound(d, gc, out, f)
	if reason != "" {
		reason += " for source " + t.m.sourceIDs[k]
	}
	return eps, reason
}

// quadratureCompound evaluates P_N(w) = E_λ[exp(λ(w − 1))] by quadrature
// over the rate: the family's own Gauss rule where it has one, otherwise
// Gauss–Legendre in the quantile domain with the tail substitution, each
// half of the range on the side that keeps its precision. The nodes double
// until the sums agree to quadTol absolutely at every grid point: first on
// a sample of the grid, which finds the count cheaply or refuses cheaply,
// then on the first half of the grid from that count, the second half
// being the first's mirror image under conjugation.
func (t *transformer) quadratureCompound(d distribution, gc, out []complex128, f float64) (eps float64, reason string) {
	sample := sampleOf(gc, quadSample)
	n, reason := t.converged(d, sample, make([]complex128, len(sample)), quadStart, quadDoublings, f)
	if reason != "" {
		return 0, reason
	}
	if n*len(gc) > quadWorkCap {
		return 0, "frequency quadrature needs " + itoa(n) + " nodes at " + itoa(len(gc)) + " grid points, above the work cap,"
	}
	// The first half from the count the sample found, allowing one more
	// doubling before refusing.
	half := len(gc)/2 + 1
	n, reason = t.converged(d, gc[:half], out[:half], max(quadStart, n/2), 2, f)
	if reason != "" {
		return 0, reason
	}
	mirror(out)
	t.quadN = max(t.quadN, n)
	return t.lastEps, ""
}

// mirror fills the second half of a conjugate-symmetric sequence from
// the first.
func mirror(out []complex128) {
	m := len(out)
	// bound: M/2 points
	for j := 1; j < (m+1)/2; j++ {
		out[m-j] = cmplx.Conj(out[j])
	}
}

// converged doubles the nodes from n, at most doublings times, until two
// counts agree on every point of gc, returning the larger count and
// leaving its values in out and the difference in lastEps; the reason
// when they never agree or the next count would exceed the work budget.
func (t *transformer) converged(d distribution, gc, out []complex128, n, doublings int, f float64) (int, string) {
	prev := make([]complex128, len(gc))
	// bound: doublings, itself at most quadDoublings
	for i := 0; i <= doublings && n <= quadMaxNodes; i++ {
		if t.work+n*len(gc) > t.cfg.MaxWork {
			return 0, t.budgetReason(t.grid)
		}
		x := make([]float64, n)
		w := make([]float64, n)
		if !d.Nodes(x, w) {
			quantileNodes(d, x, w)
		}
		evalMixture(x, w, gc, out, f)
		t.work += n * len(gc)
		if i > 0 {
			if diff := maxAbsDiff(out, prev); diff <= quadTol {
				t.lastEps = diff
				return n, ""
			}
		}
		copy(prev, out)
		n *= 2
	}
	return 0, reasonFreqNoConv
}

// sampleOf picks up to n evenly spaced points of gc.
func sampleOf(gc []complex128, n int) []complex128 {
	if len(gc) <= n {
		return gc
	}
	out := make([]complex128, n)
	for i := range out {
		out[i] = gc[i*len(gc)/n]
	}
	return out
}

// quantileNodes fills a 2n-point rule over the quantile function: n
// Gauss–Legendre points with the t^8 substitution on each half.
func quantileNodes(q dist.Quantiler, x, w []float64) {
	n := len(x) / 2
	gx := make([]float64, n)
	gw := make([]float64, n)
	quad.Legendre{}.FixedLocations(gx, gw, 0, 1)
	for i := range gx {
		tk, jac := substituteT(gx[i])
		x[2*i] = q.Quantile(0.5 * tk)
		x[2*i+1] = q.QuantileUpper(0.5 * tk)
		w[2*i] = gw[i] * jac * 0.5
		w[2*i+1] = w[2*i]
	}
}

// evalMixture fills out with Σ_i w_i·exp(f·λ_i(gc − 1)).
func evalMixture(x, w []float64, gc, out []complex128, f float64) {
	for j := range out {
		out[j] = 0
	}
	for i := range x {
		lam := complex(x[i]*f, 0)
		weight := complex(w[i], 0)
		for j, g := range gc {
			out[j] += weight * cmplx.Exp(lam*(g-1))
		}
	}
}

// substituteT returns t^8 and 8·t^7.
func substituteT(t float64) (tk, jac float64) {
	t2 := t * t
	t4 := t2 * t2
	return t4 * t4, 8 * t4 * t2 * t
}

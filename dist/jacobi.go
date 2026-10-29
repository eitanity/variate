package dist

import "math"

// Gauss–Jacobi nodes and weights for the Beta density on [0, 1] by
// Golub–Welsch: the eigenvalues and first eigenvector components of the
// Jacobi matrix of the orthogonal polynomials, found with the implicit QL
// iteration on the symmetric tridiagonal matrix.

// qlIterations bounds the QL sweeps per eigenvalue.
const qlIterations = 60

// betaNodes fills x and w with len(x) Gauss–Jacobi nodes and weights for
// the Beta(alpha, beta) density on [0, 1]; the weights sum to 1. It
// returns false when the iteration did not converge.
func betaNodes(alpha, beta float64, x, w []float64) bool {
	n := len(x)
	if n == 0 {
		return true
	}
	// Jacobi weight (1−t)^a (1+t)^b on [−1, 1] with a = beta−1, b = alpha−1
	// corresponds to x = (1+t)/2 ~ Beta(alpha, beta).
	a, b := beta-1, alpha-1
	diag := make([]float64, n)
	off := make([]float64, n)
	ab := a + b
	diag[0] = (b - a) / (ab + 2)
	// bound: n nodes
	for k := 1; k < n; k++ {
		fk := float64(k)
		den := (2*fk + ab) * (2*fk + ab + 2)
		diag[k] = (b*b - a*a) / den
		num := 4 * fk * (fk + a) * (fk + b) * (fk + ab)
		den2 := (2*fk + ab) * (2*fk + ab) * (2*fk + ab + 1) * (2*fk + ab - 1)
		off[k] = math.Sqrt(num / den2)
	}
	// z holds the first row of the eigenvector matrix.
	z := make([]float64, n)
	z[0] = 1
	if !tql(diag, off, z) {
		return false
	}
	sum := 0.0
	// bound: n nodes
	for i := 0; i < n; i++ {
		x[i] = (1 + diag[i]) / 2
		w[i] = z[i] * z[i]
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	return true
}

// tql is the implicit QL iteration for a symmetric tridiagonal matrix
// with diagonal d and sub-diagonal e (e[0] unused), accumulating the first
// row of the eigenvectors in z. On return d holds the eigenvalues.
func tql(d, e, z []float64) bool {
	n := len(d)
	// bound: n nodes
	for i := 1; i < n; i++ {
		e[i-1] = e[i]
	}
	e[n-1] = 0
	// bound: n eigenvalues
	for l := 0; l < n; l++ {
		// bound: qlIterations
		for iter := 0; ; iter++ {
			m := splitPoint(d, e, l)
			if m == l {
				break
			}
			if iter >= qlIterations {
				return false
			}
			qlSweep(d, e, z, l, m)
		}
	}
	return true
}

// splitPoint finds the first negligible sub-diagonal element at or after l.
func splitPoint(d, e []float64, l int) int {
	n := len(d)
	m := l
	// bound: n nodes
	for ; m < n-1; m++ {
		dd := math.Abs(d[m]) + math.Abs(d[m+1])
		if math.Abs(e[m]) <= 0x1p-52*dd {
			break
		}
	}
	return m
}

// qlSweep applies one implicit QL step with a Wilkinson shift to rows
// l … m, rotating the eigenvector row z with it.
func qlSweep(d, e, z []float64, l, m int) {
	g := (d[l+1] - d[l]) / (2 * e[l])
	r := math.Hypot(g, 1)
	g = d[m] - d[l] + e[l]/(g+math.Copysign(r, g))
	s, c, p := 1.0, 1.0, 0.0
	i := m - 1
	// bound: rows m−1 down to l
	for ; i >= l; i-- {
		f := s * e[i]
		bb := c * e[i]
		r = math.Hypot(f, g)
		e[i+1] = r
		if r == 0 {
			d[i+1] -= p
			e[m] = 0
			break
		}
		s = f / r
		c = g / r
		g = d[i+1] - p
		r = (d[i]-g)*s + 2*c*bb
		p = s * r
		d[i+1] = g + p
		g = c*r - bb
		f = z[i+1]
		z[i+1] = s*z[i] + c*f
		z[i] = c*z[i] - s*f
	}
	if r == 0 && i >= l {
		return
	}
	d[l] -= p
	e[l] = g
	e[m] = 0
}

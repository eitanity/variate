package dist

import "math"

// GaussRule fills x and w with the len(x)-point Gauss rule of the
// discrete measure (px, pw): nodes and weights that integrate every
// polynomial of degree below 2·len(x) as the measure does. The
// recurrence is the Stieltjes procedure on orthonormal polynomials, and
// the rule the Golub–Welsch eigenvalues and first eigenvector components
// of its Jacobi matrix, the nodes ascending. The weights sum to 1. It
// returns false when the measure has fewer finite points of positive
// weight than nodes asked for, or the iteration does not converge.
func GaussRule(px, pw, x, w []float64) bool {
	n := len(x)
	var fx, fw []float64
	mass := 0.0
	// bound: the measure's points
	for i, v := range pw {
		if v > 0 && !math.IsInf(px[i], 0) && !math.IsNaN(px[i]) {
			fx, fw = append(fx, px[i]), append(fw, v)
			mass += v
		}
	}
	if n == 0 || len(fx) < n || !(mass > 0) {
		return false
	}
	diag, off, ok := stieltjes(fx, fw, mass, n)
	if !ok {
		return false
	}
	z := make([]float64, n)
	z[0] = 1
	if !tql(diag, off, z) {
		return false
	}
	sum := 0.0
	// bound: n nodes
	for i := range n {
		x[i] = diag[i]
		w[i] = z[i] * z[i]
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	// bound: n nodes, sorted by insertion
	for i := 1; i < n; i++ {
		// bound: i swaps
		for j := i; j > 0 && x[j] < x[j-1]; j-- {
			x[j], x[j-1] = x[j-1], x[j]
			w[j], w[j-1] = w[j-1], w[j]
		}
	}
	return true
}

// stieltjes is the Jacobi matrix of the measure's first n orthonormal
// polynomials: its diagonal, and its sub-diagonal from index 1.
func stieltjes(px, pw []float64, mass float64, n int) (diag, off []float64, ok bool) {
	diag = make([]float64, n)
	off = make([]float64, n)
	q := make([]float64, len(px)) // q_k at each point
	prev := make([]float64, len(px))
	for i := range q {
		q[i] = 1 / math.Sqrt(mass)
	}
	// bound: n nodes
	for k := range n {
		a := 0.0
		for i, v := range pw {
			a += v * px[i] * q[i] * q[i]
		}
		diag[k] = a
		if k == n-1 {
			break
		}
		norm := 0.0
		for i, v := range pw {
			r := (px[i]-a)*q[i] - off[k]*prev[i]
			prev[i] = r
			norm += v * r * r
		}
		b := math.Sqrt(norm)
		if !(b > 0) {
			return nil, nil, false
		}
		off[k+1] = b
		for i := range q {
			q[i], prev[i] = prev[i]/b, q[i]
		}
	}
	return diag, off, true
}

// HermiteRule fills x and w with the len(x)-point Gauss rule of the
// standard normal: the eigenvalues and first eigenvector components of
// the Jacobi matrix of its orthonormal polynomials, sub-diagonal √k,
// the nodes ascending and the weights summing to 1. It returns false
// when the iteration does not converge.
func HermiteRule(x, w []float64) bool {
	n := len(x)
	if n == 0 {
		return false
	}
	e := make([]float64, n)
	z := make([]float64, n)
	z[0] = 1
	// bound: n nodes
	for k := 1; k < n; k++ {
		e[k] = math.Sqrt(float64(k))
	}
	clear(x)
	if !tql(x, e, z) {
		return false
	}
	sum := 0.0
	// bound: n nodes
	for i := range n {
		w[i] = z[i] * z[i]
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	// bound: n nodes, sorted by insertion
	for i := 1; i < n; i++ {
		// bound: i swaps
		for j := i; j > 0 && x[j] < x[j-1]; j-- {
			x[j], x[j-1] = x[j-1], x[j]
			w[j], w[j-1] = w[j-1], w[j]
		}
	}
	return true
}

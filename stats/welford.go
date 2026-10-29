package stats

import "math"

// Welford holds a running count, mean and sum of squared deviations.
type Welford struct {
	N    int64
	Mean float64
	M2   float64
}

// Add accumulates x.
func (w *Welford) Add(x float64) {
	w.N++
	d := x - w.Mean
	w.Mean += d / float64(w.N)
	w.M2 += d * (x - w.Mean)
}

// Merge combines another accumulator by Chan's parallel formula.
func (w *Welford) Merge(o Welford) {
	if o.N == 0 {
		return
	}
	if w.N == 0 {
		*w = o
		return
	}
	n := float64(w.N + o.N)
	d := o.Mean - w.Mean
	w.Mean += d * float64(o.N) / n
	w.M2 += o.M2 + d*d*float64(w.N)*float64(o.N)/n
	w.N += o.N
}

// Var is the sample variance (n − 1), NaN below two values.
func (w Welford) Var() float64 {
	if w.N < 2 {
		return math.NaN()
	}
	return w.M2 / float64(w.N-1)
}

// StdDev is the sample standard deviation.
func (w Welford) StdDev() float64 { return math.Sqrt(w.Var()) }

// StdErr is the standard error of the mean for independent values.
func (w Welford) StdErr() float64 { return math.Sqrt(w.Var() / float64(w.N)) }

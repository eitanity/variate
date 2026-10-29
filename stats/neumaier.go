package stats

import "math"

// Neumaier is a compensated running sum: the low-order part lost by each
// addition is kept and added back at the end.
type Neumaier struct {
	Sum, C float64
}

// Add accumulates x.
func (n *Neumaier) Add(x float64) {
	t := n.Sum + x
	if math.Abs(n.Sum) >= math.Abs(x) {
		n.C += (n.Sum - t) + x
	} else {
		n.C += (x - t) + n.Sum
	}
	n.Sum = t
}

// Merge adds another sum, both parts compensated.
func (n *Neumaier) Merge(o Neumaier) {
	n.Add(o.Sum)
	n.Add(o.C)
}

// Value returns the compensated sum.
func (n Neumaier) Value() float64 { return n.Sum + n.C }

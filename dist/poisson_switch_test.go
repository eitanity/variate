package dist

import (
	"fmt"
	"math"
	"testing"
)

// BenchmarkPoissonMethods times inversion against PTRS across the rates
// where they cross, to set the switch point.
func BenchmarkPoissonMethods(b *testing.B) {
	for _, l := range []float64{10, 12, 15, 18, 20, 25, 30, 40} {
		r := newRand(1)
		expNeg := math.Exp(-l)
		b.Run(fmt.Sprintf("inversion/lambda%g", l), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				poissonInversion(r, l, expNeg)
			}
		})
		c := newPTRS(l)
		b.Run(fmt.Sprintf("ptrs/lambda%g", l), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				c.draw(r)
			}
		})
	}
}

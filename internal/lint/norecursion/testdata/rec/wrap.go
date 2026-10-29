package rec

type Shape interface{ Area() float64 }

type Square struct{ s float64 }

func (q Square) Area() float64 { return q.s * q.s }

// Scaled wraps a Shape and is itself a Shape.
type Scaled struct {
	inner Shape
	k     float64
}

func (s Scaled) Area() float64 { return s.k * s.inner.Area() }

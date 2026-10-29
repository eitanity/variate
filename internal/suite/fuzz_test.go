package suite

import (
	"math"
	"strconv"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// graphFromBytes builds a graph from fuzz input: each byte picks a node's
// shape, and the parameters come from the bytes that follow. Most graphs
// are valid, so the compiled paths get exercised; about one choice in
// sixteen is deliberately wrong, for the error paths.
func graphFromBytes(data []byte) variate.Graph {
	next := func() byte {
		if len(data) == 0 {
			return 0
		}
		b := data[0]
		data = data[1:]
		return b
	}
	n := int(next()%8) + 1
	g := variate.Graph{}
	for i := 0; i < n; i++ {
		nd := variate.Node{ID: "n" + strconv.Itoa(i)}
		kind := next()
		// Targets are later nodes nearly always, so most graphs are DAGs.
		target := func() string {
			b := int(next())
			if b%16 == 0 || i == n-1 {
				return "n" + strconv.Itoa(b%(n+1))
			}
			return "n" + strconv.Itoa(i+1+b%(n-i-1))
		}
		shape := kind >> 1 & 3
		if shape != 0 && i == n-1 {
			shape = 0 // the last node is a leaf
		}
		if kind&1 != 0 {
			f := variate.Function{Name: "f", Category: "c" + strconv.Itoa(int(kind>>6)), Model: distFromBytes(next(), next(), next(), slotValue), Scope: variate.Scope(kind >> 4 & 1)}
			if shape != 0 && kind>>5&1 != 0 {
				f.Basis = variate.PerTime
			}
			nd.Functions = append(nd.Functions, f)
		}
		switch shape {
		case 1:
			nd.Next = target()
			if kind&8 != 0 {
				nd.NextHolding = distFromBytes(next(), next(), next(), slotHolding)
			}
		case 2, 3:
			nd.Mode = variate.Mode(kind >> 3 & 1)
			count := int(next()%3) + 1
			if nd.Mode == variate.Absolute && kind>>7 != 0 {
				count = 2 // an uncertain probability wants exactly two outcomes
			}
			for j := 0; j < count; j++ {
				o := variate.Outcome{To: target()}
				last := j == count-1
				switch {
				case nd.Mode == variate.Sequential && last:
				case nd.Mode == variate.Absolute && kind>>7 != 0:
					if j == 0 {
						o.P = distFromBytes(next()|1, next(), next(), slotProbability)
					}
				default:
					o.P = distFromBytes(next(), next(), next(), slotProbability)
				}
				if next()&7 == 0 {
					o.Holding = distFromBytes(next(), next(), next(), slotHolding)
				}
				nd.Outcomes = append(nd.Outcomes, o)
			}
		}
		g.Nodes = append(g.Nodes, nd)
	}
	for k := 0; k < int(next()%3)+1; k++ {
		g.Sources = append(g.Sources, variate.Source{ID: "s" + strconv.Itoa(k), Frequency: distFromBytes(next()|1, next(), next(), slotValue), To: "n" + strconv.Itoa(int(next())%(n+1))})
	}
	return g
}

// Slots a fuzzed distribution is meant for; the families chosen mostly
// fit the slot, with one kind in nine left to the error paths.
const (
	slotValue = iota
	slotHolding
	slotProbability
)

// distFromBytes picks a family and parameters for a slot; a nil result
// stands for a missing distribution.
func distFromBytes(kind, a, b byte, slot int) dist.Distribution {
	x := float64(a) / 32 // 0 … 8
	y := float64(b)/32 + 0.01
	var d dist.Distribution
	var err error
	if slot == slotProbability {
		switch kind % 16 {
		case 0:
			d, err = dist.NewConstant(x - 1) // may lie outside [0, 1]
		case 1, 2, 3:
			d, err = dist.NewBeta(y, y*2)
		case 4, 5:
			d, err = dist.NewPERT(float64(a)/512, float64(a)/512+0.1, float64(a)/512+0.3, dist.DefaultPERTShape)
		default:
			d, err = dist.NewConstant(float64(a) / 256)
		}
	} else {
		switch kind % 9 {
		case 0:
			if b%16 == 0 {
				return nil
			}
			d, err = dist.NewConstant(x + y)
		case 1:
			d, err = dist.NewConstant(x)
		case 2:
			d, err = dist.NewPERT(x, x+y/2, x+y, dist.DefaultPERTShape)
		case 3:
			d, err = dist.NewLognormal(x-1, y)
		case 4:
			d, err = dist.NewPareto(y, 1.2+x/4)
		case 5:
			d, err = dist.NewGamma(y, y)
		case 6:
			d, err = dist.NewUniform(x, x+y)
		case 7:
			d, err = dist.NewConstant(x - 1) // may be negative
		default:
			d, err = dist.NewExponential(math.Abs(x) + y)
		}
	}
	if err != nil {
		return nil
	}
	return d
}

// FuzzCompile: Compile on arbitrary graphs returns an error or a model,
// never panics, and a compiled model's expectations are finite or
// infinite, never NaN without delays.
func FuzzCompile(f *testing.F) {
	f.Add([]byte{3, 0x07, 1, 40, 40, 1, 200, 0x02, 0, 0x00, 1, 64, 32, 0})
	f.Add([]byte{1, 0x01, 2, 32, 64})
	f.Add([]byte("diamond-ish seed input with outcomes"))
	f.Fuzz(func(t *testing.T, data []byte) {
		g := graphFromBytes(data)
		m, _, err := g.Compile(variate.Config{AllowUnconverged: true})
		if err != nil {
			return
		}
		e, err := m.Analytic()
		if err != nil {
			t.Fatalf("analytic after compile: %v", err)
		}
		if math.IsNaN(e.Mean) || math.IsNaN(e.PZero()) || math.IsNaN(e.StepsPerPeriod) || (!m.HasDelays() && math.IsNaN(e.Var)) {
			t.Fatalf("NaN expectation for %+v: mean %g pzero %g steps %g var %g", g, e.Mean, e.PZero(), e.StepsPerPeriod, e.Var)
		}
		for _, x := range e.EntriesPer {
			if math.IsNaN(x) {
				t.Fatalf("NaN entries for %+v", g)
			}
		}
	})
}

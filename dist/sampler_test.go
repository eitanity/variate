package dist

import (
	"math"
	"math/rand/v2"
	"testing"
)

// countingDist records every Fill length.
type countingDist struct {
	inner Distribution
	fills *[]int
}

func (c countingDist) Fill(r *rand.Rand, dst []float64) {
	*c.fills = append(*c.fills, len(dst))
	c.inner.Fill(r, dst)
}
func (c countingDist) Mean() float64             { return c.inner.Mean() }
func (c countingDist) Var() float64              { return c.inner.Var() }
func (c countingDist) Support() (lo, hi float64) { return c.inner.Support() }
func (c countingDist) CDF(x float64) float64     { return c.inner.CDF(x) }
func (c countingDist) Laplace(s complex128) (complex128, bool) {
	return c.inner.Laplace(s)
}
func (c countingDist) Nodes(x, w []float64) bool  { return c.inner.Nodes(x, w) }
func (c countingDist) Quantile(p float64) float64 { return c.inner.Quantile(p) }
func (c countingDist) QuantileUpper(q float64) float64 {
	return c.inner.QuantileUpper(q)
}
func (c countingDist) Survival(x float64) float64 { return c.inner.Survival(x) }

func TestSamplerFillSchedule(t *testing.T) {
	u, _ := NewUniform(0, 1)
	var fills []int
	ss := NewSamplerSet([]Distribution{countingDist{u, &fills}}, 1024)
	ss.Reset(1, 0)
	s := ss.Sampler(0)
	for i := 0; i < 32+128+512+1024+1024+1; i++ {
		s.Next()
	}
	want := []int{32, 128, 512, 1024, 1024, 1024}
	if len(fills) != len(want) {
		t.Fatalf("fills %v", fills)
	}
	for i := range want {
		if fills[i] != want[i] {
			t.Fatalf("fills %v want %v", fills, want)
		}
	}
	// A reset empties the pool and restarts the schedule.
	fills = fills[:0]
	ss.Reset(1, 1)
	s.Next()
	if len(fills) != 1 || fills[0] != 32 {
		t.Fatalf("after reset fills %v", fills)
	}
}

// TestSamplerDeterminism: a stream depends on (seed, task, k) only, not on
// how calls to other samplers are interleaved.
func TestSamplerDeterminism(t *testing.T) {
	u, _ := NewUniform(0, 1)
	g, _ := NewGamma(2, 1)
	dists := []Distribution{u, g}
	a := NewSamplerSet(dists, 64)
	b := NewSamplerSet(dists, 64)
	a.Reset(42, 7)
	b.Reset(42, 7)
	var fromA, fromB []float64
	for i := 0; i < 500; i++ {
		fromA = append(fromA, a.Sampler(1).Next())
		a.Sampler(0).Next()
		a.Sampler(0).Next()
	}
	for i := 0; i < 500; i++ {
		fromB = append(fromB, b.Sampler(1).Next())
	}
	for i := range fromA {
		if fromA[i] != fromB[i] {
			t.Fatalf("draw %d differs: %g vs %g", i, fromA[i], fromB[i])
		}
	}
	// Different task, different stream; same task re-reset, same stream.
	b.Reset(42, 8)
	if b.Sampler(1).Next() == fromA[0] {
		t.Error("task 8 repeats task 7")
	}
	b.Reset(42, 7)
	if b.Sampler(1).Next() != fromA[0] {
		t.Error("re-reset does not reproduce")
	}
	ca := a.Control().Uint64()
	b.Reset(42, 7)
	a.Reset(42, 7)
	if a.Control().Uint64() != ca || b.Control().Uint64() != ca {
		t.Error("control stream not reproducible")
	}
}

// TestStreamIndependence: two distributions in one task are uncorrelated,
// and the control stream is independent of both.
func TestStreamIndependence(t *testing.T) {
	u, _ := NewUniform(0, 1)
	ss := NewSamplerSet([]Distribution{u, u}, 1024)
	ss.Reset(99, 3)
	n := nDraws
	var sxy, sxc float64
	for i := 0; i < n; i++ {
		x := ss.Sampler(0).Next() - 0.5
		y := ss.Sampler(1).Next() - 0.5
		c := ss.Control().Float64() - 0.5
		sxy += x * y
		sxc += x * c
	}
	// Each product has variance 1/144; the sum's sd is sqrt(n/144).
	se := math.Sqrt(float64(n) / 144)
	if z := sxy / se; math.Abs(z) > zLimit {
		t.Errorf("sampler streams correlated, z=%.2f", z)
	}
	if z := sxc / se; math.Abs(z) > zLimit {
		t.Errorf("control stream correlated with a sampler, z=%.2f", z)
	}
}

func TestSamplerKeysDistinct(t *testing.T) {
	seen := map[uint64]bool{}
	for j := uint32(0); j < 64; j++ {
		for k := uint64(0); k < 64; k++ {
			key := streamKey(j, k)
			if seen[key] {
				t.Fatalf("key collision at task %d dist %d", j, k)
			}
			seen[key] = true
		}
	}
}

func TestSamplerNoAlloc(t *testing.T) {
	p, _ := NewPERT(0, 2, 10, 4)
	ss := NewSamplerSet([]Distribution{p, p}, 256)
	ss.Reset(1, 0)
	if a := testing.AllocsPerRun(1000, func() {
		ss.Sampler(0).Next()
		ss.Sampler(1).Next()
	}); a != 0 {
		t.Errorf("Next allocates: %g per run", a)
	}
	if a := testing.AllocsPerRun(100, func() { ss.Reset(1, 1) }); a != 0 {
		t.Errorf("Reset allocates: %g per run", a)
	}
}

func BenchmarkSamplerNext(b *testing.B) {
	p, _ := NewPERT(0, 2, 10, 4)
	ss := NewSamplerSet([]Distribution{p}, 1024)
	ss.Reset(1, 0)
	s := ss.Sampler(0)
	sum := 0.0
	for i := 0; i < b.N; i++ {
		sum += s.Next()
	}
	_ = sum
}

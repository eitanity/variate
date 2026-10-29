package stats

import (
	"math"
	"math/big"
	"math/rand/v2"
	"sort"
	"testing"
)

func TestNeumaierAgainstExact(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	var n Neumaier
	exact := new(big.Float).SetPrec(300)
	plain := 0.0
	n.Add(1e16)
	exact.Add(exact, big.NewFloat(1e16))
	plain += 1e16
	for i := 0; i < 1_000_000; i++ {
		x := r.Float64()
		n.Add(x)
		plain += x
		exact.Add(exact, big.NewFloat(x))
	}
	want, _ := exact.Float64()
	if got := n.Value(); math.Abs(got-want) > 2*math.Abs(want)*0x1p-52 {
		t.Errorf("neumaier %.17g want %.17g", got, want)
	}
	if math.Abs(plain-want) < 1e3 {
		t.Logf("plain sum happened to be close: %.17g vs %.17g", plain, want)
	}
}

func TestNeumaierMerge(t *testing.T) {
	var a, b, all Neumaier
	for i := 0; i < 10000; i++ {
		x := math.Pow(10, float64(i%40)-20)
		all.Add(x)
		if i%2 == 0 {
			a.Add(x)
		} else {
			b.Add(x)
		}
	}
	a.Merge(b)
	if math.Abs(a.Value()-all.Value()) > 4*all.Value()*0x1p-52 {
		t.Errorf("merged %.17g sequential %.17g", a.Value(), all.Value())
	}
}

func TestWelfordAgainstTwoPass(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	xs := make([]float64, 200000)
	var w Welford
	for i := range xs {
		xs[i] = 1e6 + r.NormFloat64()
		w.Add(xs[i])
	}
	m := 0.0
	for _, x := range xs {
		m += x
	}
	m /= float64(len(xs))
	v := 0.0
	for _, x := range xs {
		v += (x - m) * (x - m)
	}
	v /= float64(len(xs) - 1)
	if math.Abs(w.Mean-m) > 1e-12*m || math.Abs(w.Var()-v) > 1e-9*v {
		t.Errorf("welford %.17g %.17g two-pass %.17g %.17g", w.Mean, w.Var(), m, v)
	}
}

func TestWelfordMergeIsChan(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	var seq Welford
	parts := make([]Welford, 7)
	for i := 0; i < 100000; i++ {
		x := math.Exp(3 * r.NormFloat64())
		seq.Add(x)
		parts[i%7].Add(x)
	}
	var merged Welford
	for _, p := range parts {
		merged.Merge(p)
	}
	if merged.N != seq.N || math.Abs(merged.Mean-seq.Mean) > 1e-12*seq.Mean || math.Abs(merged.M2-seq.M2) > 1e-11*seq.M2 {
		t.Errorf("merged %+v sequential %+v", merged, seq)
	}
	var empty Welford
	empty.Merge(seq)
	if empty != seq {
		t.Error("merge into empty")
	}
}

func TestLogHistQuantilesAndEdges(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	const res = 1e-3
	mean := math.Exp(0.5)
	h := NewLogHist(mean, res)
	xs := make([]float64, 500000)
	for i := range xs {
		xs[i] = math.Exp(r.NormFloat64())
		h.Add(xs[i])
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	for _, p := range []float64{0.001, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999} {
		got, clipped := h.Quantile(p)
		want := QuantileSorted(sorted, p)
		if clipped || math.Abs(got-want) > res*want {
			t.Errorf("quantile %g: hist %g exact %g clipped %v", p, got, want, clipped)
		}
	}
	// At a bin edge the exceedance is exact; between edges it is off by at
	// most the bin's mass.
	for _, k := range []int{h.index(0.3), h.index(1), h.index(5)} {
		e := h.Edge(k)
		raw := int64(len(sorted) - sort.SearchFloat64s(sorted, e))
		if got := h.AtOrAbove(e); got != raw {
			t.Errorf("edge %g: hist %d raw %d", e, got, raw)
		}
		mid := e * (1 + res/2)
		raw = int64(len(sorted) - sort.SearchFloat64s(sorted, mid))
		if got := h.AtOrAbove(mid); got < raw || got-raw > h.Count(k) {
			t.Errorf("mid %g: hist %d raw %d bin %d", mid, got, raw, h.Count(k))
		}
	}
	if h.N != int64(len(xs)) || h.Zero != 0 || h.Under != 0 || h.Over != 0 {
		t.Errorf("counts N=%d zero=%d under=%d over=%d", h.N, h.Zero, h.Under, h.Over)
	}
}

func TestLogHistOutsideAndMerge(t *testing.T) {
	h := NewLogHist(1, 1e-2)
	lo, hi := h.Range()
	for _, x := range []float64{0, 0, lo / 2, lo, hi - 1, hi, hi * 10, 1, 2} {
		h.Add(x)
	}
	if h.Zero != 2 || h.Under != 1 || h.Over != 2 || h.N != 9 {
		t.Errorf("zero=%d under=%d over=%d n=%d", h.Zero, h.Under, h.Over, h.N)
	}
	if x, c := h.Quantile(0.3); x != lo || !c {
		t.Errorf("clipped low quantile %g %v", x, c)
	}
	if x, c := h.Quantile(0.95); x != hi || !c {
		t.Errorf("clipped high quantile %g %v", x, c)
	}
	if x, _ := h.Quantile(0.2); x != 0 {
		t.Errorf("zero quantile %g", x)
	}
	g := NewLogHist(1, 1e-2)
	g.Add(3)
	g.Merge(h)
	// Above 2.5: 3, hi−1 and the two Over values.
	if g.N != 10 || g.Zero != 2 || g.AtOrAbove(2.5) != 4 {
		t.Errorf("merge N=%d zero=%d above2.5=%d", g.N, g.Zero, g.AtOrAbove(2.5))
	}
	// [0.5, 1.5) holds 1; [1.5, 4) holds 2 and 3.
	var bands [2]float64
	g.Bands([]float64{0.5, 1.5, 4}, bands[:])
	if math.Abs(bands[0]-0.1) > 1e-15 || math.Abs(bands[1]-0.2) > 1e-15 {
		t.Errorf("bands %v", bands)
	}
	g.Reset()
	if g.N != 0 || g.AtOrAbove(0) != 0 {
		t.Error("reset")
	}
}

func TestIntHist(t *testing.T) {
	h := NewIntHist(10, 1e-3)
	for _, x := range []float64{0, 1, 1, 4095, 4096, 5000, 1e13} {
		h.Add(x)
	}
	if h.Zero != 1 || h.exact[1] != 2 || h.exact[4095] != 1 || h.Over != 1 || h.Under != 0 {
		t.Errorf("counts zero=%d one=%d last=%d over=%d", h.Zero, h.exact[1], h.exact[4095], h.Over)
	}
	// Sorted: 0, 1, 1, 4095, 4096, 5000, 1e13. Type 1: the 4th of 7 at p = 0.5.
	if x, _ := h.Quantile(0.5); x != 4095 {
		t.Errorf("median %g", x)
	}
	if x, _ := h.Quantile(0.4); x != 1 {
		t.Errorf("p=0.4 quantile %g", x)
	}
	if h.AtOrAbove(4095) != 4 || h.AtOrAbove(1) != 6 || h.AtOrAbove(0.5) != 6 {
		t.Errorf("at or above: %d %d %d", h.AtOrAbove(4095), h.AtOrAbove(1), h.AtOrAbove(0.5))
	}
	small := NewIntHist(1e-9, 1e-3)
	if small.Bins() != 0 {
		t.Errorf("tiny mean should have no log bins, got %d", small.Bins())
	}
	empty := NewLogHist(0, 1e-3)
	if empty.Bins() != 0 {
		t.Error("zero mean should have no bins")
	}
	if BinsFor(1e-3) < 34000 || BinsFor(1e-3) > 35000 {
		t.Errorf("BinsFor(1e-3) = %d", BinsFor(1e-3))
	}
}

func TestQuantileSorted(t *testing.T) {
	s := []float64{1, 2, 3, 4}
	for _, c := range []struct{ p, want float64 }{{0, 1}, {0.25, 1}, {0.26, 2}, {0.5, 2}, {0.75, 3}, {1, 4}} {
		if got := QuantileSorted(s, c.p); got != c.want {
			t.Errorf("p=%g got %g want %g", c.p, got, c.want)
		}
	}
	if !math.IsNaN(QuantileSorted(nil, 0.5)) {
		t.Error("empty")
	}
}

func TestNoAlloc(t *testing.T) {
	h := NewLogHist(1, 1e-3)
	var n Neumaier
	var w Welford
	if a := testing.AllocsPerRun(1000, func() { h.Add(2.5); n.Add(2.5); w.Add(2.5) }); a != 0 {
		t.Errorf("Add allocates: %g", a)
	}
}

func BenchmarkHistAdd(b *testing.B) {
	h := NewLogHist(1, 1e-3)
	r := rand.New(rand.NewPCG(1, 1))
	xs := make([]float64, 1024)
	for i := range xs {
		xs[i] = math.Exp(r.NormFloat64())
	}
	for i := 0; i < b.N; i++ {
		h.Add(xs[i&1023])
	}
}

func BenchmarkWelfordAdd(b *testing.B) {
	var w Welford
	for i := 0; i < b.N; i++ {
		w.Add(float64(i & 1023))
	}
}

func BenchmarkNeumaierAdd(b *testing.B) {
	var n Neumaier
	for i := 0; i < b.N; i++ {
		n.Add(float64(i & 1023))
	}
}

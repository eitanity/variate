package dist

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/stat/distuv"
)

const benchBuf = 1024

func benchFill(b *testing.B, d Distribution) {
	b.Helper()
	r := newRand(7)
	buf := make([]float64, benchBuf)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Fill(r, buf)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*benchBuf), "ns/draw")
}

func BenchmarkPERT(b *testing.B) {
	for _, s := range [][2]float64{{1, 5}, {3, 3}, {5, 1}} {
		for _, scale := range []float64{1, 1e9} {
			d, err := NewPERT(0, scale*(s[0]-1)/4, scale, DefaultPERTShape)
			if err != nil {
				b.Fatal(err)
			}
			b.Run(fmt.Sprintf("a%g_b%g_scale%g", s[0], s[1], scale), func(b *testing.B) { benchFill(b, d) })
		}
	}
}

func BenchmarkBetaDistuv(b *testing.B) {
	for _, s := range [][2]float64{{1, 5}, {3, 3}, {5, 1}} {
		d := distuv.Beta{Alpha: s[0], Beta: s[1], Src: newRand(7)}
		b.Run(fmt.Sprintf("a%g_b%g", s[0], s[1]), func(b *testing.B) {
			buf := make([]float64, benchBuf)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range buf {
					buf[j] = d.Rand()
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*benchBuf), "ns/draw")
		})
	}
}

func BenchmarkGamma(b *testing.B) {
	for _, k := range []float64{0.5, 1, 7.5} {
		d, err := NewGamma(k, 1)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("shape%g", k), func(b *testing.B) { benchFill(b, d) })
	}
}

func BenchmarkPoisson(b *testing.B) {
	for _, l := range []float64{0.5, 9.999, 21.999, 22, 1e4} {
		d, err := NewPoisson(l)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("lambda%g", l), func(b *testing.B) { benchFill(b, d) })
	}
}

func BenchmarkPoissonCount(b *testing.B) {
	for _, l := range []float64{0.5, 37.5} {
		b.Run(fmt.Sprintf("lambda%g", l), func(b *testing.B) {
			r := newRand(7)
			var sink uint64
			for i := 0; i < b.N; i++ {
				sink += PoissonCount(r, l)
			}
			_ = sink
		})
	}
}

// BenchmarkDist reports ns per draw for every catalogue family built so far.
func BenchmarkDist(b *testing.B) {
	inf := math.Inf(1)
	u, _ := NewUniform(0, 1)
	lu, _ := NewLogUniform(1, 1e6)
	tr, _ := NewTriangular(0, 0.3, 1)
	e, _ := NewExponential(1)
	ln, _ := NewLognormal(0, 1)
	w, _ := NewWeibull(1.5, 1)
	n, _ := NewNormal(0, 1)
	tn, _ := NewTruncatedNormal(0, 1, 0, inf)
	tnr, _ := NewTruncatedNormal(0, 1, 6, inf)
	p, _ := NewPareto(1, 1.5)
	g, _ := NewGPD(0, 1, 0.3)
	for _, c := range []struct {
		name string
		d    Distribution
	}{
		{"uniform", u}, {"loguniform", lu}, {"triangular", tr}, {"exponential", e}, {"lognormal", ln},
		{"weibull", w}, {"normal", n}, {"truncnormal_half", tn}, {"truncnormal_robert", tnr}, {"pareto", p}, {"gpd", g},
	} {
		b.Run(c.name, func(b *testing.B) { benchFill(b, c.d) })
	}
}

// BenchmarkDistuv is gonum's distuv as the baseline for the same families.
func BenchmarkDistuv(b *testing.B) {
	src := newRand(7)
	cases := []struct {
		name string
		rnd  func() float64
	}{
		{"uniform", distuv.Uniform{Min: 0, Max: 1, Src: src}.Rand},
		{"exponential", distuv.Exponential{Rate: 1, Src: src}.Rand},
		{"lognormal", distuv.LogNormal{Mu: 0, Sigma: 1, Src: src}.Rand},
		{"weibull", distuv.Weibull{K: 1.5, Lambda: 1, Src: src}.Rand},
		{"normal", distuv.Normal{Mu: 0, Sigma: 1, Src: src}.Rand},
		{"pareto", distuv.Pareto{Xm: 1, Alpha: 1.5, Src: src}.Rand},
		{"triangular", distuv.NewTriangle(0, 1, 0.3, src).Rand},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			buf := make([]float64, benchBuf)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range buf {
					buf[j] = c.rnd()
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*benchBuf), "ns/draw")
		})
	}
}

func BenchmarkDiscrete(b *testing.B) {
	bi, _ := NewBinomial(20, 0.3)
	bb, _ := NewBinomial(1e6, 0.3)
	ge, _ := NewGeometric(0.1)
	nb, _ := NewNegativeBinomial(2.5, 0.5)
	ca, _ := NewCategorical([]float64{1, 2, 5, 10}, []float64{0.5, 0.3, 0.15, 0.05})
	for _, c := range []struct {
		name string
		d    Distribution
	}{{"binomial_inversion", bi}, {"binomial_btpe", bb}, {"geometric", ge}, {"negbin", nb}, {"categorical", ca}} {
		b.Run(c.name, func(b *testing.B) { benchFill(b, c.d) })
	}
}

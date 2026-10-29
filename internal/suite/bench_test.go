package suite

import (
	"context"
	"math/rand/v2"
	"strconv"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/dist"
)

// bigGraph builds a wide DAG of n nodes in layers of about 32, each node
// selecting among three nodes of the next layer and booking a PERT value
// on entry; path lengths are n/32.
func bigGraph(n int) variate.Graph {
	const width = 32
	r := rand.New(rand.NewPCG(9, 9))
	g := variate.Graph{ID: "big"}
	layers := (n + width - 1) / width
	for i := 0; i < n; i++ {
		nd := variate.Node{ID: "n" + strconv.Itoa(i)}
		layer := i / width
		if r.IntN(3) > 0 {
			v, _ := dist.NewPERT(1, 2, 4, dist.DefaultPERTShape)
			nd.Functions = []variate.Function{{Name: "f", Category: "c" + strconv.Itoa(i%3), Model: v}}
		}
		if layer < layers-1 {
			next := min(n, (layer+2)*width)
			base := (layer + 1) * width
			for j := 0; j < 3; j++ {
				to := base + r.IntN(next-base)
				p, _ := dist.NewConstant(1.0 / 3)
				nd.Outcomes = append(nd.Outcomes, variate.Outcome{To: "n" + strconv.Itoa(to), P: p})
			}
		}
		g.Nodes = append(g.Nodes, nd)
	}
	for k := 0; k < 4; k++ {
		f, _ := dist.NewPERT(2, 5, 10, dist.DefaultPERTShape)
		g.Sources = append(g.Sources, variate.Source{ID: "s" + strconv.Itoa(k), Frequency: f, To: "n" + strconv.Itoa(r.IntN(width))})
	}
	return g
}

func benchRun(b *testing.B, g variate.Graph, cfg variate.Config, periods int64) {
	b.Helper()
	cfg.Periods = periods
	m, _, err := g.Compile(cfg)
	if err != nil {
		b.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 1); err != nil {
		b.Fatal(err)
	}
	var steps uint64
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := w.Run(context.Background(), uint64(i)+2)
		if err != nil {
			b.Fatal(err)
		}
		steps += r.Steps
	}
	b.StopTimer()
	b.ReportMetric(float64(steps)/b.Elapsed().Seconds()/1e6, "Msteps/s")
	b.ReportMetric(float64(periods)*float64(b.N)/b.Elapsed().Seconds(), "periods/s")
}

// BenchmarkWalk: steps per second on random DAGs with one worker.
// walkBlock fixes the block size of the walk benchmarks, so that they
// measure the walk and not the block policy of a fixed run.
const walkBlock = 4096

func BenchmarkWalk(b *testing.B) {
	for _, seed := range []uint64{1, 2, 3} {
		b.Run("seed"+strconv.FormatUint(seed, 10), func(b *testing.B) {
			benchRun(b, genGraph(b, seed, false), variate.Config{Workers: 1, BlockPeriods: walkBlock}, 20000)
		})
	}
	b.Run("big1000", func(b *testing.B) {
		benchRun(b, bigGraph(1000), variate.Config{Workers: 1, BlockPeriods: walkBlock}, 2000)
	})
}

// BenchmarkChain: the pump-station model, one worker and all cores.
func BenchmarkChain(b *testing.B) {
	m, err := config.Load("../../examples/pump_station.yaml")
	if err != nil {
		b.Fatal(err)
	}
	b.Run("workers1", func(b *testing.B) { benchRun(b, m.Graph, variate.Config{Workers: 1}, 50000) })
	b.Run("workersAll", func(b *testing.B) { benchRun(b, m.Graph, variate.Config{}, 200000) })
}

type nopObserver struct{}

func (nopObserver) Observe(*variate.Round) error { return nil }

// BenchmarkStream: the pump-station model with a no-op observer, with an
// empty Watch (every round delivered) and a narrow one.
func BenchmarkStream(b *testing.B) {
	m, err := config.Load("../../examples/pump_station.yaml")
	if err != nil {
		b.Fatal(err)
	}
	b.Run("off", func(b *testing.B) { benchRun(b, m.Graph, variate.Config{Workers: 1}, 50000) })
	b.Run("every", func(b *testing.B) {
		benchRun(b, m.Graph, variate.Config{Workers: 1, Observer: nopObserver{}}, 50000)
	})
	b.Run("narrow", func(b *testing.B) {
		benchRun(b, m.Graph, variate.Config{Workers: 1, Observer: nopObserver{}, Watch: variate.Watch{Leaves: []string{"inspection"}}}, 50000)
	})
}

// BenchmarkSummarize: a summary of a finished run on a warm Summary.
func BenchmarkSummarize(b *testing.B) {
	m, err := config.Load("../../examples/pump_station.yaml")
	if err != nil {
		b.Fatal(err)
	}
	cfg := variate.Config{Periods: 20000}
	c, _, _ := m.Graph.Compile(cfg)
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 1); err != nil {
		b.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.Summarize(&s)
	}
}

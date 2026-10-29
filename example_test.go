package variate_test

import (
	"context"
	"fmt"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// compoundModel is the smallest graph: a source with an uncertain rate
// feeding one node that books a value per event.
func compoundModel() variate.Graph {
	rate, _ := dist.NewPERT(2, 5, 12, dist.DefaultPERTShape)
	severity, _ := dist.NewLognormalFromMedianP90(1000, 5000)
	return variate.Graph{
		ID:      "compound",
		Sources: []variate.Source{{ID: "incident", Frequency: rate, To: "loss"}},
		Nodes: []variate.Node{{
			ID:        "loss",
			Functions: []variate.Function{{Name: "cost", Category: "money", Model: severity}},
		}},
	}
}

// Compile validates a graph; Analytic gives the exact mean and variance
// of the period total without simulating.
func ExampleGraph_Compile() {
	m, warnings, err := compoundModel().Compile(variate.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, w := range warnings {
		fmt.Println("warning:", w.Code)
	}
	e, _ := m.Analytic()
	fmt.Printf("mean %.0f  sd %.0f  P(T = 0) %.4f\n", e.Mean, sqrt(e.Var), e.PZero())
	// Output:
	// warning: expected-work
	// mean 12468  sd 12201  P(T = 0) 0.0118
}

// A fixed run of a compiled model: the workspace holds every buffer, Run
// fills its Result, Summarize turns the result into named statistics.
func ExampleCompiled_NewWorkspace() {
	m, _, _ := compoundModel().Compile(variate.Config{})
	cfg := variate.Config{Periods: 20000}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 42)
	if err != nil {
		fmt.Println(err)
		return
	}
	var s variate.Summary
	w.Summarize(&s)
	e, _ := m.Analytic()
	z := (s.Total.Mean - e.Mean) / s.Total.StdErr
	fmt.Printf("periods %d  mean within 3 standard errors of exact: %v\n", r.Periods, z > -3 && z < 3)
	fmt.Printf("p50 and p99 ordered: %v\n", s.Total.P50 < s.Total.P99)
	bands := w.Bands([]float64{0, 5000, 20000, 1e9}, make([]float64, 3))
	fmt.Printf("bands sum to one: %v\n", bands[0]+bands[1]+bands[2] > 0.999)
	// Output:
	// periods 20000  mean within 3 standard errors of exact: true
	// p50 and p99 ordered: true
	// bands sum to one: true
}

// An adaptive run stops when every convergence criterion is met.
func ExampleCompiled_Simulate() {
	m, _, _ := compoundModel().Compile(variate.Config{MinHits: 100, TargetRSE: 0.05, ContribRSE: 0.05, TailP: 0.99, TailHits: 20})
	r, err := m.Simulate(context.Background(), variate.Config{MinHits: 100, TargetRSE: 0.05, ContribRSE: 0.05, TailP: 0.99, TailHits: 20, Seed: 1})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("converged %v  periods > 0: %v\n", r.Converged, r.Periods > 0)
	// Output:
	// converged true  periods > 0: true
}

// leafCounter is an Observer: it receives every delivered round.
type leafCounter struct{ rounds, steps int }

func (c *leafCounter) Observe(r *variate.Round) error {
	c.rounds++
	c.steps += len(r.Steps)
	return nil
}

// A Watch narrows the rounds an Observer receives; here to those whose
// value is at least 10000. The Watch is read at Compile, the Observer at
// NewWorkspace.
func ExampleObserver() {
	cfg := variate.Config{Periods: 200, Watch: variate.Watch{MinValue: 10000}}
	m, _, _ := compoundModel().Compile(cfg)
	obs := &leafCounter{}
	cfg.Observer = obs
	w, _ := m.NewWorkspace(cfg)
	defer w.Close()
	r, _ := w.Run(context.Background(), 3)
	fmt.Printf("events %d, delivered %d, one step each: %v\n", r.Events[0], obs.rounds, obs.steps == obs.rounds)
	// Output:
	// events 1095, delivered 36, one step each: true
}

// The exact distribution of the period total, where the transform
// applies: a bracket on P(T >= x) and on every quantile.
func ExampleCompiled_Transform() {
	m, _, _ := compoundModel().Compile(variate.Config{})
	tr, err := m.Transform(variate.TransformConfig{Tol: 1e-2})
	if err != nil {
		fmt.Println(err)
		return
	}
	lo, hi := tr.Total.Quantile(0.99)
	fmt.Printf("applicable %v  grid %d  P99 within [%.0f, %.0f]\n", tr.Applicable, tr.M, lo, hi)
	// Output:
	// applicable true  grid 65536  P99 within [56382, 56437]
}

func sqrt(x float64) float64 {
	z := x
	for i := 0; i < 60; i++ {
		z = (z + x/z) / 2
	}
	return z
}

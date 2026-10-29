// Command gograph builds a model with the variate API instead of a file:
// the pattern a domain module follows to compile its own vocabulary
// into a graph. It prints the exact mean and a short simulated run.
package main

import (
	"context"
	"fmt"
	"os"
	"sort"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	g, err := scenario()
	if err != nil {
		return err
	}
	c, warnings, err := g.Compile(variate.Config{})
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	for _, w := range warnings {
		fmt.Println("warning:", w.Code, w.Subject, w.Message)
	}
	e, err := c.Analytic()
	if err != nil {
		return fmt.Errorf("analytic: %w", err)
	}
	fmt.Printf("exact mean per period: %.0f\n", e.Mean)
	w, err := c.NewWorkspace(variate.Config{Periods: 20000})
	if err != nil {
		return fmt.Errorf("workspace: %w", err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}
	var s variate.Summary
	w.Summarize(&s)
	fmt.Printf("simulated mean %.0f (stderr %.0f), p99 %.0f over %d periods\n", s.Total.Mean, s.Total.StdErr, s.Total.P99, r.Periods)
	names := make([]string, 0, len(s.ByCategory))
	// unordered: the names are sorted before use
	for name := range s.ByCategory {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		stat := s.ByCategory[name]
		fmt.Printf("  %-10s %.0f (%.0f%%)\n", name, stat.Mean, 100*stat.Share)
	}
	return nil
}

// scenario is a threat that a control blocks most of the time; what gets
// through has a minor or a major impact. A domain module would build
// this from its own types: threats, controls and impact estimates.
func scenario() (variate.Graph, error) {
	rate, err := dist.NewPERT(1, 3, 8, dist.DefaultPERTShape)
	if err != nil {
		return variate.Graph{}, fmt.Errorf("rate: %w", err)
	}
	block, err := dist.NewBeta(8, 2) // the control's effectiveness, drawn once per period
	if err != nil {
		return variate.Graph{}, fmt.Errorf("block: %w", err)
	}
	minor, err := dist.NewPERT(500, 2000, 8000, dist.DefaultPERTShape)
	if err != nil {
		return variate.Graph{}, fmt.Errorf("minor: %w", err)
	}
	major, err := dist.NewLogPERT(1e4, 5e4, 1e6, dist.DefaultPERTShape)
	if err != nil {
		return variate.Graph{}, fmt.Errorf("major: %w", err)
	}
	mostlyMinor, err := dist.NewConstant(0.8) // a fixed probability is a constant
	if err != nil {
		return variate.Graph{}, fmt.Errorf("probability: %w", err)
	}
	return variate.Graph{
		ID: "scenario",
		Sources: []variate.Source{
			{ID: "threat", Frequency: rate, To: "control"},
		},
		Nodes: []variate.Node{
			{ID: "control", Outcomes: []variate.Outcome{
				{To: "blocked", P: block},
				{To: "impact"},
			}},
			{ID: "blocked"},
			{ID: "impact", Outcomes: []variate.Outcome{
				{To: "minor", P: mostlyMinor},
				{To: "major"},
			}},
			{ID: "minor", Functions: []variate.Function{{Name: "response", Category: "response", Model: minor}}},
			{ID: "major", Functions: []variate.Function{{Name: "loss", Category: "loss", Model: major}}},
		},
	}, nil
}

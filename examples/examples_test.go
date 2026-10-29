package examples_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// TestExamplesRun loads every example model, runs a short fixed run of
// at least two thousand expected rounds and checks the simulated mean
// against the exact one.
func TestExamplesRun(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		seen++
		m, err := config.Load(e.Name())
		if err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		c, _, err := m.Graph.Compile(m.Config)
		if err != nil {
			t.Fatalf("%s: compile: %v", e.Name(), err)
		}
		ex, _ := c.Analytic()
		m.Config.Periods = max(5000, int64(2000/ex.RoundsPerPeriod))
		r, err := c.Simulate(context.Background(), m.Config)
		if err != nil {
			t.Fatalf("%s: run: %v", e.Name(), err)
		}
		if mean := r.Total.Mean(); mean < 0.8*ex.Mean || mean > 1.25*ex.Mean {
			t.Errorf("%s: simulated mean %g, exact %g", e.Name(), mean, ex.Mean)
		}
	}
	if seen != 17 {
		t.Errorf("%d example models, README lists 17", seen)
	}
}

// TestReadmeListsEveryExample: each model file appears in the README.
func TestReadmeListsEveryExample(t *testing.T) {
	readme, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	files, _ := filepath.Glob("*.yaml")
	for _, f := range files {
		if !strings.Contains(string(readme), "`"+f+"`") {
			t.Errorf("README does not list %s", f)
		}
	}
}

// TestGoGraph: the programmatic example compiles and runs.
func TestGoGraph(t *testing.T) {
	g := gographModel()
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.Simulate(context.Background(), variate.Config{Periods: 2000, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Periods != 2000 {
		t.Errorf("periods %d", r.Periods)
	}
}

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// demoModel mirrors web/models/index.json.
type demoModel struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	File  string `json:"file"`
	Knobs []struct {
		Key   string  `json:"key"`
		At    string  `json:"at"`
		Min   float64 `json:"min"`
		Max   float64 `json:"max"`
		Value float64 `json:"value"`
	} `json:"knobs"`
}

// TestDemoModels: every demo model, with its knobs at their defaults
// and at their extremes, parses and compiles, and every placeholder has
// a knob.
func TestDemoModels(t *testing.T) {
	dir := filepath.Join("..", "..", "web", "models")
	data, err := os.ReadFile(filepath.Join(dir, "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	if len(models) == 0 {
		t.Fatal("no models")
	}
	for _, m := range models {
		text, err := os.ReadFile(filepath.Join(dir, m.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		for _, which := range []string{"value", "min", "max"} {
			yaml := string(text)
			for _, k := range m.Knobs {
				v := k.Value
				switch which {
				case "min":
					v = k.Min
				case "max":
					v = k.Max
				}
				yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(v, 'g', -1, 64))
			}
			if strings.Contains(yaml, "{{") {
				t.Errorf("%s: placeholder without a knob", m.ID)
			}
			model, err := config.Parse([]byte(yaml), "")
			if err != nil {
				t.Errorf("%s at %s: %v", m.ID, which, err)
				continue
			}
			if _, _, err := model.Graph.Compile(variate.Config{AllowUnconverged: true}); err != nil {
				t.Errorf("%s at %s: compile: %v", m.ID, which, err)
			}
		}
	}
}

// TestHeavyDemoInfiniteVariance: the heavy-tails demo with the open
// Pareto's alpha at its default of 2.5 has a finite variance; at its
// lowest, 1.1, the variance is infinite and the model warns of it.
func TestHeavyDemoInfiniteVariance(t *testing.T) {
	dir := filepath.Join("..", "..", "web", "models")
	data, err := os.ReadFile(filepath.Join(dir, "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	for _, m := range models {
		if m.ID != "heavy" {
			continue
		}
		text, err := os.ReadFile(filepath.Join(dir, m.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		for _, low := range []bool{false, true} {
			yaml := string(text)
			for _, k := range m.Knobs {
				v := k.Value
				if low && k.Key == "open_alpha" {
					v = k.Min
				}
				yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(v, 'g', -1, 64))
			}
			model, err := config.Parse([]byte(yaml), "")
			if err != nil {
				t.Fatal(err)
			}
			_, ws, err := model.Graph.Compile(variate.Config{AllowUnconverged: true})
			if err != nil {
				t.Fatal(err)
			}
			warned := false
			for _, w := range ws {
				warned = warned || (w.Code == "infinite-variance" && w.Subject == "open_loss:loss")
			}
			if warned != low {
				t.Errorf("open alpha at its lowest %v: infinite-variance warning %v", low, warned)
			}
		}
		return
	}
	t.Fatal("no heavy demo")
}

// TestBacklogDemoLongRun: the backlog demo with one crew and half an
// order a month, a load of two thirds, has exact long-run figures by
// Pollaczek–Khinchine; with two crews and lognormal work it has none.
func TestBacklogDemoLongRun(t *testing.T) {
	for _, crews := range []float64{1, 2} {
		g := demoGraph(t, "backlog", map[string]float64{"crews": crews, "orders": 0.5})
		m, _, err := g.Compile(variate.Config{Repeats: 10})
		if err != nil {
			t.Fatal(err)
		}
		lr := m.QueueLongRun(0)
		if (lr.Method == "Pollaczek–Khinchine") != (crews == 1) {
			t.Errorf("crews %g: long-run method %q, reason %q", crews, lr.Method, lr.Reason)
		}
	}
}

// demoGraph is demo id's model with its knobs at their defaults but for
// those set.
func demoGraph(t *testing.T, id string, set map[string]float64) variate.Graph {
	t.Helper()
	dir := filepath.Join("..", "..", "web", "models")
	data, err := os.ReadFile(filepath.Join(dir, "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	for _, m := range models {
		if m.ID != id {
			continue
		}
		text, err := os.ReadFile(filepath.Join(dir, m.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		yaml := string(text)
		for _, k := range m.Knobs {
			v, ok := set[k.Key]
			if !ok {
				v = k.Value
			}
			yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(v, 'g', -1, 64))
		}
		model, err := config.Parse([]byte(yaml), dir)
		if err != nil {
			t.Fatal(err)
		}
		return model.Graph
	}
	t.Fatalf("no demo %s", id)
	return variate.Graph{}
}

// TestDemoAtKnobs: a knob that is a value of a quantity, not of the
// model, names a quantity the model has, and the run reports the chance
// of reaching it at the knob's default; the budget knob of the pump
// station is one.
func TestDemoAtKnobs(t *testing.T) {
	found := false
	for _, m := range demoModels(t) {
		for _, k := range m.Knobs {
			if k.At == "" {
				continue
			}
			found = found || (m.ID == "pump_station" && k.Key == "budget")
			lv, err := config.ParseLevel(k.At + "=" + strconv.FormatFloat(k.Value, 'g', -1, 64))
			if err != nil {
				t.Fatal(err)
			}
			cfg := variate.Config{Periods: 200, Seed: 1, AllowUnconverged: true, At: []variate.Level{lv}}
			c, _, err := demoGraph(t, m.ID, nil).Compile(cfg)
			if err != nil {
				t.Fatalf("%s %s: %v", m.ID, k.Key, err)
			}
			w, err := c.NewWorkspace(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Run(context.Background(), 1); err != nil {
				t.Fatal(err)
			}
			var s variate.Summary
			w.Summarize(&s)
			w.Close()
			if len(s.At) != 1 || s.At[0].N != 200 {
				t.Errorf("%s %s: at %+v, want one value counted over 200 periods", m.ID, k.Key, s.At)
			}
		}
	}
	if !found {
		t.Error("the pump station has no budget knob")
	}
}

// demoModels reads web/models/index.json.
func demoModels(t *testing.T) []demoModel {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "web", "models", "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	return models
}

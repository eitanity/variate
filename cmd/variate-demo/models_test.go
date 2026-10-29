package main

import (
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

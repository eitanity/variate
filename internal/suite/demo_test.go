package suite

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestEveryExampleHasADemo: every example model is in the browser demo
// as a model with knobs, so that a feature with an example is one the
// demo shows too.
func TestEveryExampleHasADemo(t *testing.T) {
	data, err := os.ReadFile("../../web/models/index.json")
	if err != nil {
		t.Fatal(err)
	}
	var demos []struct {
		ID, File string
		Knobs    []struct{ Key string }
	}
	if err := json.Unmarshal(data, &demos); err != nil {
		t.Fatal(err)
	}
	files := map[string]bool{}
	for _, d := range demos {
		files[d.File] = true
		if len(d.Knobs) == 0 {
			t.Errorf("demo %s has no knobs", d.ID)
		}
		if _, err := os.Stat(filepath.Join("../../web/models", d.File)); err != nil {
			t.Errorf("demo %s: %v", d.ID, err)
		}
	}
	examples, _ := filepath.Glob("../../examples/*.yaml")
	if len(examples) == 0 {
		t.Fatal("no examples found")
	}
	for _, e := range examples {
		if !files[filepath.Base(e)] {
			t.Errorf("%s has no demo model in web/models; add one with knobs to index.json", filepath.Base(e))
		}
	}
}

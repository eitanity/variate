package suite

import (
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/internal/browser"
)

// TestBrowserExactCurves: with the browser's settings, the examples
// whose exact curve needs a pass above the browser's memory at the
// engine's tolerance get one at the browser's.
func TestBrowserExactCurves(t *testing.T) {
	for _, name := range []string{"ageing", "budget", "seasonal", "threshold"} { // bound: four examples
		m, err := config.Load("../../examples/" + name + ".yaml")
		if err != nil {
			t.Fatal(err)
		}
		browser.Apply(&m.Config)
		c, _, err := m.Graph.Compile(m.Config)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		cfg := m.Config
		tr, err := c.Transform(variate.TransformConfig{TailP: cfg.TailP, MaxGrid: cfg.TransformGrid, Tol: cfg.TransformTol, MaxWork: cfg.TransformWork, MaxBytes: cfg.TransformBytes})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !tr.Applicable || tr.Bytes > browser.TransformBytes {
			t.Errorf("%s: applicable %v, %d MB at %d points: %s", name, tr.Applicable, tr.Bytes>>20, tr.M, tr.Reason)
		}
	}
}

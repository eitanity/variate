package browser_test

import (
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/browser"
)

// TestApply: one worker and the transform's browser budgets where the
// config leaves them unset, a budget the config sets kept, and what the
// run keeps left as the caller chose.
func TestApply(t *testing.T) {
	cfg := variate.Config{Workers: 8, TransformTol: 1e-4}
	browser.Apply(&cfg)
	if cfg.Workers != 1 || cfg.TransformWork != browser.TransformWork || cfg.TransformBytes != browser.TransformBytes || cfg.TransformTol != 1e-4 {
		t.Errorf("workers %d, work %d, bytes %d, tol %g", cfg.Workers, cfg.TransformWork, cfg.TransformBytes, cfg.TransformTol)
	}
	if cfg.NodeHist {
		t.Error("node histograms forced on")
	}
	unset := variate.Config{}
	browser.Apply(&unset)
	if unset.MaxMemory != browser.MaxMemory {
		t.Errorf("max memory %d, want the browser's %d", unset.MaxMemory, browser.MaxMemory)
	}
	if unset.TransformTol != browser.TransformTol {
		t.Errorf("tol %g, want the browser's %g", unset.TransformTol, browser.TransformTol)
	}
}

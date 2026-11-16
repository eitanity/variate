// Package browser holds the settings a model runs with in a browser,
// built to WebAssembly: one worker, as the page has one thread, and the
// transform's budgets and tolerance, which keep it within a browser's
// time and wasm32's memory. Variate's own browser entry applies them, as
// does any host that builds the engine into its own WebAssembly.
package browser

import "code.eitanity.com/variate"

const (
	// TransformWork is the transform's work budget unless the settings
	// say otherwise: the browser cannot interrupt a running transform,
	// so it answers within a minute or so rather than hours.
	TransformWork = 1 << 29
	// TransformTol is the transform's tolerance unless the settings say
	// otherwise. At the engine's 1e-3 a model with many sources or
	// period-scoped draws needs a pass of 2^20 grid points, above
	// TransformBytes; within 1% it closes grids earlier.
	TransformTol = 0.01
	// TransformBytes is the grids a transform pass may hold unless the
	// settings say otherwise. TinyGo doubles the wasm memory each time
	// the heap grows, after a collection leaving less than a third free,
	// and stops the engine when a doubling fails; it reserves up to three
	// times a pass's grids. Within 2 GB a doubling stays inside wasm32's
	// 4 GB, so a pass holds at most a third of that.
	TransformBytes = 640 << 20
	// MaxHeap is the most the TinyGo heap may reserve: any doubling from
	// within it stays inside wasm32's 4 GB.
	MaxHeap = 2 << 30
)

// Apply sets the browser's settings on cfg: one worker, and the
// transform's budgets and tolerance where unset. What the run keeps, such
// as the node histograms, stays the caller's choice.
func Apply(cfg *variate.Config) {
	cfg.Workers = 1
	if cfg.TransformWork == 0 {
		cfg.TransformWork = TransformWork
	}
	if cfg.TransformTol == 0 {
		cfg.TransformTol = TransformTol
	}
	if cfg.TransformBytes == 0 {
		cfg.TransformBytes = TransformBytes
	}
}

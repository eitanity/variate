// Package browser holds the settings the browser entry point applies to
// every model, shared with the check that runs the engine under TinyGo
// the way the page does.
package browser

import "code.eitanity.com/variate"

const (
	// TransformWork is the transform's work budget unless the settings
	// say otherwise: the browser cannot interrupt a running transform,
	// so it answers within a minute or so rather than hours.
	TransformWork = 1 << 29
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

// Apply sets the browser's settings on cfg: one worker, the node
// histograms the page draws, and the transform's budgets where unset.
func Apply(cfg *variate.Config) {
	cfg.Workers = 1
	cfg.NodeHist = true
	if cfg.TransformWork == 0 {
		cfg.TransformWork = TransformWork
	}
	if cfg.TransformBytes == 0 {
		cfg.TransformBytes = TransformBytes
	}
}

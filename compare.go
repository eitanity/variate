package variate

import (
	"context"
	"errors"
	"strconv"
)

// CompareConfig is what a comparison reports and how its runs go.
type CompareConfig struct {
	// Figure is the quantity compared, as a sweep's: "mean" (the
	// default), "p<percent>", "rp:<periods>", "pzero", "layer:<id>",
	// "category:<name>" or "source:<id>".
	Figure string
	// Run compiles every model and runs the simulated rows: its Seed is
	// the one seed of every run, its Periods their length (0 = the length
	// the base run's plan settles on).
	Run Config
}

// CompareRow is one option against the base: both figures by one
// method, and the option's difference from the base with its standard
// error.
type CompareRow struct {
	Method string
	// Base is the base model's figure by the row's method, Value the
	// option's.
	Base, Value Range
	// Diff is Value − Base: one point when exact or simulated, the
	// widest difference of the two brackets otherwise.
	Diff Range
	// SE is the standard error of Diff: 0 when exact, NaN for a bracket
	// or too few blocks, and for a simulated row measured on the two
	// runs' paired blocks.
	SE float64
}

// Comparison is each option against the base, in the options' order.
type Comparison struct {
	Figure  string
	Seed    uint64
	Periods int64 // of the simulated rows; 0 when there are none
	Rows    []CompareRow
}

// Compare reports cfg.Figure for each option against the graph as it
// is. A row is exact where the analytic expectations give the figure
// for both, a bracket where the transform applies to both, and simulated
// otherwise: fixed runs at the one seed and the base run's length and
// blocks, so that the option draws alike wherever it does not differ,
// with the standard error of the difference measured on the runs'
// paired blocks.
func (g Graph) Compare(ctx context.Context, options []Graph, cfg CompareConfig) (*Comparison, error) {
	fig, err := parseFigure("compare", cfg.Figure)
	if err != nil {
		return nil, err
	}
	if len(options) == 0 {
		return nil, errors.New("compare: no option to compare")
	}
	s := newSweeper(ctx, "compare", g, SweepConfig{Figure: cfg.Figure, Run: cfg.Run}, fig)
	out := &Comparison{Figure: s.figName(), Seed: cfg.Run.Seed, Rows: make([]CompareRow, len(options))}
	// bound: the options
	for i, o := range options {
		if err := ctx.Err(); err != nil {
			return nil, runError{stage: "compare", err: err}
		}
		f, err := s.figures([]Graph{o})
		if err != nil {
			return nil, errors.New("compare: option " + strconv.Itoa(i+1) + ": " + err.Error())
		}
		v := f.at[0]
		out.Rows[i] = CompareRow{Method: f.method, Base: f.base, Value: v,
			Diff: Range{Lo: v.Lo - f.base.Hi, Hi: v.Hi - f.base.Lo}, SE: f.se[0]}
		if f.method == MethodSimulation {
			out.Periods = s.base.periods
		}
	}
	return out, nil
}

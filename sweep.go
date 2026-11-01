package variate

import (
	"context"
	"errors"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// The ways a sweep row is computed, cheapest first.
const (
	MethodExact      = "exact"      // the analytic expectations
	MethodTransform  = "transform"  // the transform's bracket
	MethodSimulation = "simulation" // paired fixed runs at one seed
)

const (
	defaultSweepDelta = 0.1
	// sweepGroups is the number of groups of whole blocks over which the
	// paired difference of a simulated row is measured.
	sweepGroups = 32
)

// SweepConfig is what a sweep nudges and reports.
type SweepConfig struct {
	// Figure is the quantity reported: "mean" (the default), "p<percent>"
	// for a quantile (p99, p99.5), "rp:<periods>" for the total at a
	// return period, "pzero" for P(T = 0), "layer:<id>", "category:<name>"
	// or "source:<id>" for that mean.
	Figure string
	// Delta is the relative nudge: each input is multiplied by 1 − Delta
	// and by 1 + Delta. 0 = 0.1.
	Delta float64
	// Inputs restricts the sweep to the inputs whose name or kind is one
	// of these, or whose name starts with one followed by a space ("node
	// pump" takes every input of node pump). nil = every input.
	Inputs []string
	// Top keeps the Top inputs with the largest swing of the exact mean,
	// which ranks them before any costlier row is computed. 0 = all.
	Top int
	// Run compiles every model and runs the simulated rows: its Seed is
	// the one seed of every run, its Periods their length (0 = the length
	// the base run's plan settles on).
	Run Config
}

// Range is a figure known to lie in [Lo, Hi]: one point when it is exact
// or simulated, the transform's bracket otherwise.
type Range struct{ Lo, Hi float64 }

// Mid is the middle of the range.
func (r Range) Mid() float64 { return (r.Lo + r.Hi) / 2 }

// SweepRow is one input's row: the figure with the input scaled down, as
// it is, and scaled up.
type SweepRow struct {
	Input          Input
	Method         string
	Down, Base, Up Range
	// Swing is |Up − Down| between the middles.
	Swing float64
	// DownSE and UpSE are the standard errors of Down − Base and Up −
	// Base: 0 when exact, NaN for a bracket or too few blocks, and for a
	// simulated row measured on the runs' paired blocks.
	DownSE, UpSE float64
	// Clamped reports that a probability was held at 1, or held to its
	// node's sum, at one of the nudges.
	Clamped bool
}

// Sweep is a sensitivity sweep: rows by swing, the largest first.
type Sweep struct {
	Figure  string
	Delta   float64
	Seed    uint64
	Periods int64 // of the simulated rows; 0 when there are none
	Rows    []SweepRow
}

// sweepFigure is a parsed SweepConfig.Figure.
type sweepFigure struct {
	kind int
	p    float64 // quantile level
	name string  // layer, category or source
}

const (
	figMean = iota
	figQuantile
	figPZero
	figLayer
	figCategory
	figSource
)

func parseFigure(s string) (sweepFigure, error) {
	bad := errors.New("sweep: figure " + s + " is not mean, p<percent>, rp:<periods>, pzero, layer:<id>, category:<name> or source:<id>")
	switch {
	case s == "" || s == "mean":
		return sweepFigure{kind: figMean}, nil
	case s == "pzero":
		return sweepFigure{kind: figPZero}, nil
	case strings.HasPrefix(s, "rp:"):
		n, err := strconv.ParseFloat(s[3:], 64)
		if err != nil || !(n > 1) || math.IsInf(n, 1) {
			return sweepFigure{}, bad
		}
		return sweepFigure{kind: figQuantile, p: 1 - 1/n}, nil
	case strings.HasPrefix(s, "p"):
		v, err := strconv.ParseFloat(s[1:], 64)
		if err != nil || !(v > 0 && v < 100) {
			return sweepFigure{}, bad
		}
		return sweepFigure{kind: figQuantile, p: v / 100}, nil
	}
	for _, named := range []struct {
		prefix string
		kind   int
	}{{"layer:", figLayer}, {"category:", figCategory}, {"source:", figSource}} {
		if strings.HasPrefix(s, named.prefix) && len(s) > len(named.prefix) {
			return sweepFigure{kind: named.kind, name: s[len(named.prefix):]}, nil
		}
	}
	return sweepFigure{}, bad
}

// sweeper holds a sweep's settings and the base model's figures, each
// computed once when a row first needs it.
type sweeper struct {
	ctx   context.Context
	g     Graph
	cfg   SweepConfig
	fig   sweepFigure
	delta float64
	base  struct {
		exact, transform          *Range
		exactErr, transErr        error
		series                    *Result
		block                     int64
		periods                   int64
		seriesErr                 error
		seriesDone, transformDone bool
	}
}

// Sweep nudges each input of the graph down and up by cfg.Delta and
// reports the change in cfg.Figure. A row is exact where the analytic
// expectations give the figure, a bracket where the transform applies
// to the base and both nudged models, and simulated otherwise: fixed
// runs at the one seed, so that the nudged models draw from the base
// model's streams, with the standard error of each difference measured
// on the runs' paired blocks.
func (g Graph) Sweep(ctx context.Context, cfg SweepConfig) (*Sweep, error) {
	fig, err := parseFigure(cfg.Figure)
	if err != nil {
		return nil, err
	}
	s := &sweeper{ctx: ctx, g: g, cfg: cfg, fig: fig, delta: cfg.Delta}
	if s.delta == 0 {
		s.delta = defaultSweepDelta
	}
	if !(s.delta > 0 && s.delta < 1) {
		return nil, errors.New("sweep: delta must be in (0, 1)")
	}
	s.cfg.Run.Transform = false
	s.cfg.Run.Series = false
	s.cfg.Run.Observer = nil
	inputs := SelectInputs(g.Inputs(), cfg.Inputs)
	if len(inputs) == 0 {
		return nil, errors.New("sweep: no input matches")
	}
	if cfg.Top > 0 && cfg.Top < len(inputs) {
		if inputs, err = s.rankByMean(inputs); err != nil {
			return nil, err
		}
	}
	out := &Sweep{Figure: cfg.Figure, Delta: s.delta, Seed: cfg.Run.Seed}
	if out.Figure == "" {
		out.Figure = "mean"
	}
	for _, in := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, runError{stage: "sweep", err: err}
		}
		row, err := s.row(in)
		if err != nil {
			return nil, err
		}
		if row.Method == MethodSimulation {
			out.Periods = s.base.periods
		}
		out.Rows = append(out.Rows, row)
	}
	sort.SliceStable(out.Rows, func(a, b int) bool { return out.Rows[a].Swing > out.Rows[b].Swing })
	return out, nil
}

// SelectInputs keeps the inputs a filter names, by name, by kind, or by
// a name's leading words; no filters keep them all.
func SelectInputs(all []Input, filters []string) []Input {
	if len(filters) == 0 {
		return all
	}
	var out []Input
	for _, in := range all {
		for _, f := range filters {
			if in.Name == f || in.Kind == f || strings.HasPrefix(in.Name, f+" ") {
				out = append(out, in)
				break
			}
		}
	}
	return out
}

// rankByMean keeps the cfg.Top inputs whose exact-mean swing is largest.
func (s *sweeper) rankByMean(inputs []Input) ([]Input, error) {
	type ranked struct {
		in    Input
		swing float64
	}
	rs := make([]ranked, 0, len(inputs))
	for _, in := range inputs {
		var v [2]float64
		for i, f := range []float64{1 - s.delta, 1 + s.delta} {
			g, _, err := s.g.Scaled(in.Name, f)
			if err != nil {
				return nil, err
			}
			r, err := s.exact(g, sweepFigure{kind: figMean})
			if err != nil {
				return nil, errors.New("sweep: top needs the exact mean: " + err.Error())
			}
			v[i] = r.Lo
		}
		rs = append(rs, ranked{in: in, swing: math.Abs(v[1] - v[0])})
	}
	sort.SliceStable(rs, func(a, b int) bool { return rs[a].swing > rs[b].swing })
	out := make([]Input, s.cfg.Top)
	for i := range out {
		out[i] = rs[i].in
	}
	return out, nil
}

// row computes one input's row by the cheapest method that holds for
// the base and both nudged models.
func (s *sweeper) row(in Input) (SweepRow, error) {
	row := SweepRow{Input: in}
	var nudged [2]Graph
	for i, f := range []float64{1 - s.delta, 1 + s.delta} {
		g, clamped, err := s.g.Scaled(in.Name, f)
		if err != nil {
			return row, err
		}
		nudged[i] = g
		row.Clamped = row.Clamped || clamped
	}
	if base, err := s.baseExact(); err == nil {
		down, derr := s.exact(nudged[0], s.fig)
		up, uerr := s.exact(nudged[1], s.fig)
		if derr == nil && uerr == nil {
			row.Method, row.Base, row.Down, row.Up = MethodExact, base, down, up
			return finishRow(row), nil
		}
	}
	if base, err := s.baseTransform(); err == nil {
		down, derr := s.transform(nudged[0], s.fig)
		up, uerr := s.transform(nudged[1], s.fig)
		if derr == nil && uerr == nil {
			row.Method, row.Base, row.Down, row.Up = MethodTransform, base, down, up
			row.DownSE, row.UpSE = math.NaN(), math.NaN()
			return finishRow(row), nil
		}
	}
	return s.simulatedRow(row, nudged)
}

func finishRow(row SweepRow) SweepRow {
	row.Swing = math.Abs(row.Up.Mid() - row.Down.Mid())
	return row
}

// compile compiles one of the sweep's models.
func (s *sweeper) compile(g Graph) (*Compiled, error) {
	m, _, err := g.Compile(s.cfg.Run)
	return m, err
}

func (s *sweeper) baseExact() (Range, error) {
	if s.base.exact == nil && s.base.exactErr == nil {
		r, err := s.exact(s.g, s.fig)
		s.base.exact, s.base.exactErr = &r, err
	}
	if s.base.exactErr != nil {
		return Range{}, s.base.exactErr
	}
	return *s.base.exact, nil
}

func (s *sweeper) baseTransform() (Range, error) {
	if !s.base.transformDone {
		s.base.transformDone = true
		r, err := s.transform(s.g, s.fig)
		s.base.transform, s.base.transErr = &r, err
	}
	if s.base.transErr != nil {
		return Range{}, s.base.transErr
	}
	return *s.base.transform, nil
}

// errNoMethod is a method that does not give the figure.
var errNoMethod = errors.New("the method does not give this figure")

// exact is the figure from the analytic expectations, when they give it.
func (s *sweeper) exact(g Graph, fig sweepFigure) (Range, error) {
	if fig.kind == figQuantile || fig.kind == figLayer {
		return Range{}, errNoMethod
	}
	// The means need the model only as far as its reach.
	m, _, _, err := g.flattenChecked(s.cfg.Run)
	if err != nil {
		return Range{}, err
	}
	e, _, _, _ := m.means()
	v := math.NaN()
	switch fig.kind {
	case figMean:
		v = e.Mean
	case figPZero:
		v = m.zeroProbability()
	case figCategory:
		if c := slices.Index(m.categories, fig.name); c >= 0 {
			v = e.MeanByCategory[c]
		} else {
			return Range{}, errors.New("sweep: no category " + fig.name)
		}
	case figSource:
		if k := slices.Index(m.sourceIDs, fig.name); k >= 0 {
			v = e.MeanBySource[k]
		} else {
			return Range{}, errors.New("sweep: no source " + fig.name)
		}
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return Range{}, errNoMethod
	}
	return Range{Lo: v, Hi: v}, nil
}

// transform is the figure's bracket from the transform, when it applies.
func (s *sweeper) transform(g Graph, fig sweepFigure) (Range, error) {
	if fig.kind != figQuantile && fig.kind != figPZero && fig.kind != figLayer {
		return Range{}, errNoMethod
	}
	m, err := s.compile(g)
	if err != nil {
		return Range{}, err
	}
	cfg := s.cfg.Run
	tr, err := m.Transform(TransformConfig{TailP: m.crit.tailP, MaxGrid: cfg.TransformGrid, Tol: cfg.TransformTol, MaxWork: cfg.TransformWork, MaxBytes: cfg.TransformBytes})
	if err != nil {
		return Range{}, err
	}
	if !tr.Applicable {
		return Range{}, errors.New("transform: " + tr.Reason)
	}
	var lo, hi float64
	switch fig.kind {
	case figQuantile:
		lo, hi = tr.Total.Quantile(fig.p)
	case figPZero:
		lo, hi = tr.PZero, tr.PZero
	default:
		j := m.layerIndex(fig.name)
		if j < 0 {
			return Range{}, errors.New("sweep: no layer " + fig.name)
		}
		lo, hi = tr.Total.LayerMean(m.layers[j])
	}
	if math.IsNaN(lo) || math.IsNaN(hi) {
		return Range{}, errors.New("transform: the figure lies beyond the grid")
	}
	return Range{Lo: lo, Hi: hi}, nil
}

func (m *Compiled) layerIndex(id string) int {
	for j, y := range m.layers {
		if y.ID == id {
			return j
		}
	}
	return -1
}

// simulatedRow runs the nudged models for the base run's length at its
// seed and pairs their blocks with the base run's.
func (s *sweeper) simulatedRow(row SweepRow, nudged [2]Graph) (SweepRow, error) {
	base, err := s.baseSeries()
	if err != nil {
		return row, err
	}
	m, err := s.compile(s.g)
	if err != nil {
		return row, err
	}
	x, err := s.column(m, base)
	if err != nil {
		return row, err
	}
	bv := s.fig.of(x)
	row.Method, row.Base = MethodSimulation, Range{Lo: bv, Hi: bv}
	// The nudged runs take the base run's blocks, so that each task
	// draws from the same streams and the blocks pair.
	for i, g := range nudged {
		r, mm, _, err := s.runSeries(g, s.base.periods, s.base.block)
		if err != nil {
			return row, err
		}
		y, err := s.column(mm, r)
		if err != nil {
			return row, err
		}
		v := s.fig.of(y)
		se := s.pairedSE(x, y)
		if i == 0 {
			row.Down, row.DownSE = Range{Lo: v, Hi: v}, se
		} else {
			row.Up, row.UpSE = Range{Lo: v, Hi: v}, se
		}
	}
	return finishRow(row), nil
}

// baseSeries runs the base model once with its series kept: for
// cfg.Run.Periods, or for the length an adaptive run settles on.
func (s *sweeper) baseSeries() (*Result, error) {
	if s.base.seriesDone {
		return s.base.series, s.base.seriesErr
	}
	s.base.seriesDone = true
	periods := s.cfg.Run.Periods
	if periods == 0 {
		m, err := s.compile(s.g)
		if err != nil {
			s.base.seriesErr = err
			return nil, err
		}
		w, err := m.NewWorkspace(s.cfg.Run)
		if err != nil {
			s.base.seriesErr = err
			return nil, err
		}
		r, err := w.Run(s.ctx, s.cfg.Run.Seed)
		if err == nil {
			periods = r.Periods
		}
		w.Close()
		if err != nil {
			s.base.seriesErr = err
			return nil, err
		}
	}
	s.base.periods = periods
	s.base.series, _, s.base.block, s.base.seriesErr = s.runSeries(s.g, periods, 0)
	return s.base.series, s.base.seriesErr
}

// runSeries runs a model for a fixed length at the sweep's seed with
// its series kept, in blocks of the given length (0 = the model's own),
// and returns the block length it ran.
func (s *sweeper) runSeries(g Graph, periods, block int64) (*Result, *Compiled, int64, error) {
	m, err := s.compile(g)
	if err != nil {
		return nil, nil, 0, err
	}
	cfg := s.cfg.Run
	cfg.Periods, cfg.Series = periods, true
	if block > 0 {
		cfg.BlockPeriods = int(block)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		return nil, nil, 0, err
	}
	defer w.Close()
	r, err := w.Run(s.ctx, cfg.Seed)
	if err != nil {
		return nil, nil, 0, err
	}
	res := *r
	return &res, m, w.block, nil
}

// column is the series of the quantity the figure reads.
func (s *sweeper) column(m *Compiled, r *Result) ([]float64, error) {
	dst := make([]float64, r.Series.Periods())
	switch s.fig.kind {
	case figLayer:
		j := m.layerIndex(s.fig.name)
		if j < 0 {
			return nil, errors.New("sweep: no layer " + s.fig.name)
		}
		return r.Series.LayerColumn(j, dst), nil
	case figCategory:
		c := slices.Index(m.categories, s.fig.name)
		if c < 0 {
			return nil, errors.New("sweep: no category " + s.fig.name)
		}
		return r.Series.CategoryColumn(c, dst), nil
	case figSource:
		k := slices.Index(m.sourceIDs, s.fig.name)
		if k < 0 {
			return nil, errors.New("sweep: no source " + s.fig.name)
		}
		return r.Series.SourceColumn(k, dst), nil
	}
	return r.Series.TotalColumn(dst), nil
}

// of is the figure over a run's per-period values.
func (f sweepFigure) of(x []float64) float64 {
	n := len(x)
	if n == 0 {
		return math.NaN()
	}
	switch f.kind {
	case figQuantile:
		y := slices.Clone(x)
		slices.Sort(y)
		k := int(math.Ceil(f.p*float64(n))) - 1
		return y[min(max(k, 0), n-1)]
	case figPZero:
		zero := 0
		for _, v := range x {
			if v == 0 {
				zero++
			}
		}
		return float64(zero) / float64(n)
	}
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	return sum / float64(n)
}

// pairedSE is the standard error of fig(y) − fig(x) measured over
// groups of whole blocks: the same blocks of the two runs come from the
// same task streams, so their differences pair. NaN with fewer than 16
// groups.
func (s *sweeper) pairedSE(x, y []float64) float64 {
	block := max(s.base.block, 1)
	tasks := (int64(len(x)) + block - 1) / block
	groups := min(tasks, sweepGroups)
	if groups < minBlocks {
		return math.NaN()
	}
	per := tasks / groups
	d := make([]float64, groups)
	for g := range groups {
		lo := int64(g) * per * block
		hi := min(lo+per*block, int64(len(x)))
		if g == groups-1 {
			hi = int64(len(x))
		}
		d[g] = s.fig.of(y[lo:hi]) - s.fig.of(x[lo:hi])
	}
	mean := 0.0
	for _, v := range d {
		mean += v
	}
	mean /= float64(groups)
	ss := 0.0
	for _, v := range d {
		ss += (v - mean) * (v - mean)
	}
	return math.Sqrt(ss / float64(groups-1) / float64(groups))
}

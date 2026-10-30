package variate

import (
	"math"
)

// TransformConfig controls the FFT transform.
type TransformConfig struct {
	MaxGrid  int     // default 1<<22 points
	Tol      float64 // default 1e-3: max of (Upper − Lower)/Upper over x up to x*
	TailP    float64 // default from the compile criteria
	MaxWork  int64   // default 1<<34 grid-point evaluations (a rate quadrature's nodes × points, a function's or closed-form rate's points) over the whole transform
	MaxBytes int64   // default 2<<30: the grids a pass holds at once
}

// Transformed is the exact distribution of the period total inside a
// bracket, or the reason the method does not apply.
type Transformed struct {
	Applicable bool
	Reason     string
	H          float64
	M          int
	Theta      float64
	PZero      float64
	QuadNodes  int
	Passes     int
	Work       int64 // grid-point evaluations spent
	Total      Bounds
	ByCategory []Bounds
	BySource   []Bounds // nil when the model has more than maxSourceBounds sources
}

// Bounds brackets P(T ≥ x) at x = 0, h, 2h, …, with the parts of the
// bracket reported separately.
type Bounds struct {
	H            float64
	Lower, Upper []float64
	DiscGap      []float64
	Alias        float64
	Roundoff     []float64
}

const (
	defaultMaxGrid   = 1 << 22
	defaultTol       = 1e-3
	defaultMaxWork   = 1 << 34
	defaultMaxBytes  = 2 << 30
	bytesPerPoint    = 16 // a complex grid point
	minGrid          = 1 << 16
	maxFolds         = 64
	roundoffSafety   = 8
	unitRoundoff     = 0x1p-53
	tiltMax          = 50 // θ·M·h
	goldenIterations = 40
	quadStart        = 16
	quadDoublings    = 12
	quadMaxNodes     = quadStart << quadDoublings
	quadSample       = 256
	quadWorkCap      = 1 << 31 // λ-quadrature nodes × grid points per evaluation
	quadTol          = 1e-14
	maxPasses        = 1 << 12
	maxAtomScan      = 1 << 22 // lattice steps scanned for a period-scoped value's atoms
	gcdSteps         = 64
	minAtom          = 1e-9 // mass at an upper end that makes it an atom; a continuous end has density × ulp
	maxPeriodScoped  = 3
	maxSourceBounds  = 16 // sources beyond which no per-source bounds are kept
)

// Quantile brackets the p-quantile of T: the smallest lattice points at
// which the upper and lower curves fall to 1 − p.
func (b Bounds) Quantile(p float64) (lo, hi float64) {
	target := 1 - p
	lo, hi = math.NaN(), math.NaN()
	for k := range b.Lower {
		if math.IsNaN(hi) && b.Upper[k] <= target {
			hi = float64(k) * b.H
		}
		if math.IsNaN(lo) && b.Lower[k] <= target {
			lo = float64(k) * b.H
		}
	}
	if math.IsNaN(lo) {
		lo = float64(len(b.Lower)) * b.H
	}
	if math.IsNaN(hi) {
		hi = float64(len(b.Upper)) * b.H
	}
	// The lower curve lies below the upper, so it drops to the target first.
	return lo, hi
}

// Transform computes the distribution of the period total by FFT with a
// bracket, for models without delays.
func (m *Compiled) Transform(cfg TransformConfig) (*Transformed, error) {
	t := newTransformer(m, cfg)
	out := &Transformed{}
	if reason := t.applicable(); reason != "" {
		out.Reason = reason
		return out, nil
	}
	if err := t.run(out); err != nil {
		return nil, err
	}
	out.Work = t.work
	return out, nil
}

// transformer holds one Transform's working state.
type transformer struct {
	m        *Compiled
	e        *Expectation
	cfg      TransformConfig
	tailP    float64
	lam      []float64 // E[λ_k]
	draws    []float64 // expected function draws per period, by function
	unit     float64   // a value every constant function is a multiple of; 0 if none
	quadN    int
	passes   int
	lastEps  float64 // the last quadrature's difference between its final two node counts
	order    []int32 // nodes in the order the recursion computes them
	uses     []int32 // outcomes and sources leading to each node
	work     int64   // grid-point evaluations so far
	grid     int     // the grid being evaluated
	passWork int64   // evaluations of the last pass
	passM    int     // its grid
}

func newTransformer(m *Compiled, cfg TransformConfig) *transformer {
	if cfg.MaxGrid <= 0 {
		cfg.MaxGrid = defaultMaxGrid
	}
	if cfg.MaxGrid < minGrid {
		cfg.MaxGrid = minGrid
	}
	if cfg.Tol <= 0 {
		cfg.Tol = defaultTol
	}
	if cfg.MaxWork <= 0 {
		cfg.MaxWork = defaultMaxWork
	}
	if cfg.MaxBytes <= 0 {
		cfg.MaxBytes = defaultMaxBytes
	}
	t := &transformer{m: m, e: m.expect, cfg: cfg, tailP: cfg.TailP}
	if t.tailP <= 0 || t.tailP >= 1 {
		t.tailP = m.crit.tailP
	}
	t.lam = m.countMean
	t.draws = make([]float64, len(m.fns))
	for f := range m.fns {
		t.draws[f] = t.e.EntriesPer[m.fnNode[f]]
	}
	t.unit = m.atomUnit()
	t.order, t.uses = m.transformOrder()
	return t
}

// atomUnit returns the largest value every function atom is an integer
// multiple of, found by Euclid's algorithm with a relative tolerance, or
// 0 when there are no atoms or no such value above 1e-6 of the smallest.
// The atoms are the positive constants and the upper ends that hold more
// than minAtom of mass, as a cap does and a bounded continuous value's
// end does not. A span that divides the unit puts every atom on the
// lattice, so the two roundings agree there and the bracket can close.
func (m *Compiled) atomUnit() float64 {
	unit := 0.0
	smallest := 0.0
	for _, f := range m.fns {
		d := m.pools[f.pool]
		lo, hi := d.Support()
		atom := 0.0
		switch {
		case lo == hi && lo > 0:
			atom = lo
		case hi > 0 && !math.IsInf(hi, 1) && d.Survival(math.Nextafter(hi, 0)) > minAtom:
			atom = hi
		default:
			continue
		}
		if smallest == 0 || atom < smallest {
			smallest = atom
		}
		unit = gcdFloat(unit, atom)
	}
	if unit < 1e-6*smallest {
		return 0
	}
	return unit
}

// LatticeUnit is the value every period total is a multiple of when each
// function is a constant booked on entry, or 0 when the total can take
// values between multiples.
func (m *Compiled) LatticeUnit() float64 {
	for f, fn := range m.fns {
		lo, hi := m.pools[fn.pool].Support()
		if lo != hi || m.fnPerTime[f] {
			return 0
		}
	}
	return m.atomUnit()
}

// gcdFloat is the greatest common divisor of two non-negative values to
// a relative tolerance of 1e-9; gcd(0, b) is b.
func gcdFloat(a, b float64) float64 {
	if a < b {
		a, b = b, a
	}
	tol := 1e-9 * a
	// bound: gcdSteps
	for i := 0; i < gcdSteps && b > tol; i++ {
		a, b = b, math.Mod(a, b)
		if b < tol || a-b < tol {
			b = 0
		}
	}
	return a
}

// snap lowers a span to the largest unit/2^j not above it.
func (t *transformer) snap(h float64) float64 {
	if t.unit == 0 || h <= 0 {
		return h
	}
	j := math.Ceil(math.Log2(t.unit / h))
	if j < 0 {
		j = 0
	}
	return t.unit / math.Pow(2, j)
}

// applicable returns the reason the method does not apply, or "".
func (t *transformer) applicable() string {
	m := t.m
	if m.hasDelays {
		return "the model has delays"
	}
	if m.hasRepair {
		return "the model has a source with repair or a reliability diagram"
	}
	scoped := 0
	for k, slot := range m.poolPeriod {
		if slot == noIndex || t.isFrequency(index(k)) {
			continue
		}
		scoped++
	}
	if scoped > maxPeriodScoped {
		return "more than " + itoa(maxPeriodScoped) + " period-scoped quantities besides the frequencies"
	}
	if math.IsInf(t.e.Mean, 0) || math.IsNaN(t.e.Mean) {
		return "the mean is not finite"
	}
	return ""
}

func (t *transformer) isFrequency(pool int32) bool {
	for _, s := range t.m.srcs {
		if s.pool == pool {
			return true
		}
	}
	return false
}

// run performs the two passes and fills the result.
func (t *transformer) run(out *Transformed) error {
	c := math.Sqrt(t.tailP / (1 - t.tailP))
	// First pass: the span comes from the closed-form Cantelli (or Markov)
	// solve at each grid size.
	first, reason := t.converge(minGrid, func(m int) (float64, bool) { return t.firstSpan(c, m) }, 0)
	if first == nil {
		out.Reason = reason
		return nil
	}
	_, xStar := first.bounds(first.total).Quantile(t.tailP)
	h1 := first.h
	if xStar < 2*h1 {
		// Nothing to resolve: the TailP quantile sits at the bottom of the
		// grid (a rate so low that P(T = 0) exceeds TailP). A finer lattice
		// would shrink without bound and fold every value beyond it.
		first.fill(out)
		out.Applicable = true
		out.QuadNodes = t.quadN
		out.Passes = t.passes
		return nil
	}
	// Second pass: a nested finer lattice, h₂ = h₁ / 2^j, the finest that
	// still holds 2·x* at each grid size, from the size the first pass needed.
	second, reason := t.converge(first.m, func(m int) (float64, bool) {
		h := h1
		// bound: halvings until the grid just holds 2·x*
		for h/2*float64(m) >= 2*xStar && h/2 > 0 {
			h /= 2
		}
		return h, true
	}, xStar)
	if second == nil {
		out.Reason = reason
		return nil
	}
	second.fill(out)
	out.Applicable = true
	out.QuadNodes = t.quadN
	out.Passes = t.passes
	return nil
}

// passBytes estimates the grids a pass holds at m points: per
// pipeline the total, a grid per category, a grid per source when
// their bounds are kept, the scratch, the root table and the FFT plan,
// then the brackets as floats.
func (t *transformer) passBytes(m int) int64 {
	srcs := len(t.m.srcs)
	if srcs > maxSourceBounds {
		srcs = 0
	}
	grids := int64(1+len(t.m.categories)+srcs) + 8
	return 2 * int64(m) * (bytesPerPoint*grids + 8*int64(2*(1+len(t.m.categories)+srcs)))
}

// bytesReason explains a refusal by the memory budget.
func (t *transformer) bytesReason(m int, b int64) string {
	return "the transform would hold " + itoa(int(b>>20)) + " MB at " + itoa(m) + " grid points, above " + itoa(int(t.cfg.MaxBytes>>20)) + " MB"
}

// budgetReason explains a refusal by the work budget.
func (t *transformer) budgetReason(m int) string {
	return "the work budget of " + i64toa(t.cfg.MaxWork) + " grid-point evaluations would be exceeded at " + itoa(m) + " grid points"
}

// firstSpan solves M·h = 2·x*(h) with x* from Cantelli on the lattice
// moments; ok is false when the grid cannot hold the draws.
func (t *transformer) firstSpan(c float64, m int) (float64, bool) {
	e := t.e
	num := 2 * (e.Mean + c*math.Sqrt(e.Var))
	draws := 2 * (e.DrawsMean + c*math.Sqrt(e.DrawsVar))
	if math.IsInf(e.Var, 1) || math.IsNaN(e.Var) {
		// Markov: x*(h) = (E[T] + h·E[D])/(1 − TailP).
		num = 2 * e.Mean / (1 - t.tailP)
		draws = 2 * e.DrawsMean / (1 - t.tailP)
	}
	den := float64(m) - draws
	if den <= 0 {
		return 0, false
	}
	return t.snap(num / den), true
}

// converge runs the pipelines at grid sizes doubling from minGrid, with
// the span the pass's rule gives for each size, until the bracket is
// within Tol up to the deepest point that must be resolved: the pass's own
// upper TailP quantile, or x* when it is known and smaller. nil and a
// reason otherwise.
func (t *transformer) converge(start int, span func(m int) (float64, bool), xStar float64) (*passResult, string) {
	var last *passResult
	var lastX float64
	// bound: doublings up to MaxGrid
	for m := start; m <= t.cfg.MaxGrid; m *= 2 {
		h, ok := span(m)
		if !ok {
			continue
		}
		if t.passM > 0 && t.work+t.passWork*int64(m/t.passM) > t.cfg.MaxWork {
			// The next pass costs the last one's work scaled by the grid.
			return nil, t.budgetReason(m)
		}
		if b := t.passBytes(m); b > t.cfg.MaxBytes {
			return nil, t.bytesReason(m, b)
		}
		r, err := t.pass(h, m, xStar)
		if err != "" {
			return nil, err
		}
		last = r
		_, lastX = r.bounds(r.total).Quantile(t.tailP)
		if xStar > 0 {
			lastX = math.Min(lastX, xStar)
		}
		if r.withinTol(lastX, t.cfg.Tol, t.tailP) {
			return r, ""
		}
	}
	if last == nil {
		return nil, "too many draws per period for the grid"
	}
	return nil, last.dominant(lastX)
}

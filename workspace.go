package variate

import (
	"context"
	"errors"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/stats"
)

const (
	defaultBlockPeriods = 4096
	defaultPoolSize     = 1024
	defaultHistRes      = 1e-3
	defaultBatch        = 64
	defaultMaxMemory    = 4 << 30
	// maxTasks keeps the stream key injective.
	maxTasks = 1 << 32
)

// runError wraps an error that ended a run with the stage it came from.
type runError struct {
	stage string
	err   error
}

func (e runError) Error() string { return e.stage + ": " + e.err.Error() }

func (e runError) Unwrap() error { return e.err }

// Workspace holds every buffer a run needs, allocated once, and the
// persistent worker pool.
type Workspace struct {
	m           *Compiled
	cfg         Config
	l           layout
	s           int
	block       int64
	batch       int64
	horizon     int64
	workers     []*worker
	slots       []*slot
	carry       []float64 // S rows of spill from the last finalised task
	carryAt     int64     // first period the carry buffer holds
	result      Result
	hists       []*stats.LogHist // global, by row quantity; nil without a histogram
	delays      bool
	hasFn       []bool
	sched       sched
	wg          sync.WaitGroup
	closed      bool
	footprint   int64     // the planned footprint in bytes
	means       []float64 // analytic mean per row quantity, for histogram ranges
	rows        []PlanRow // the plan with observed values
	unmet       []string  // reused by evaluate
	labels      []string  // criterion and driver per row, built once
	planBinding string
	lastUnmet   string
	adaptive    bool
	// Round stream.
	stream  bool
	recSize int
	round   Round  // reused for every delivery
	fileRec []byte // one record read back from a slot file
	seq     uint64
	clock   clock
	start   time.Time
	stopped bool
}

// NewWorkspace allocates everything a run needs, sized from the model and
// the configuration, and starts the worker pool.
func (m *Compiled) NewWorkspace(cfg Config) (*Workspace, error) {
	if cfg.Periods < 0 {
		return nil, errors.New("config: Periods must be >= 0")
	}
	if cfg.Periods == 0 && cfg.Series {
		return nil, errors.New("config: Series needs a fixed Periods")
	}
	w := &Workspace{m: m, cfg: cfg, l: newLayout(m), s: m.spill, delays: m.hasDelays || m.renewal}
	w.rows = m.Plan()
	w.labels = make([]string, len(w.rows))
	for i, r := range w.rows {
		w.labels[i] = r.Criterion + " " + r.Driver
	}
	crit, driver := m.Binding()
	w.planBinding = "plan: " + crit + " " + driver
	w.unmet = make([]string, 0, len(w.rows))
	w.result.Unmet = make([]string, 0, len(w.rows))
	w.block = blockPeriods(cfg, int64(w.s))
	horizon := cfg.Periods
	if horizon == 0 {
		horizon = m.crit.maxPeriods
	}
	if (horizon+w.block-1)/w.block >= maxTasks {
		return nil, errors.New("config: more than 2^32 tasks; raise BlockPeriods or lower MaxPeriods")
	}
	w.batch = int64(cfg.Batch)
	if w.batch <= 0 {
		w.batch = defaultBatch
	}
	nw := cfg.Workers
	if nw <= 0 {
		nw = runtime.GOMAXPROCS(0)
	}
	pool := cfg.PoolSize
	if pool <= 0 {
		pool = defaultPoolSize
	}
	if err := w.allocResult(); err != nil {
		return nil, err
	}
	nslots := int(min(int64(2*nw+2), w.batch))
	if err := w.checkFootprint(nw, nslots, pool); err != nil {
		return nil, err
	}
	w.carry = make([]float64, w.s*w.l.width)
	if cfg.Series {
		w.result.Series = &Series{l: w.l, rows: make([]float64, int(cfg.Periods)*w.l.width), n: cfg.Periods}
	}
	// bound: workers or slots, fixed at construction
	// bound: slots
	for i := 0; i < nslots; i++ {
		w.slots = append(w.slots, newSlot(w.l, w.s, len(m.srcs), len(m.nodes), len(m.outs), len(m.tailThresholds)))
	}
	if err := w.setupStream(nw); err != nil {
		return nil, err
	}
	w.startPool(nw, pool)
	return w, nil
}

// setupStream sizes the round-stream buffers and opens one temporary file
// per slot when an Observer is set.
func (w *Workspace) setupStream(nw int) error {
	w.clock = realClock{}
	if w.cfg.Observer == nil {
		return nil
	}
	m := w.m
	w.stream = true
	w.recSize = recordSize(m.maxPath)
	w.round.Steps = make([]Step, 0, m.maxPath)
	w.fileRec = make([]byte, w.recSize)
	buf := int64(w.cfg.RecordBuffer)
	if buf <= 0 {
		perBlock := m.expectedMatches() * float64(w.block)
		buf = min(int64(perBlock*float64(w.recSize)*recordBufferSlack)+int64(w.recSize), maxRecordBuffer)
	}
	for _, sl := range w.slots {
		sl.recBuf = make([]byte, buf)
		f, err := os.CreateTemp("", "variate-rounds-*")
		if err != nil {
			return runError{stage: recordFileStage, err: err}
		}
		sl.recFile = f
	}
	_ = nw
	return nil
}

// startPool allocates the workers' private state and starts the worker
// and finaliser goroutines, the only goroutines the engine creates.
func (w *Workspace) startPool(nw, pool int) {
	m := w.m
	res := w.histRes()
	// bound: workers or slots, fixed at construction
	for i := 0; i < nw; i++ {
		wk := &worker{ss: dist.NewSamplerSet(m.pools, pool), period: make([]float64, m.periodSlots)}
		if m.renewal {
			wk.phase = make([]int32, len(m.srcs))
			wk.downUntil = make([]float64, len(m.srcs))
			wk.failed = make([]int32, len(m.chains))
			wk.chainT = make([]float64, len(m.srcs))
		}
		if w.stream {
			scratch := w.cfg.RoundScratch
			if scratch <= 0 {
				scratch = defaultRoundScratch
			}
			wk.cur.Steps = make([]Step, 0, m.maxPath)
			wk.scratch = make([]byte, scratch*w.recSize)
			wk.idx = make([]int32, scratch)
			wk.tmp = make([]int32, scratch)
			wk.keys = make([]float64, scratch)
		}
		for par := 0; par < 2; par++ {
			wk.hists[par] = make([]*stats.LogHist, w.l.width)
			for q, h := range w.hists {
				if h != nil {
					wk.hists[par][q] = w.newHistLike(q, res)
				}
			}
		}
		w.workers = append(w.workers, wk)
	}
	w.sched.cv = sync.NewCond(&w.sched.mu)
	w.sched.slotState = make([]int8, len(w.slots))
	w.sched.slotTask = make([]int64, len(w.slots))
	w.sched.ctx = context.Background()
	w.wg.Add(nw + 1)
	for _, wk := range w.workers {
		go func(wk *worker) {
			defer w.wg.Done()
			w.workerLoop(wk)
		}(wk)
	}
	go func() {
		defer w.wg.Done()
		w.finaliserLoop()
	}()
}

func (w *Workspace) histRes() float64 {
	if w.cfg.HistRes > 0 {
		return w.cfg.HistRes
	}
	return defaultHistRes
}

// newHistLike allocates a worker histogram with the layout of global
// quantity q.
func (w *Workspace) newHistLike(q int, res float64) *stats.LogHist {
	l := w.l
	if q >= 1+l.cats+l.srcs && q < 1+l.cats+l.srcs+l.nodes {
		return stats.NewIntHist(w.means[q], res)
	}
	return stats.NewLogHist(w.means[q], res)
}

// allocLayers gives each layer its accumulator and histogram, ranged
// from a scale the payouts are near: the share of the total's mean
// within the limit.
func (w *Workspace) allocLayers(e *Expectation, res float64) {
	r := &w.result
	r.Layers = make([]Accum, len(w.m.layers))
	for j, y := range w.m.layers {
		q := w.l.layer(j)
		scale := y.Share * math.Min(math.Max(e.Mean, 1e-9), y.Limit)
		w.means[q] = scale
		r.Layers[j].Hist = stats.NewLogHist(scale, res)
		w.hists[q] = r.Layers[j].Hist
	}
}

// checkFootprint evaluates the memory the workspace would hold and
// refuses above MaxMemory.
func (w *Workspace) checkFootprint(nw, nslots, pool int) error {
	limit := w.cfg.MaxMemory
	if limit <= 0 {
		limit = defaultMaxMemory
	}
	a := int64(w.l.width)
	bins := int64(0)
	for _, h := range w.hists {
		if h != nil {
			bins += int64(h.Bins()) + exactBucketsOf(h)
		}
	}
	// Each slot also counts events, reach and outcomes per source, and
	// the rows summed per tail threshold.
	counters := int64(len(w.m.srcs))*int64(1+len(w.m.nodes)+len(w.m.outs)) + int64(len(w.m.tailThresholds))*int64(w.l.tailWidth()+1)
	slotsB := int64(nslots) * (8*a*int64(2*w.s+1) + 64*a + 8*counters)
	carryB := 8 * a * int64(w.s)
	globalB := 64*a + 8*bins
	workerHistB := int64(2*nw) * 8 * bins
	poolsB := int64(nw) * int64(len(w.m.pools)) * int64(pool) * 8
	seriesB := int64(0)
	if w.cfg.Series {
		seriesB = 8 * a * w.cfg.Periods
	}
	total := slotsB + carryB + globalB + workerHistB + poolsB + seriesB
	w.footprint = total
	if total > limit {
		return errors.New("workspace footprint " + itoa(int(total>>20)) + " MB exceeds MaxMemory " + itoa(int(limit>>20)) +
			" MB: slots " + itoa(int(slotsB>>20)) + ", histograms " + itoa(int((globalB+workerHistB)>>20)) +
			", pools " + itoa(int(poolsB>>20)) + ", series " + itoa(int(seriesB>>20)) + " MB")
	}
	return nil
}

func exactBucketsOf(h *stats.LogHist) int64 {
	if lo, _ := h.Range(); lo == exactBuckets {
		return exactBuckets
	}
	return 0
}

// exactBuckets mirrors the integer histogram's exact bucket count.
const exactBuckets = 4096

// allocResult sizes the accumulators and histograms from the analytic
// means.
func (w *Workspace) allocResult() error {
	m := w.m
	e, err := m.Analytic()
	if err != nil {
		return err
	}
	res := w.histRes()
	r := &w.result
	r.Nodes, r.Outcomes, r.Categories, r.Sources = m.nodeIDs, m.outcomeNames, m.categories, m.sourceIDs
	r.Events = make([]uint64, len(m.srcs))
	r.Reach = make([][]uint64, len(m.srcs))
	r.OutcomeCount = make([][]uint64, len(m.srcs))
	for k := range m.srcs {
		r.Reach[k] = make([]uint64, len(m.nodes))
		r.OutcomeCount[k] = make([]uint64, len(m.outs))
	}
	w.hists = make([]*stats.LogHist, w.l.width)
	w.means = make([]float64, w.l.width)
	w.means[w.l.total()] = e.Mean
	r.Total.Hist = stats.NewLogHist(e.Mean, res)
	w.hists[w.l.total()] = r.Total.Hist
	r.ByCategory = make([]Accum, len(m.categories))
	for c := range r.ByCategory {
		q := w.l.cat(index(c))
		w.means[q] = e.MeanByCategory[c]
		r.ByCategory[c].Hist = stats.NewLogHist(e.MeanByCategory[c], res)
		w.hists[q] = r.ByCategory[c].Hist
	}
	r.BySource = make([]Accum, len(m.srcs))
	for k := range r.BySource {
		q := w.l.src(index(k))
		w.means[q] = e.MeanBySource[k]
		r.BySource[k].Hist = stats.NewLogHist(e.MeanBySource[k], res)
		w.hists[q] = r.BySource[k].Hist
	}
	r.EntriesPer = make([]Accum, len(m.nodes))
	r.TailCounts = make([]int64, len(m.tailThresholds))
	r.TailSums = make([][]float64, len(m.tailThresholds))
	for j := range r.TailSums {
		r.TailSums[j] = make([]float64, w.l.tailWidth())
	}
	w.allocLayers(e, res)
	r.Contrib = make([]Accum, len(m.nodes))
	w.hasFn = make([]bool, len(m.nodes))
	for v := range m.nodes {
		w.hasFn[v] = m.nodes[v].fnStart != m.nodes[v].rateEnd
		w.means[w.l.entry(index(v))] = e.EntriesPer[v]
		w.means[w.l.contrib(index(v))] = e.Contrib[v]
		if w.cfg.NodeHist {
			r.EntriesPer[v].Hist = stats.NewIntHist(e.EntriesPer[v], res)
			w.hists[w.l.entry(index(v))] = r.EntriesPer[v].Hist
			if w.hasFn[v] {
				r.Contrib[v].Hist = stats.NewLogHist(e.Contrib[v], res)
				w.hists[w.l.contrib(index(v))] = r.Contrib[v].Hist
			}
		}
	}
	return nil
}

// accum returns the Accum of row quantity i.
func (w *Workspace) accum(i int) *Accum {
	l := w.l
	switch {
	case i == 0:
		return &w.result.Total
	case i < 1+l.cats:
		return &w.result.ByCategory[i-1]
	case i < 1+l.cats+l.srcs:
		return &w.result.BySource[i-1-l.cats]
	case i < 1+l.cats+l.srcs+l.nodes:
		return &w.result.EntriesPer[i-1-l.cats-l.srcs]
	case i < 1+l.cats+l.srcs+2*l.nodes:
		return &w.result.Contrib[i-1-l.cats-l.srcs-l.nodes]
	default:
		return &w.result.Layers[i-1-l.cats-l.srcs-2*l.nodes]
	}
}

// resetResult zeroes the result before a run without reallocating.
func (w *Workspace) resetResult(seed uint64) {
	r := &w.result
	r.Seed = seed
	r.Periods = 0
	r.Converged, r.Stopped = false, false
	r.Steps, r.BeyondHorizon, r.TailPeriods = 0, 0, 0
	clear(r.Events)
	for k := range r.Reach {
		clear(r.Reach[k])
		clear(r.OutcomeCount[k])
	}
	// bound: row width
	for i := 0; i < w.l.width; i++ {
		w.accum(i).reset()
	}
	for _, wk := range w.workers {
		for par := 0; par < 2; par++ {
			for _, h := range wk.hists[par] {
				if h != nil {
					h.Reset()
				}
			}
		}
	}
	clear(w.carry)
	w.carryAt = 0
	if r.Series != nil {
		clear(r.Series.rows)
	}
	for i := range w.rows {
		w.rows[i].Observed, w.rows[i].Met = 0, false
	}
	w.lastUnmet = ""
	r.Unmet = r.Unmet[:0]
	r.Binding = ""
}

// Run simulates a fixed number of periods on the worker pool and returns
// the workspace's Result, which the next Run overwrites.
func (w *Workspace) Run(ctx context.Context, seed uint64) (*Result, error) {
	if w.closed {
		return nil, errors.New("run: workspace is closed")
	}
	w.resetResult(seed)
	w.adaptive = w.cfg.Periods == 0
	w.seq = 0
	w.stopped = false
	w.start = w.clock.Now()
	s := &w.sched
	s.mu.Lock()
	if w.adaptive {
		s.tasks = (w.m.crit.maxPeriods + w.block - 1) / w.block
		w.horizon = s.tasks * w.block
	} else {
		w.horizon = w.cfg.Periods
		s.tasks = (w.horizon + w.block - 1) / w.block
	}
	clear(s.slotState)
	s.running = 0
	s.active, s.done, s.stop, s.err = true, false, false, nil
	s.next, s.finalNext = 0, 0
	s.seed = seed
	s.ctx = ctx
	s.cv.Broadcast()
	// bound: wakes until the finaliser reports the run done
	for !s.done {
		s.cv.Wait()
		if ctx.Err() != nil && !s.stop {
			s.stop = true
			s.cv.Broadcast()
		}
	}
	s.active = false
	err := s.err
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	if cerr := ctx.Err(); cerr != nil {
		return nil, runError{stage: "run", err: cerr}
	}
	return &w.result, nil
}

// Close stops the worker pool. The workspace cannot be used afterwards.
func (w *Workspace) Close() {
	if w.closed {
		return
	}
	w.closed = true
	w.sched.mu.Lock()
	w.sched.closing = true
	w.sched.cv.Broadcast()
	w.sched.mu.Unlock()
	w.wg.Wait()
	for _, sl := range w.slots {
		if sl.recFile != nil {
			name := sl.recFile.Name()
			_ = sl.recFile.Close() // the file is being discarded
			if err := os.Remove(name); err != nil {
				continue
			}
		}
	}
}

// blockPeriods is the periods per task: the setting, else the default,
// which a fixed run lowers so that it has at least twice minBlocks
// blocks where it can, since a model with delays estimates its error
// from them; never below the spill.
func blockPeriods(cfg Config, spill int64) int64 {
	block := max(int64(cfg.BlockPeriods), 0)
	if block == 0 {
		block = defaultBlockPeriods
		if cfg.Periods > 0 {
			block = min(block, max(cfg.Periods/(2*minBlocks), 1))
		}
	}
	return max(block, spill, 1)
}

// Bands fills dst[i] with P(edges[i] ≤ T < edges[i+1]) for the period
// total; dst must have len(edges)−1 entries.
func (w *Workspace) Bands(edges, dst []float64) []float64 {
	w.result.Total.Hist.Bands(edges, dst)
	return dst
}

// Means returns the analytic means the histogram ranges were sized from,
// by row quantity; for diagnostics.
func (w *Workspace) Means() []float64 { return w.means }

var _ = math.Inf

// Simulate is the one-shot form: a workspace for cfg, one run with
// cfg.Seed, and the workspace closed. The Result stays valid.
func (m *Compiled) Simulate(ctx context.Context, cfg Config) (*Result, error) {
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		return nil, err
	}
	defer w.Close()
	return w.Run(ctx, cfg.Seed)
}

// Footprint is the planned size of the workspace's buffers in bytes, the
// figure checked against MaxMemory.
func (w *Workspace) Footprint() int64 { return w.footprint }

// NodeHist reports whether the per-node histograms are kept.
func (w *Workspace) NodeHist() bool { return w.cfg.NodeHist }

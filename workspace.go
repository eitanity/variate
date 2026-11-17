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
	// maxSlotValues bounds a slot's rows under the default block.
	maxSlotValues    = 1 << 19
	defaultPoolSize  = 1024
	defaultHistRes   = 1e-3
	defaultBatch     = 64
	defaultMaxMemory = 4 << 30
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
	burnIn      int64 // burn-in tasks, before task 0, over every timeline
	lines       int64 // independent timelines the tasks are dealt into
	block       int64
	batch       int64
	horizon     int64
	started     bool // a started run: a task is one repeat of the horizon, its block
	workers     []*worker
	slots       []*slot
	stores      []pending // per timeline, what its finished tasks booked past the blocks handed so far
	result      Result
	hists       []*stats.LogHist // global, by row quantity; nil without a histogram
	delays      bool
	hasFn       []bool
	sched       sched
	wg          sync.WaitGroup
	footprint   int64     // the planned footprint in bytes
	means       []float64 // analytic mean per row quantity, for histogram ranges
	rows        []PlanRow // the plan with observed values
	unmet       []string  // reused by evaluate
	labels      []string  // criterion and driver per row, built once
	planBinding string
	// Config.When: the half-lines of its conditions, a slot's sums of the
	// value ones, the periods merged in all and per batch group, the
	// quantities each condition reports, and scratch sums.
	halves        []half
	slotWhen      int
	anyOneIn      bool
	whenN         int64
	whenGN        []int64
	whenPartsTmpl []WhenPart
	whenScratch   condSums
	lastUnmet     string
	adaptive      bool
	// Round stream.
	stream  bool
	recSize int
	round   Round  // reused for every delivery
	fileRec []byte // one record read back from a slot file
	seq     uint64
	repeat  int64 // the repeat whose rounds are being delivered, in a started run
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
	if err := checkRunConfig(m, cfg); err != nil {
		return nil, err
	}
	w := &Workspace{m: m, cfg: cfg, l: newLayout(m), delays: (m.hasDelays || m.renewal) && !m.Started(), started: m.Started()}
	w.rows = m.Plan()
	w.labels = make([]string, len(w.rows))
	for i, r := range w.rows {
		w.labels[i] = r.Criterion + " " + r.Driver
	}
	crit, driver := m.Binding()
	w.planBinding = "plan: " + crit + " " + driver
	w.unmet = make([]string, 0, len(w.rows))
	w.result.Unmet = make([]string, 0, len(w.rows))
	w.block = blockPeriods(cfg, w.l.width)
	horizon := cfg.Periods
	if horizon == 0 {
		horizon = m.crit.maxPeriods
	}
	tasks := (horizon + w.block - 1) / w.block
	w.lines = timelines(m, cfg, tasks)
	if w.started {
		w.block, w.lines = m.horizon, 1
		tasks, _ = w.runLength()
	}
	w.burnIn = w.lines * ((m.burnIn + w.block - 1) / w.block)
	if tasks+w.burnIn >= maxTasks {
		return nil, errors.New("config: more than 2^32 tasks with the burn-in; raise BlockPeriods or lower MaxPeriods")
	}
	w.batch = int64(cfg.Batch)
	if w.batch <= 0 {
		w.batch = defaultBatch
	}
	pool := cfg.PoolSize
	if pool <= 0 {
		pool = defaultPoolSize
	}
	if err := w.allocResult(); err != nil {
		return nil, err
	}
	if w.delays {
		w.allocGroups()
	}
	w.allocWhen()
	nw, nslots, err := w.fitWorkers(pool)
	if err != nil {
		return nil, err
	}
	w.sizeWhen(w.memoryLimit())
	w.allocRows()
	w.allocSlots(nslots)
	if err := w.setupStream(nw); err != nil {
		return nil, err
	}
	w.startPool(nw, pool)
	return w, nil
}

// checkRunConfig refuses the settings a run cannot take: the round stream
// with queues, periods in a started run, and a series without a length.
func checkRunConfig(m *Compiled, cfg Config) error {
	switch {
	case m.HasQueues() && cfg.Observer != nil:
		return errors.New("config: the round stream does not follow a round through a queue yet: a model with queues runs without it")
	case m.GivenState() && cfg.Observer != nil:
		return errors.New("config: the round stream follows a round from its arrival, which a given state's rounds had before the run: a run from a given state runs without it")
	case m.Started() && cfg.Periods != 0:
		return errors.New("config: a started run counts repeats of its horizon, not periods: set Repeats")
	case cfg.Periods == 0 && cfg.Repeats == 0 && cfg.Series:
		return errors.New("config: Series needs a fixed Periods, or Repeats in a started run")
	}
	return nil
}

// allocRows allocates the series a run keeps, and a started run's
// per-period statistics.
func (w *Workspace) allocRows() {
	m, cfg := w.m, w.cfg
	if cfg.Series {
		n := cfg.Periods + cfg.Repeats*m.horizon
		w.result.Series = &Series{l: w.l, rows: make([]float64, int(n)*w.l.width), n: n, horizon: m.horizon}
	}
	if w.started {
		h := int(m.horizon)
		w.result.ByPeriod = &PeriodStats{l: w.l, h: h, w: make([]stats.Welford, h*w.l.width), waitCo: make([]float64, h*len(m.queues)), stayCo: make([]float64, h*len(m.queues))}
	}
}

// allocSlots allocates the finaliser's pending list per timeline and the
// ring of slots.
func (w *Workspace) allocSlots(nslots int) {
	m := w.m
	w.stores = make([]pending, w.lines)
	for i := range w.stores {
		w.stores[i] = newPending(m.pendCap)
	}
	// bound: slots, fixed at construction
	for i := 0; i < nslots; i++ {
		sl := newSlot(w.l, w.block, m.pendCap, nslots, len(m.srcs), len(m.nodes), len(m.outs))
		sl.coarse = w.slotCoarse()
		sl.when = make([]float64, w.slotWhen)
		if w.anyOneIn && w.started {
			sl.hrow = make([]float64, w.l.layerAt)
		}
		for _, q := range w.result.Queues {
			sl.qwait = append(sl.qwait, q.wait.Hist.Empty())
			sl.qwaitC = append(sl.qwaitC, q.wait.Groups[0].Empty())
			sl.qstay = append(sl.qstay, q.stay.Hist.Empty())
			sl.qstayC = append(sl.qstayC, q.stay.Groups[0].Empty())
		}
		w.slots = append(w.slots, sl)
	}
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
		wk := &worker{ss: dist.NewSamplerSet(m.pools, pool), period: make([]float64, m.periodSlots), sums: make([]float64, len(m.trigs))}
		for _, q := range m.queues {
			wk.queues = append(wk.queues, newWQueue(q))
		}
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
	w.sched.batch = w.batch
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
// quantity q's: the integer histogram where the global one is.
func (w *Workspace) newHistLike(q int, res float64) *stats.LogHist {
	if exactBucketsOf(w.hists[q]) > 0 {
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

// allocTriggers gives each trigger's sum its accumulator and histogram,
// ranged from the sum's mean, or its level when the mean is unknown; a
// count of entries takes the integer histogram.
func (w *Workspace) allocTriggers(res float64) {
	r := &w.result
	counted := make([]bool, len(w.m.trigs))
	for _, i := range w.m.counts {
		counted[i] = true
	}
	r.Triggers = make([]Accum, len(w.m.trigs))
	for i, tr := range w.m.trigs {
		q := w.l.trig(i)
		scale := tr.firing.SumMean
		if !(scale > 0) || math.IsInf(scale, 1) {
			scale = math.Max(tr.over, 1)
		}
		w.means[q] = scale
		if counted[i] {
			r.Triggers[i].Hist = stats.NewIntHist(scale, res)
		} else {
			r.Triggers[i].Hist = stats.NewLogHist(scale, res)
		}
		w.hists[q] = r.Triggers[i].Hist
	}
}

// fitWorkers picks the worker and slot counts: Workers when set, else
// the most workers up to GOMAXPROCS whose workspace fits MaxMemory, as
// each holds its own histograms and the results do not depend on how
// many there are. It refuses when the count set, or a single worker,
// does not fit.
func (w *Workspace) fitWorkers(pool int) (nw, nslots int, err error) {
	nw = w.cfg.Workers
	if nw > 0 {
		nslots = w.slotsFor(nw)
		return nw, nslots, w.checkFootprint(nw, nslots, pool)
	}
	// bound: GOMAXPROCS
	for nw = runtime.GOMAXPROCS(0); nw > 1; nw-- {
		nslots = w.slotsFor(nw)
		if w.checkFootprint(nw, nslots, pool) == nil {
			return nw, nslots, nil
		}
	}
	nslots = w.slotsFor(1)
	return 1, nslots, w.checkFootprint(1, nslots, pool)
}

// slotsFor is the number of result slots nw workers cycle through.
func (w *Workspace) slotsFor(nw int) int {
	return int(min(int64(2*nw+2), w.batch))
}

// memoryLimit is MaxMemory, or its default.
func (w *Workspace) memoryLimit() int64 {
	if w.cfg.MaxMemory > 0 {
		return w.cfg.MaxMemory
	}
	return defaultMaxMemory
}

// checkFootprint evaluates the memory the workspace would hold and
// refuses above MaxMemory.
func (w *Workspace) checkFootprint(nw, nslots, pool int) error {
	limit := w.memoryLimit()
	a := int64(w.l.width)
	bins := int64(0)
	for _, h := range w.hists {
		if h != nil {
			bins += int64(h.Bins()) + exactBucketsOf(h)
		}
	}
	// Each slot also counts events, reach and outcomes per source.
	counters := int64(len(w.m.srcs)) * int64(1+len(w.m.nodes)+len(w.m.outs))
	// With delays, each quantity with a histogram has batch-group
	// histograms, and each slot a coarse copy of one, all sharing a bin map.
	coarse, coarseMaps := int64(0), int64(0)
	for q, h := range w.hists {
		if h != nil {
			if g := w.accum(q).Groups; g != nil {
				coarse += int64(g[0].Bins())
				coarseMaps += g[0].MapBytes()
			}
		}
	}
	// Each slot holds a row per period of its block, a scratch row and its
	// pending list; the finaliser holds one more pending list per timeline.
	pendB := int64(w.m.pendCap) * (pendPointBytes + pendSegBytes)
	whenSlotB, whenB := w.whenBytes()
	slotsB := int64(nslots) * (8*a*(w.block+1) + 64*a + 8*counters + 8*coarse + pendB + 8*int64(nslots+1) + whenSlotB)
	carryB := w.lines * pendB
	globalB := 64*a + 8*bins + 8*batchGroups*coarse + coarseMaps + whenB
	if w.lines > 1 {
		globalB += 24 * batchGroups * a // the group sums and counts
	}
	workerHistB := int64(2*nw) * 8 * bins
	poolsB := int64(nw) * int64(len(w.m.pools)) * int64(pool) * 8
	seriesB := int64(0)
	if w.cfg.Series {
		seriesB = 8 * a * (w.cfg.Periods + w.cfg.Repeats*w.m.horizon)
	}
	if w.started {
		globalB += 24 * a * w.m.horizon // the per-period means
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
	// The histograms are sized from the means alone.
	e := m.scaleMeans()
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
		if m.crit.breakdown {
			r.ByCategory[c].Hist = stats.NewLogHist(e.MeanByCategory[c], res)
			w.hists[q] = r.ByCategory[c].Hist
		}
	}
	r.BySource = make([]Accum, len(m.srcs))
	for k := range r.BySource {
		q := w.l.src(index(k))
		w.means[q] = e.MeanBySource[k]
		if m.crit.breakdown {
			r.BySource[k].Hist = stats.NewLogHist(e.MeanBySource[k], res)
			w.hists[q] = r.BySource[k].Hist
		}
	}
	r.EntriesPer = make([]Accum, len(m.nodes))
	w.allocLayers(e, res)
	w.allocTriggers(res)
	w.allocQueues(res)
	r.Contrib = make([]Accum, len(m.nodes))
	w.hasFn = make([]bool, len(m.nodes))
	for v := range m.nodes {
		w.hasFn[v] = m.nodes[v].hasFunctions()
		w.means[w.l.entry(index(v))] = e.EntriesPer[v]
		w.means[w.l.contrib(index(v))] = e.Contrib[v]
		if w.cfg.NodeHist {
			r.EntriesPer[v].Hist = stats.NewIntHist(e.EntriesPer[v], res)
			w.hists[w.l.entry(index(v))] = r.EntriesPer[v].Hist
			if w.hasFn[v] && m.crit.breakdown {
				r.Contrib[v].Hist = stats.NewLogHist(e.Contrib[v], res)
				w.hists[w.l.contrib(index(v))] = r.Contrib[v].Hist
			}
		}
	}
	return nil
}

// allocQueues gives each queue its columns' accumulators, and its waits
// and stays their histograms, ranged from the mean service time.
func (w *Workspace) allocQueues(res float64) {
	m, r := w.m, &w.result
	r.QueueCols = make([]Accum, queueCols*len(m.queues))
	r.Levels = make([]Accum, len(m.levels))
	r.Queues = make([]QueueResult, len(m.queues))
	for c, q := range m.queues {
		scale := m.poolMean[q.work] / q.speed
		if !(scale > 0) || math.IsInf(scale, 1) {
			scale = 1
		}
		r.Queues[c] = QueueResult{wait: newItemAccum(scale, res), stay: newItemAccum(scale, res)}
	}
}

// allocGroups gives each quantity with a histogram its batch-group
// histograms, for the batch-means error of its exceedance shares, and,
// where the groups are independent timelines, each quantity its group
// sums, for the error of its mean.
func (w *Workspace) allocGroups() {
	// bound: row width
	for q := 0; q < w.l.width; q++ {
		a := w.accum(q)
		if w.lines > 1 {
			a.GroupSum = make([]stats.Neumaier, batchGroups)
		}
		h := w.hists[q]
		if h != nil {
			a.Groups = make([]*stats.LogHist, batchGroups)
			a.Groups[0] = stats.NewCoarse(h, groupRes)
			for g := 1; g < batchGroups; g++ {
				a.Groups[g] = a.Groups[0].Empty()
			}
		}
		if h != nil || w.lines > 1 {
			a.GroupN = make([]int64, batchGroups)
		}
	}
}

// timelines is the number of independent timelines a run's tasks are
// dealt into: one per batch group in a model with holding times, so the
// groups' errors hold however long the holding times' correlations
// last; one for a run that keeps its periods as one history, the series
// and the round stream, and for a model without holding times, whose
// blocks share nothing. A run of fewer tasks has one timeline per task.
func timelines(m *Compiled, cfg Config, tasks int64) int64 {
	if !m.hasDelays || cfg.Series || cfg.Observer != nil {
		return 1
	}
	return min(batchGroups, tasks)
}

// slotCoarse returns a slot's coarse histograms, one for each quantity
// with a histogram, in a model with delays; nil otherwise.
func (w *Workspace) slotCoarse() []*stats.LogHist {
	if !w.delays {
		return nil
	}
	coarse := make([]*stats.LogHist, w.l.width)
	for q, h := range w.hists {
		if h != nil {
			coarse[q] = w.accum(q).Groups[0].Empty()
		}
	}
	return coarse
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
	case i < l.trigAt:
		return &w.result.Layers[i-l.layerAt]
	case i < l.queueAt:
		return &w.result.Triggers[i-l.trigAt]
	case i < l.levelAt:
		return &w.result.QueueCols[i-l.queueAt]
	default:
		return &w.result.Levels[i-l.levelAt]
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
	for i := range w.stores {
		w.stores[i].reset()
	}
	w.resetWhen()
	for i := range r.Queues {
		r.Queues[i].wait.reset()
		r.Queues[i].stay.reset()
	}
	if r.Series != nil {
		clear(r.Series.rows)
	}
	if r.ByPeriod != nil {
		clear(r.ByPeriod.w)
		clear(r.ByPeriod.waitCo)
		clear(r.ByPeriod.stayCo)
	}
	for i := range w.rows {
		w.rows[i].Observed, w.rows[i].Met = 0, false
	}
	w.lastUnmet = ""
	r.Unmet = r.Unmet[:0]
	r.Binding = ""
}

// Run simulates a fixed number of periods on the worker pool and returns
// the workspace's Result, which the next Run overwrites. One Run at a
// time: a second while one is in progress is refused, and a Close
// during a Run ends it with an error.
func (w *Workspace) Run(ctx context.Context, seed uint64) (*Result, error) {
	s := &w.sched
	s.mu.Lock()
	switch {
	case s.closing:
		s.mu.Unlock()
		return nil, errors.New("run: workspace is closed")
	case s.busy:
		s.mu.Unlock()
		return nil, errors.New("run: a run is already in progress")
	}
	s.busy = true
	s.mu.Unlock()
	w.resetResult(seed)
	w.adaptive = w.cfg.Periods == 0 && w.cfg.Repeats == 0
	w.seq = 0
	w.stopped = false
	w.start = w.clock.Now()
	s.mu.Lock()
	s.tasks, w.horizon = w.runLength()
	clear(s.slotState)
	s.running = 0
	s.active, s.done, s.stop, s.err = true, false, false, nil
	s.next, s.handNext, s.finalNext = -w.burnIn, -w.burnIn, -w.burnIn
	s.seed = seed
	s.ctx = ctx
	s.cv.Broadcast()
	// bound: wakes until the finaliser reports the run done, or Close
	for !s.done && !s.closing {
		s.cv.Wait()
		if ctx.Err() != nil && !s.stop {
			s.stop = true
			s.cv.Broadcast()
		}
	}
	s.active, s.busy = false, false
	err := s.err
	closed := !s.done
	s.mu.Unlock()
	if closed {
		// The pool is stopping; nothing may still write the result once
		// Run returns.
		w.wg.Wait()
		return nil, errors.New("run: workspace closed during the run")
	}
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
	s := &w.sched
	s.mu.Lock()
	if s.closing {
		s.mu.Unlock()
		return
	}
	s.closing = true
	s.cv.Broadcast()
	s.mu.Unlock()
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
// from them, and which a wide row lowers so that a slot's rows hold at
// most maxSlotValues values.
func blockPeriods(cfg Config, width int) int64 {
	block := max(int64(cfg.BlockPeriods), 0)
	if block == 0 {
		block = defaultBlockPeriods
		if cfg.Periods > 0 {
			block = min(block, max(cfg.Periods/(2*minBlocks), 1))
		}
		block = min(block, maxSlotValues/int64(max(width, 1)))
	}
	return max(block, 1)
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
// cfg.Seed, or a random seed when that is 0, and the workspace closed.
// The Result stays valid and records the seed.
func (m *Compiled) Simulate(ctx context.Context, cfg Config) (*Result, error) {
	cfg.PickSeed()
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

// Workers is the number of workers the workspace runs: Workers when
// set, else the most up to GOMAXPROCS that fit MaxMemory.
func (w *Workspace) Workers() int { return len(w.workers) }

// Timelines is the number of independent timelines the run's tasks are
// dealt into: up to 32 in a model with holding times, one with a series,
// an observer, or no holding times.
func (w *Workspace) Timelines() int { return int(w.lines) }

// BlockPeriods is the periods per task the workspace runs.
func (w *Workspace) BlockPeriods() int64 { return w.block }

// runLength is a run's tasks and the rows they fold: the periods, cut
// into blocks, or a started run's repeats, a task each; an adaptive run
// takes the cap and stops early.
func (w *Workspace) runLength() (tasks, rows int64) {
	switch {
	case w.started && w.adaptive:
		return w.repeatCap(), w.repeatCap()
	case w.started:
		return w.cfg.Repeats, w.cfg.Repeats
	case w.adaptive:
		tasks = (w.m.crit.maxPeriods + w.block - 1) / w.block
		return tasks, tasks * w.block
	}
	return (w.cfg.Periods + w.block - 1) / w.block, w.cfg.Periods
}

// repeatCap is the most repeats an adaptive started run may take: the
// cap on its rows, below the tasks the stream keys tell apart.
func (w *Workspace) repeatCap() int64 {
	return min(w.m.crit.maxPeriods, maxTasks-1)
}

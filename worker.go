package variate

import (
	"context"
	"math/rand/v2"
	"sync"

	"code.eitanity.com/variate/stats"
)

// Slot states. A slot goes free → running (simulating its block) →
// simulated → handed (the finaliser has added what earlier tasks booked
// into its block) → closing (its rows being closed) → done → free.
const (
	slotFree int8 = iota
	slotRunning
	slotSimulated
	slotHanded
	slotClosing
	slotDone
)

// worker is one simulation goroutine's private state.
type worker struct {
	k      keyed        // the keyed draws' generator
	lane   uint64       // the key of the task's timeline or repeat
	pkey   uint64       // the period's key
	spk    []uint64     // per source, its key in the period
	nround []uint32     // per source, its rounds so far in the period
	rkey   uint64       // the round's key
	evPCG  []rand.PCG   // per source, its events stream in the period
	ev     []*rand.Rand // over evPCG
	period []float64
	// Renewal source state, carried within a block: the Erlang phase and
	// the time a repair ends, from the start of the current period.
	phase     []int32
	downUntil []float64
	failed    []int32             // per chain, the units down
	chainT    []float64           // per source, the time its chains are simulated to, from the period's start
	outage    float64             // the current round's outage, when its block fired it
	tested    float64             // the value the current node's threshold tests
	vals      []float64           // per function, its value at the current node, for the expressions that read it
	sums      []float64           // per trigger, the period's sum of its inputs so far
	queues    []wqueue            // per queue, its state within the task
	start     *starter            // the exact starts' scratch, with a carried queue
	qseq      uint64              // rounds parked at the queues in the task, to order ties
	given     bool                // the round walking is a given state's: not counted in the reach or the outcomes
	hists     [2][]*stats.LogHist // by batch parity
	// Round stream scratch: the round being walked, the period's matching
	// records, and the sort scratch.
	cur      Round
	hit      bool // the round reached a watched node
	live     bool // the round can still match the Watch
	arrival  float64
	scratch  []byte
	nScratch int
	idx, tmp []int32
	keys     []float64
}

// sched is the shared state of the worker pool and the finaliser, guarded
// by one mutex and one condition variable. Workers close a handed slot's
// rows, or claim a free slot and the next task index; the finaliser hands
// slots their bookings and folds them, each in task order. Burn-in tasks
// have negative indices.
type sched struct {
	mu        sync.Mutex
	cv        *sync.Cond
	closing   bool
	busy      bool  // a caller is inside Run
	active    bool  // a run is in progress
	batch     int64 // tasks per histogram batch
	tasks     int64 // tasks in the run
	next      int64 // next task index to claim
	handNext  int64 // next task index to hand its bookings
	finalNext int64 // next task index to finalise
	seed      uint64
	ctx       context.Context
	slotState []int8
	slotTask  []int64
	running   int // slots being simulated or closed
	err       error
	stop      bool // claim no more tasks
	done      bool // the finaliser has finished the run
	// With a carried queue, each timeline's blocks run in order: chain is
	// the timelines, and lineNext the task each may run next.
	chain    int64
	lineNext []int64
}

// canClaim: a run is active, not stopped, with a task left and a free
// slot, and the task is at most one batch ahead of the finaliser. The
// workers' histograms alternate by batch, so a task two batches ahead
// would write into the set the finaliser is merging.
func (s *sched) canClaim() bool {
	ahead := s.next < 0 || s.next/s.batch <= max(s.finalNext, 0)/s.batch+1
	return s.active && !s.stop && s.next < s.tasks && ahead && s.inTurn() && s.freeSlot() >= 0
}

// inTurn: without a carried queue any task, else one whose timeline's
// last block is simulated.
func (s *sched) inTurn() bool {
	if s.chain == 0 {
		return true
	}
	return s.lineNext[lineOf(s.next, s.chain)] == s.next
}

// lineOf is task j's timeline, of n.
func lineOf(j, n int64) int64 { return ((j % n) + n) % n }

// canClose: a run is active, not stopped, and a slot is handed.
func (s *sched) canClose() bool {
	return s.active && !s.stop && s.slotIn(slotHanded) >= 0
}

// canHand: the next task to hand has been simulated.
func (s *sched) canHand() bool {
	return s.active && !s.stop && !s.done && s.slotOf(s.handNext, slotSimulated) >= 0
}

// canFinalise: the next task's slot is done, or the run is stopped with
// nothing running.
func (s *sched) canFinalise() bool {
	if !s.active || s.done {
		return false
	}
	if s.stop {
		return s.running == 0
	}
	return s.slotOf(s.finalNext, slotDone) >= 0
}

func (s *sched) freeSlot() int { return s.slotIn(slotFree) }

// slotIn returns the first slot in state st, or −1.
func (s *sched) slotIn(st int8) int {
	for i, x := range s.slotState {
		if x == st {
			return i
		}
	}
	return -1
}

// slotOf returns the slot of task in state st, or −1.
func (s *sched) slotOf(task int64, st int8) int {
	for i, x := range s.slotState {
		if x == st && s.slotTask[i] == task {
			return i
		}
	}
	return -1
}

// workerLoop runs until Close. Within a run it closes a handed slot's
// rows, or else claims a slot and a task and simulates it.
func (w *Workspace) workerLoop(wk *worker) {
	s := &w.sched
	// bound: until Close
	for {
		s.mu.Lock()
		// bound: wakes until a slot to close, a task or Close
		for !s.closing && !s.canClose() && !s.canClaim() {
			s.cv.Wait()
		}
		if s.closing {
			s.mu.Unlock()
			return
		}
		if i := s.slotIn(slotHanded); i >= 0 && s.canClose() {
			s.slotState[i] = slotClosing
			s.running++
			s.mu.Unlock()
			sl := w.slots[i]
			w.closeRows(sl, wk.hists[(sl.task/w.batch)&1])
			w.endJob(i, slotDone, sl.err)
			continue
		}
		i := s.freeSlot()
		j := s.next
		s.next++
		s.slotState[i] = slotRunning
		s.slotTask[i] = j
		s.running++
		seed := s.seed
		if s.ctx.Err() != nil {
			s.stop = true
		}
		s.mu.Unlock()

		sl := w.slots[i]
		sl.id = uint32(j & (maxTasks - 1)) //nolint:gosec // distinct for burn-in and run tasks while they number below maxTasks
		sl.task = j
		w.placeTask(sl, j)
		w.runTask(wk, sl, seed)
		w.endJob(i, slotSimulated, sl.err)
	}
}

// placeTask sets task j's timeline and block. Task j is block ⌊j/n⌋ of
// timeline j mod n, for n timelines, so the tasks of each timeline are
// every nth; a timeline's burn-in tasks are its negative blocks. A run
// task holds the periods [j·L, j·L + L) of the run's count, the last cut
// at the horizon.
func (w *Workspace) placeTask(sl *slot, j int64) {
	if w.started {
		sl.line, sl.a, sl.b = 0, 0, w.block
		return
	}
	pos := j / w.lines
	if j%w.lines < 0 {
		pos--
	}
	sl.line = j - pos*w.lines
	sl.a = max(pos*w.block, -w.m.burnIn)
	sl.b = pos*w.block + min(w.block, w.horizon-j*w.block)
}

// endJob moves slot i to state st after a worker's job on it, recording
// the first error, which stops the run.
func (w *Workspace) endJob(i int, st int8, err error) {
	s := &w.sched
	s.mu.Lock()
	s.slotState[i] = st
	s.running--
	if st == slotSimulated && s.chain > 0 {
		sl := w.slots[i]
		s.lineNext[sl.line] = sl.task + s.chain
	}
	if err != nil && s.err == nil {
		s.err = err
		s.stop = true
	}
	s.cv.Broadcast()
	s.mu.Unlock()
}

// finaliserLoop hands each simulated slot, in task order, what the tasks
// before it booked into its block, and folds closed slots into the
// result, also in task order, merging each batch's worker histograms when
// the batch completes and, on an adaptive run, deciding at each batch
// boundary whether to stop. Once stopped, slots not yet folded are
// abandoned.
func (w *Workspace) finaliserLoop() {
	s := &w.sched
	// bound: until Close
	for {
		s.mu.Lock()
		// bound: wakes until a slot to hand or fold, the run is stopped, or Close
		for !s.closing && !s.canHand() && !s.canFinalise() {
			s.cv.Wait()
		}
		if s.closing {
			s.mu.Unlock()
			return
		}
		if s.canHand() {
			w.handNextSlot()
			continue
		}
		if s.stop {
			w.endRun(s.finalNext)
			s.done = true
			s.cv.Broadcast()
			s.mu.Unlock()
			continue
		}
		i := s.slotOf(s.finalNext, slotDone)
		s.mu.Unlock()

		stopNow, last := w.finaliseOne(i)

		s.mu.Lock()
		s.slotState[i] = slotFree
		s.finalNext++
		if stopNow || last {
			s.stop = true
			if s.running == 0 {
				w.endRun(s.finalNext)
				s.done = true
			}
		}
		s.cv.Broadcast()
		s.mu.Unlock()
	}
}

// handNextSlot hands the next task's slot its bookings. It is called with
// the scheduler's lock held and returns with it released.
func (w *Workspace) handNextSlot() {
	s := &w.sched
	i := s.slotOf(s.handNext, slotSimulated)
	s.mu.Unlock()
	sl := w.slots[i]
	err := w.handOff(sl)
	s.mu.Lock()
	s.slotState[i] = slotHanded
	if sl.task < 0 {
		s.slotState[i] = slotDone // a burn-in task has no rows to close
	}
	s.handNext++
	if err != nil && s.err == nil {
		s.err = err
		s.stop = true
	}
	s.cv.Broadcast()
	s.mu.Unlock()
}

// finaliseOne folds the slot's task into the result and, at a batch
// boundary, merges the batch's histograms and applies the criteria. It
// reports whether the run should stop and whether this was the last task.
func (w *Workspace) finaliseOne(i int) (stopNow, last bool) {
	sl := w.slots[i]
	w.finalise(sl)
	if sl.task < 0 {
		return false, false
	}
	if w.stream && !w.stopped {
		stop, err := w.deliver(sl)
		if err != nil {
			w.sched.mu.Lock()
			if w.sched.err == nil {
				w.sched.err = err
			}
			w.sched.mu.Unlock()
			return true, false
		}
		if stop {
			w.stopped = true
			w.result.Stopped = true
			w.result.Converged = false
			w.result.Binding = "stopped"
			return true, false
		}
	}
	last = sl.task == w.sched.tasks-1
	boundary := (sl.task+1)%w.batch == 0 || last
	if boundary {
		w.mergeBatchHists((sl.task / w.batch) & 1)
	}
	periods := w.rowsBefore(sl.task + 1)
	switch {
	case w.adaptive && boundary:
		stopNow = w.adaptiveStop(periods)
	case !w.adaptive && last:
		w.fixedVerdict(w.horizon)
	}
	return stopNow, last
}

// endRun closes the run once the tasks before stop are folded, adding
// what they booked past the folded periods to BeyondHorizon.
func (w *Workspace) endRun(stop int64) {
	periods := w.rowsBefore(max(stop, 0))
	if !w.adaptive && !w.stopped {
		periods = w.horizon
	}
	w.flushPending(stop)
	w.result.Periods = periods
}

// rowsBefore is the rows of the result the first n tasks fold: their
// periods, or in a started run one horizon row per repeat.
func (w *Workspace) rowsBefore(n int64) int64 {
	if w.started {
		return n
	}
	return n * w.block
}

// mergeBatchHists adds every worker's histogram set of one parity into the
// global histograms and clears it.
func (w *Workspace) mergeBatchHists(parity int64) {
	for _, wk := range w.workers {
		for i, h := range wk.hists[parity] {
			if h != nil {
				w.hists[i].Merge(h)
				h.Reset()
			}
		}
	}
}

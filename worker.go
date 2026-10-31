package variate

import (
	"context"
	"sync"

	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/stats"
)

// Slot states.
const (
	slotFree int8 = iota
	slotRunning
	slotDone
)

// worker is one simulation goroutine's private state.
type worker struct {
	ss     *dist.SamplerSet
	period []float64
	// Renewal source state, carried within a block: the Erlang phase and
	// the time a repair ends.
	phase     []int32
	downUntil []float64
	failed    []int32             // per chain, the units down
	chainT    []float64           // per source, the time its chains are simulated to
	outage    float64             // the current round's outage, when its block fired it
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
// by one mutex and one condition variable. Workers claim a free slot, then
// the next task index; the finaliser takes slots in task order.
type sched struct {
	mu        sync.Mutex
	cv        *sync.Cond
	closing   bool
	busy      bool  // a caller is inside Run
	active    bool  // a run is in progress
	batch     int64 // tasks per histogram batch
	tasks     int64 // tasks in the run
	next      int64 // next task index to claim
	finalNext int64 // next task index to finalise
	seed      uint64
	ctx       context.Context
	slotState []int8
	slotTask  []int64
	running   int
	err       error
	stop      bool // claim no more tasks
	done      bool // the finaliser has finished the run
}

// canClaim: a run is active, not stopped, with a task left and a free
// slot, and the task is at most one batch ahead of the finaliser. The
// workers' histograms alternate by batch, so a task two batches ahead
// would write into the set the finaliser is merging.
func (s *sched) canClaim() bool {
	return s.active && !s.stop && s.next < s.tasks && s.next/s.batch <= s.finalNext/s.batch+1 && s.freeSlot() >= 0
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
	return s.slotOf(s.finalNext) >= 0
}

func (s *sched) freeSlot() int {
	for i, st := range s.slotState {
		if st == slotFree {
			return i
		}
	}
	return -1
}

func (s *sched) slotOf(task int64) int {
	for i, st := range s.slotState {
		if st == slotDone && s.slotTask[i] == task {
			return i
		}
	}
	return -1
}

// workerLoop runs until Close. Within a run it claims a slot and a task,
// simulates, and marks the slot done.
func (w *Workspace) workerLoop(wk *worker) {
	s := &w.sched
	// bound: until Close
	for {
		s.mu.Lock()
		// bound: wakes until a task or Close
		for !s.closing && !s.canClaim() {
			s.cv.Wait()
		}
		if s.closing {
			s.mu.Unlock()
			return
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
		sl.id = uint32(j & (maxTasks - 1)) // j < maxTasks by construction
		sl.task = j
		sl.a = j * w.block
		sl.b = min(sl.a+w.block, w.horizon)
		w.runTask(wk, sl, seed, wk.hists[(j/w.batch)&1])

		s.mu.Lock()
		s.slotState[i] = slotDone
		s.running--
		if sl.err != nil && s.err == nil {
			s.err = sl.err
			s.stop = true
		}
		s.cv.Broadcast()
		s.mu.Unlock()
	}
}

// finaliserLoop folds done slots into the result in task order, merging
// each batch's worker histograms when the batch completes, and, on an
// adaptive run, deciding at each batch boundary whether to stop. Once
// stopped, slots still done are abandoned.
func (w *Workspace) finaliserLoop() {
	s := &w.sched
	// bound: until Close
	for {
		s.mu.Lock()
		// bound: wakes until the next task is done, the run is stopped, or Close
		for !s.closing && !s.canFinalise() {
			s.cv.Wait()
		}
		if s.closing {
			s.mu.Unlock()
			return
		}
		if s.stop {
			w.endRun(s.finalNext * w.block)
			s.done = true
			s.cv.Broadcast()
			s.mu.Unlock()
			continue
		}
		i := s.slotOf(s.finalNext)
		s.mu.Unlock()

		stopNow, last := w.finaliseOne(i)

		s.mu.Lock()
		s.slotState[i] = slotFree
		s.finalNext++
		if stopNow || last {
			s.stop = true
			if s.running == 0 {
				w.endRun(s.finalNext * w.block)
				s.done = true
			}
		}
		s.cv.Broadcast()
		s.mu.Unlock()
	}
}

// finaliseOne folds the slot's task into the result and, at a batch
// boundary, merges the batch's histograms and applies the criteria. It
// reports whether the run should stop and whether this was the last task.
func (w *Workspace) finaliseOne(i int) (stopNow, last bool) {
	sl := w.slots[i]
	w.finalise(sl)
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
	periods := (sl.task + 1) * w.block
	switch {
	case w.adaptive && boundary:
		stopNow = w.adaptiveStop(periods)
	case !w.adaptive && last:
		w.fixedVerdict(w.horizon)
	}
	return stopNow, last
}

// endRun closes the run at the given number of finalised periods.
func (w *Workspace) endRun(periods int64) {
	if !w.adaptive && !w.stopped {
		periods = w.horizon
	}
	w.flushCarry(periods)
	w.result.Periods = periods
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

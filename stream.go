package variate

import (
	"errors"
	"io"
	"time"
)

const (
	defaultRoundScratch = 65536
	maxRecordBuffer     = 64 << 20
	recordBufferSlack   = 1.5
)

// beginRound starts recording a round, when streaming; the round's
// source, arrival and period are filled in at the end, for the rounds
// that are kept.
func (w *Workspace) beginRound(wk *worker, t float64) {
	wk.cur.Steps = wk.cur.Steps[:0]
	wk.cur.Value = 0
	wk.arrival = t
	wk.hit = w.m.watchNode == nil
	wk.live = true
}

// prune stops recording a round once it enters a node from which no
// watched leaf, or no watched node it has yet to hit, can be reached.
func (w *Workspace) prune(wk *worker, v int32) {
	m := w.m
	if (m.deadLeaf != nil && m.deadLeaf[v]) || (m.deadNode != nil && !wk.hit && m.deadNode[v]) {
		wk.live = false
	}
}

// recordStep appends a step to the current round.
func (w *Workspace) recordStep(wk *worker, v, j int32, t, value float64) {
	// A round enters at most maxPath nodes, the slice's capacity.
	n := len(wk.cur.Steps)
	wk.cur.Steps = wk.cur.Steps[:n+1]
	wk.cur.Steps[n] = Step{Node: v, Outcome: j, T: t, Value: value}
	wk.cur.Value += value
	if w.m.watchNode != nil && w.m.watchNode[v] {
		wk.hit = true
	}
}

// endRound checks the round against the Watch and keeps it in the period
// scratch; a full scratch fails the run.
func (w *Workspace) endRound(wk *worker, sl *slot, k, leaf int32, p int64, end float64) {
	m := w.m
	r := &wk.cur
	if p < 0 || !wk.hit || (m.watchLeaf != nil && !m.watchLeaf[leaf]) || r.Value < m.minValue {
		return
	}
	r.Source, r.Arrival, r.Period, r.Leaf, r.End = k, wk.arrival, p, leaf, end
	size := w.recSize
	if (wk.nScratch+1)*size > len(wk.scratch) {
		sl.err = scratchFullError(len(wk.scratch) / size)
		return
	}
	putRound(wk.scratch[wk.nScratch*size:], r)
	wk.nScratch++
}

// flushPeriod sorts the period's matching rounds by arrival and appends
// them to the slot's record buffer, continuing into the slot's file.
func (w *Workspace) flushPeriod(wk *worker, sl *slot) {
	n := wk.nScratch
	wk.nScratch = 0
	if n == 0 || sl.err != nil {
		return
	}
	size := w.recSize
	sortRecords(wk.scratch, size, n, wk.idx, wk.tmp, wk.keys)
	// bound: matching rounds of the block
	for i := 0; i < n; i++ {
		rec := wk.scratch[int(wk.idx[i])*size : (int(wk.idx[i])+1)*size]
		if sl.recLen+size <= len(sl.recBuf) {
			copy(sl.recBuf[sl.recLen:], rec)
			sl.recLen += size
		} else if err := w.spillRecord(sl, rec); err != nil {
			sl.err = err
			return
		}
		sl.records++
	}
}

// spillRecord writes a record to the slot's temporary file.
func (w *Workspace) spillRecord(sl *slot, rec []byte) error {
	if sl.recFile == nil {
		return noRecordFileError()
	}
	if _, err := sl.recFile.WriteAt(rec, sl.fileLen); err != nil {
		return recordFileError(err)
	}
	sl.fileLen += int64(len(rec))
	return nil
}

// deliver hands the slot's rounds to the observer in order, assigning
// Seq, pacing against the clock, and translating ErrStop into a stop.
func (w *Workspace) deliver(sl *slot) (stop bool, err error) {
	size := w.recSize
	// bound: records in the buffer
	for off := 0; off+size <= sl.recLen; off += size {
		if stop, err = w.observe(sl.recBuf[off : off+size]); stop || err != nil {
			return stop, err
		}
	}
	// bound: records in the file
	for off := int64(0); off < sl.fileLen; off += int64(size) {
		if _, rerr := sl.recFile.ReadAt(w.fileRec, off); rerr != nil && !errors.Is(rerr, io.EOF) {
			return false, recordFileError(rerr)
		}
		if stop, err = w.observe(w.fileRec); stop || err != nil {
			return stop, err
		}
	}
	return false, nil
}

// observe decodes one record, paces, and calls the observer. This is the
// one place a panic is recovered: a panicking observer fails the run and
// leaves the workspace usable.
func (w *Workspace) observe(rec []byte) (stop bool, err error) {
	getRound(rec, &w.round)
	w.round.Seq = w.seq
	w.seq++
	if w.cfg.Pace > 0 {
		due := w.start.Add(time.Duration(w.round.Arrival * float64(w.cfg.Pace)))
		if d := due.Sub(w.clock.Now()); d > 0 {
			w.clock.Sleep(d)
		}
	}
	defer func() {
		if r := recover(); r != nil {
			err = observerPanicError()
		}
	}()
	err = w.cfg.Observer.Observe(&w.round)
	if errors.Is(err, ErrStop) {
		return true, nil
	}
	if err != nil {
		return false, observerError(err)
	}
	return false, nil
}

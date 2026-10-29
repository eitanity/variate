package variate

import (
	"errors"
	"io"
)

// Counter counts delivered rounds by leaf.
type Counter struct {
	ByLeaf []uint64
	Total  uint64
}

// NewCounter sizes the counter for a model.
func NewCounter(m *Compiled) *Counter {
	return &Counter{ByLeaf: make([]uint64, len(m.nodes))}
}

// Observe counts the round.
func (c *Counter) Observe(r *Round) error {
	c.Total++
	if int(r.Leaf) < len(c.ByLeaf) {
		c.ByLeaf[r.Leaf]++
	}
	return nil
}

// RecordWriter writes each delivered round as one fixed-layout
// little-endian binary record: the 56-byte header and 24 bytes per step.
type RecordWriter struct {
	w   io.Writer
	buf []byte
}

// NewRecordWriter returns a writer for rounds of at most maxPath steps;
// the model's LongestPath gives that.
func NewRecordWriter(w io.Writer, maxPath int) *RecordWriter {
	return &RecordWriter{w: w, buf: make([]byte, recordSize(maxPath))}
}

// Observe writes the round's record.
func (rw *RecordWriter) Observe(r *Round) error {
	n := recordSize(len(r.Steps))
	if n > len(rw.buf) {
		return errors.New("record writer: round longer than the writer's path capacity")
	}
	putRound(rw.buf[:n], r)
	if _, err := rw.w.Write(rw.buf[:n]); err != nil {
		return runError{stage: "record writer", err: err}
	}
	return nil
}

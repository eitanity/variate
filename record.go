package variate

import (
	"encoding/binary"
	"math"
)

// Record layout: a 56-byte header followed by 24 bytes per step. The
// header holds Seq, Period, Source, Arrival, End, Leaf, Value (48 bytes),
// a uint32 step count and 4 bytes of padding. Steps hold Node, Outcome,
// T and Value. All little-endian.
const (
	recordHeader = 56
	recordStep   = 24
)

// recordSize is the fixed size of a record for a path of at most maxPath
// steps.
func recordSize(maxPath int) int { return recordHeader + recordStep*maxPath }

// putRound encodes r into buf, which must hold recordSize(len(r.Steps)).
func putRound(buf []byte, r *Round) {
	le := binary.LittleEndian
	le.PutUint64(buf[0:], r.Seq)
	le.PutUint64(buf[8:], bits64(r.Period))
	le.PutUint32(buf[16:], bits32(r.Source))
	le.PutUint32(buf[20:], bits32(r.Leaf))
	le.PutUint64(buf[24:], math.Float64bits(r.Arrival))
	le.PutUint64(buf[32:], math.Float64bits(r.End))
	le.PutUint64(buf[40:], math.Float64bits(r.Value))
	le.PutUint32(buf[48:], bits32(index(len(r.Steps))))
	le.PutUint32(buf[52:], 0)
	for i, s := range r.Steps {
		o := recordHeader + recordStep*i
		le.PutUint32(buf[o:], bits32(s.Node))
		le.PutUint32(buf[o+4:], bits32(s.Outcome))
		le.PutUint64(buf[o+8:], math.Float64bits(s.T))
		le.PutUint64(buf[o+16:], math.Float64bits(s.Value))
	}
}

// getRound decodes a record into r, reusing r.Steps.
func getRound(buf []byte, r *Round) {
	le := binary.LittleEndian
	r.Seq = le.Uint64(buf[0:])
	r.Period = from64(le.Uint64(buf[8:]))
	r.Source = from32(le.Uint32(buf[16:]))
	r.Leaf = from32(le.Uint32(buf[20:]))
	r.Arrival = math.Float64frombits(le.Uint64(buf[24:]))
	r.End = math.Float64frombits(le.Uint64(buf[32:]))
	r.Value = math.Float64frombits(le.Uint64(buf[40:]))
	n := int(le.Uint32(buf[48:]))
	r.Steps = r.Steps[:n]
	for i := range r.Steps {
		o := recordHeader + recordStep*i
		r.Steps[i] = Step{
			Node:    from32(le.Uint32(buf[o:])),
			Outcome: from32(le.Uint32(buf[o+4:])),
			T:       math.Float64frombits(le.Uint64(buf[o+8:])),
			Value:   math.Float64frombits(le.Uint64(buf[o+16:])),
		}
	}
}

// The record stores signed fields as their two's-complement bit patterns;
// these helpers are reinterpretations, not value conversions.
func bits64(x int64) uint64 { return uint64(x) } //nolint:gosec // bit pattern
func bits32(x int32) uint32 { return uint32(x) } //nolint:gosec // bit pattern
func from64(u uint64) int64 { return int64(u) }  //nolint:gosec // bit pattern
func from32(u uint32) int32 { return int32(u) }  //nolint:gosec // bit pattern

// recordArrival reads the arrival time of an encoded record.
func recordArrival(buf []byte) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(buf[24:]))
}

// sortRecords orders the first n records of buf, each size bytes, by
// arrival, keeping the generation order of ties. idx and keys are
// scratch of at least n entries. It is a stable merge sort by index.
func sortRecords(buf []byte, size, n int, idx, tmp []int32, keys []float64) {
	// bound: n records
	for i := 0; i < n; i++ {
		idx[i] = int32(i) //nolint:gosec // n is the scratch length, far below 2^31
		keys[i] = recordArrival(buf[i*size:])
	}
	// bound: log2(n) merge passes
	for width := 1; width < n; width *= 2 {
		// bound: n records
		for lo := 0; lo < n; lo += 2 * width {
			mid := min(lo+width, n)
			hi := min(lo+2*width, n)
			mergeIdx(idx, tmp, keys, lo, mid, hi)
		}
		copy(idx[:n], tmp[:n])
	}
}

func mergeIdx(idx, tmp []int32, keys []float64, lo, mid, hi int) {
	i, j := lo, mid
	// bound: n records
	for k := lo; k < hi; k++ {
		if i < mid && (j >= hi || keys[idx[i]] <= keys[idx[j]]) {
			tmp[k] = idx[i]
			i++
		} else {
			tmp[k] = idx[j]
			j++
		}
	}
}

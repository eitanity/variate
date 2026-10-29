package dist

import "math/rand/v2"

const (
	// initialFill is the fill length a sampler starts a task with; each
	// refill grows it fourfold up to the buffer length, so the first fills
	// are 32, 128, 512 and then the whole buffer.
	initialFill = 8
	fillGrowth  = 4
)

// Sampler is a pool of pre-drawn values for one distribution. Every call
// to Next consumes one value; when the pool is used up it is refilled with
// fresh draws, never reused.
type Sampler struct {
	d   Distribution
	pcg *rand.PCG
	r   *rand.Rand
	buf []float64 // a sub-slice of the set's slab
	n   int       // current fill length
	i   int       // next value; equals n when a refill is due
}

// Next returns the next pre-drawn value, refilling the pool when empty.
func (s *Sampler) Next() float64 {
	if s.i == s.n {
		s.n = min(fillGrowth*s.n, len(s.buf))
		s.d.Fill(s.r, s.buf[:s.n])
		s.i = 0
	}
	v := s.buf[s.i]
	s.i++
	return v
}

// reset points the stream at the key for a task and empties the pool, so
// the first Next fills initialFill·fillGrowth values.
func (s *Sampler) reset(seed, key uint64) {
	s.pcg.Seed(seed, key)
	s.n = initialFill
	s.i = initialFill
}

// SamplerSet holds one Sampler per compiled distribution plus the control
// stream, all backed by one slab allocated once. It is reused across
// tasks by Reset.
type SamplerSet struct {
	slab     []float64
	samplers []Sampler
	keys     []uint64 // distribution index k as the low key word
	ctlKey   uint64   // one past the last distribution
	control  *rand.Rand
	ctlPCG   *rand.PCG
}

// Reset re-seeds every stream for task j under seed and empties the
// pools. Distribution k uses the key splitmix64(j<<32 | k); the control
// stream uses k = len(dists), one past the last distribution. It
// allocates nothing.
func (ss *SamplerSet) Reset(seed uint64, task uint32) {
	for k := range ss.samplers {
		ss.samplers[k].reset(seed, streamKey(task, ss.keys[k]))
	}
	ss.ctlPCG.Seed(seed, streamKey(task, ss.ctlKey))
}

// Sampler returns the pool of distribution k.
func (ss *SamplerSet) Sampler(k int) *Sampler { return &ss.samplers[k] }

// Control returns the task's control stream.
func (ss *SamplerSet) Control() *rand.Rand { return ss.control }

// Len returns the number of distributions.
func (ss *SamplerSet) Len() int { return len(ss.samplers) }

// streamKey is the second PCG seed word for distribution k of task j; k is
// below 2³², so the two halves never overlap.
func streamKey(task uint32, k uint64) uint64 {
	return splitmix64(uint64(task)<<32 | k)
}

// splitmix64 is the SplitMix64 output function, a bijection on uint64.
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

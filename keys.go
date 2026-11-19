package variate

import (
	"math/rand/v2"

	"code.eitanity.com/variate/dist"
)

// A draw's random numbers are keyed by what the draw is: the run's
// seed, the timeline or repeat, the period, the source, the round's
// number among the source's rounds of the period, and the node,
// function, holding or choice by name. Two models at one seed then draw
// alike wherever an edit does not reach: an added node, function or
// source leaves every other draw as it was, and a round that switches
// branch shifts nothing else.

// Tags keep the draws of one round or one source's period apart.
const (
	tagTime uint64 = iota + 1
	tagCount
	tagChoose
	tagOwn
	tagGroup
	tagEvents
	tagStart
	tagGiven
	tagServed
)

// splitmix is SplitMix64's finaliser, a bijection that scatters its
// input's bits.
func splitmix(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// mix is the key of b within a.
func mix(a, b uint64) uint64 { return splitmix(a ^ splitmix(b)) }

// within is the key of b within a where b is a key already scattered,
// such as a name's: one scattering less than mix.
func within(a, b uint64) uint64 { return splitmix(a ^ b) }

// nameKey is a name's key, FNV-1a scattered.
func nameKey(s string) uint64 {
	h := uint64(14695981039346656037)
	// bound: the name's bytes
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return splitmix(h)
}

// keyed is a generator reseeded for each keyed draw. Every draw writes
// its state, so it lives in the worker, a cache line from any other
// worker's.
type keyed struct {
	_    [64]byte
	seed uint64
	pcg  rand.PCG
	one  [1]float64
	r    *rand.Rand
	_    [64]byte
}

// init points the generator at its own state; g must not move after.
func (g *keyed) init() { g.r = rand.New(&g.pcg) }

// at returns the generator for key k under the run's seed.
func (g *keyed) at(k uint64) *rand.Rand {
	g.pcg.Seed(g.seed, k)
	return g.r
}

// value draws one value of d from key k.
func (g *keyed) value(d dist.Distribution, k uint64) float64 {
	d.Fill(g.at(k), g.one[:])
	return g.one[0]
}

// from draws one value of d from r.
func (g *keyed) from(d dist.Distribution, r *rand.Rand) float64 {
	d.Fill(r, g.one[:])
	return g.one[0]
}

// cachePad is a cache line's room, in elements of the smallest padded
// state, int32, either side of a worker's state.
const cachePad = 16

// The padded slices are n zero values with cachePad elements' room
// either side, for state one worker writes often, so no other worker's
// shares its cache lines.

func paddedF64(n int) []float64 {
	s := make([]float64, n+2*cachePad)
	return s[cachePad : cachePad+n : cachePad+n]
}

func paddedU64(n int) []uint64 {
	s := make([]uint64, n+2*cachePad)
	return s[cachePad : cachePad+n : cachePad+n]
}

func paddedU32(n int) []uint32 {
	s := make([]uint32, n+2*cachePad)
	return s[cachePad : cachePad+n : cachePad+n]
}

func paddedI32(n int) []int32 {
	s := make([]int32, n+2*cachePad)
	return s[cachePad : cachePad+n : cachePad+n]
}

func paddedPCG(n int) []rand.PCG {
	s := make([]rand.PCG, n+2*cachePad)
	return s[cachePad : cachePad+n : cachePad+n]
}

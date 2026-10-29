package dist

import "math/rand/v2"

// NewSamplerSet allocates the pools: poolSize values per distribution.
func NewSamplerSet(dists []Distribution, poolSize int) *SamplerSet {
	if poolSize < initialFill*fillGrowth {
		poolSize = initialFill * fillGrowth
	}
	ss := &SamplerSet{
		slab:     make([]float64, len(dists)*poolSize),
		samplers: make([]Sampler, len(dists)),
		keys:     make([]uint64, len(dists)),
		ctlPCG:   rand.NewPCG(0, 0),
	}
	ss.control = rand.New(ss.ctlPCG)
	var key uint64
	for k, d := range dists {
		pcg := rand.NewPCG(0, 0)
		ss.samplers[k] = Sampler{
			d: d, pcg: pcg, r: rand.New(pcg),
			buf: ss.slab[k*poolSize : (k+1)*poolSize],
			n:   initialFill, i: initialFill,
		}
		ss.keys[k] = key
		key++
	}
	ss.ctlKey = key
	return ss
}

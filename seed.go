package variate

import "math/rand/v2"

// maxDrawnSeed bounds a drawn seed below 2^53, so that it survives a
// JavaScript number and can be typed back in to repeat the run.
const maxDrawnSeed = 1 << 53

// PickSeed replaces a seed of 0 with a random one from 1 to 2^53 - 1, so
// that a model left at 0 runs independently each time and the Result
// records the seed it ran at; any other seed is kept.
func (c *Config) PickSeed() {
	if c.Seed == 0 {
		c.Seed = rand.Uint64N(maxDrawnSeed-1) + 1
	}
}

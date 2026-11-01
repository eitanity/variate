package variate_test

import (
	"testing"

	"code.eitanity.com/variate"
)

func TestPickSeed(t *testing.T) {
	c := variate.Config{Seed: 7}
	c.PickSeed()
	if c.Seed != 7 {
		t.Fatalf("a given seed changed to %d", c.Seed)
	}
	seen := map[uint64]bool{}
	for range 100 {
		c := variate.Config{}
		c.PickSeed()
		if c.Seed == 0 || c.Seed >= 1<<53 {
			t.Fatalf("drawn seed %d is outside 1 to 2^53 - 1", c.Seed)
		}
		seen[c.Seed] = true
	}
	if len(seen) < 99 {
		t.Fatalf("100 draws gave only %d seeds", len(seen))
	}
}

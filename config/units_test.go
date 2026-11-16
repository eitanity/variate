package config_test

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// TestUnitScale: a time and a rate in hours, days and weeks into a
// period of a year, and the refusals.
func TestUnitScale(t *testing.T) {
	year := 8760.0
	for _, c := range []struct {
		unit       string
		time, rate float64
	}{{"hours", 1 / year, year}, {"days", 24 / year, year / 24}, {"weeks", 168 / year, year / 168}, {"periods", 1, 1}, {"", 1, 1}} {
		tm, err := config.UnitScale(c.unit, year, true)
		if err != nil || tm != c.time {
			t.Errorf("%q time: %v %v, want %v", c.unit, tm, err, c.time)
		}
		rt, err := config.UnitScale(c.unit, year, false)
		if err != nil || rt != c.rate {
			t.Errorf("%q rate: %v %v, want %v", c.unit, rt, err, c.rate)
		}
	}
	if _, err := config.UnitScale("fortnights", year, true); err == nil {
		t.Error("an unknown unit accepted")
	}
	if _, err := config.UnitScale("hours", 0, true); err == nil || !strings.Contains(err.Error(), "period") {
		t.Errorf("no period: %v", err)
	}
}

// TestBlockAsData: the cooling plant's block built through BlockChain
// and BlockSource is the one its file gives, chain for chain, with the
// same figures, and compiles to the same exact mean; a block without a
// repair time is the same Poisson source either way.
func TestBlockAsData(t *testing.T) {
	m, err := config.Load("../examples/blocks.yaml")
	if err != nil {
		t.Fatal(err)
	}
	year := 365 * 24.0
	r := func(lo, hi float64) *variate.InputRange { return &variate.InputRange{Low: lo, High: hi} }
	chains := make([]variate.Chain, 0, 3)
	for _, b := range []struct {
		k, n       int
		mtbf, mttr float64
		upR, downR *variate.InputRange
	}{{2, 3, 6000, 36, r(4000, 9000), r(24, 72)}, {1, 2, 12000, 24, r(8000, 20000), nil}, {1, 1, 30000, 8, nil, nil}} {
		ch, err := config.BlockChain(b.k, b.n, b.mtbf, b.mttr, b.upR, b.downR, "hours", year)
		if err != nil {
			t.Fatal(err)
		}
		chains = append(chains, ch)
	}
	g := m.Graph
	file := g.Sources[0]
	built := variate.Source{ID: file.ID, To: file.To}
	fig, err := config.BlockSource(file.ID, chains, year, &built)
	if err != nil {
		t.Fatal(err)
	}
	if fig != m.Blocks[0] {
		t.Errorf("figures %+v, the file's %+v", fig, m.Blocks[0])
	}
	if len(built.Chains) != len(file.Chains) {
		t.Fatalf("%d chains, the file's %d", len(built.Chains), len(file.Chains))
	}
	for i := range built.Chains {
		a, b := built.Chains[i], file.Chains[i]
		if a.K != b.K || a.N != b.N || a.Fail != b.Fail || a.Repair != b.Repair || !sameRange(a.MTBFRange, b.MTBFRange) || !sameRange(a.MTTRRange, b.MTTRRange) {
			t.Errorf("chain %d: %+v, the file's %+v", i, a, b)
		}
	}
	g.Sources[0] = built
	if mean(t, g) != mean(t, m.Graph) {
		t.Error("the exact means differ")
	}
	ch, err := config.BlockChain(1, 1, 2000, 0, nil, nil, "hours", year)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Source
	if _, err := config.BlockSource("pump", []variate.Chain{ch}, year, &s); err != nil || s.Chains != nil || s.Frequency == nil || s.Frequency.Mean() != year/2000 {
		t.Errorf("no repair: chains %v, frequency %v, %v; want a Poisson source at %v a period", s.Chains, s.Frequency, err, year/2000)
	}
}

func sameRange(a, b *variate.InputRange) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func mean(t *testing.T, g variate.Graph) float64 {
	t.Helper()
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := c.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(e.Mean) {
		t.Fatal("no exact mean")
	}
	return e.Mean
}

// TestBlockChainRefusals: what a block may not be, refused as data as it
// is in a file.
func TestBlockChainRefusals(t *testing.T) {
	for _, c := range []struct {
		name       string
		k, n       int
		mtbf, mttr float64
		downR      *variate.InputRange
		unit       string
		period     float64
		want       string
	}{
		{"no unit", 1, 1, 10, 1, nil, "", 8760, "unit"},
		{"no period", 1, 1, 10, 1, nil, "hours", 0, "period"},
		{"k above n", 3, 2, 10, 1, nil, "hours", 8760, "k ≤ n"},
		{"mtbf", 1, 1, 0, 1, nil, "hours", 8760, "mtbf must be positive"},
		{"mttr", 1, 1, 10, -1, nil, "hours", 8760, "mttr must be zero or positive"},
		{"range on no repair", 1, 1, 10, 0, &variate.InputRange{Low: 1, High: 2}, "hours", 8760, "replaced at once"},
	} {
		if _, err := config.BlockChain(c.k, c.n, c.mtbf, c.mttr, nil, c.downR, c.unit, c.period); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error naming %q", c.name, err, c.want)
		}
	}
}

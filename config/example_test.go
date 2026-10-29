package config_test

import (
	"context"
	"fmt"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// A model file: run settings at the top, then the graphs. Parse takes
// the text and the directory data files resolve in (empty: none).
func ExampleParse() {
	const text = `
periods: 10000
seed: 11
period: { days: 365 }
graphs:
  - id: pumps
    sources:
      - { id: trip, frequency: { pert: [2, 5, 12] }, to: repair }
    nodes:
      - id: repair
        functions:
          - { name: callout, category: labour, pert: [500, 1200, 4000] }
        outcomes:
          - { to: replace, p: { beta: [2, 8] } }
          - { to: done, holding: { pert: [1, 3, 10], unit: days } }
      - id: replace
        functions:
          - { name: pump, category: capital, logpert: [2e4, 5e4, 2e5] }
      - id: done
`
	m, err := config.Parse([]byte(text), "")
	if err != nil {
		fmt.Println(err)
		return
	}
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		fmt.Println(err)
		return
	}
	e, _ := c.Analytic()
	r, err := c.Simulate(context.Background(), m.Config)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("period %g hours  periods %d  exact mean %.0f\n", m.PeriodHours, r.Periods, e.Mean)
	fmt.Printf("simulated mean within 5%% of exact: %v\n", within(r.Total.Mean(), e.Mean, 0.05*e.Mean))
	// Output:
	// period 8760 hours  periods 10000  exact mean 76134
	// simulated mean within 5% of exact: true
}

func within(x, want, tol float64) bool { return x > want-tol && x < want+tol }

// Settings arrive from the file or from flags; SetSetting takes either
// spelling.
func ExampleSetSetting() {
	var cfg variate.Config
	_ = config.SetSetting(&cfg, "min-hits", "500")
	_ = config.SetSetting(&cfg, "tail", "0.99")
	fmt.Println(cfg.MinHits, cfg.TailP)
	// Output:
	// 500 0.99
}

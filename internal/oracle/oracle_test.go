package oracle_test

import (
	"math"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/internal/oracle"
)

func TestOracleMatchesAnalytic(t *testing.T) {
	pertF, _ := dist.NewPERT(1, 2, 4, 4)
	pertB, _ := dist.NewPERT(1, 2, 3, 4)
	ten, _ := dist.NewConstant(10)
	p3, _ := dist.NewConstant(0.3)
	p7, _ := dist.NewConstant(0.7)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: pertF, To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: p3}, {To: "c", P: p7}}},
			{ID: "b", Next: "d", Functions: []variate.Function{{Name: "fb", Category: "labour", Model: pertB}}},
			{ID: "c", Next: "d"},
			{ID: "d", Functions: []variate.Function{{Name: "fd", Category: "parts", Model: ten}}},
		},
	}
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	const periods = 20000
	r := oracle.Run(g, periods, 0, 11)
	mean, m2 := 0.0, 0.0
	for _, x := range r.Total {
		mean += x
	}
	mean /= periods
	for _, x := range r.Total {
		m2 += (x - mean) * (x - mean)
	}
	m2 /= periods
	if z := (mean - e.Mean) / math.Sqrt(e.Var/periods); math.Abs(z) > 4.5 {
		t.Errorf("mean %g analytic %g z=%.2f", mean, e.Mean, z)
	}
	if math.Abs(m2-e.Var) > 0.1*e.Var {
		t.Errorf("variance %g analytic %g", m2, e.Var)
	}
	entries := 0.0
	for _, x := range r.Entries[1] {
		entries += x
	}
	if z := (entries/periods - e.EntriesPer[1]) / math.Sqrt(e.EntriesPer[1]/periods); math.Abs(z) > 4.5 {
		t.Errorf("entries at b %g analytic %g z=%.2f", entries/periods, e.EntriesPer[1], z)
	}
	if r.Reach[0][3] != r.Events[0] || r.Reach[0][1]+r.Reach[0][2] != r.Events[0] {
		t.Errorf("reach %v events %v", r.Reach[0], r.Events)
	}
}

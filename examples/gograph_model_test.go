package examples_test

import (
	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// gographModel mirrors examples/gograph/main.go so the test need not
// build a program; keep the two in step.
func gographModel() variate.Graph {
	rate, _ := dist.NewPERT(1, 3, 8, dist.DefaultPERTShape)
	block, _ := dist.NewBeta(8, 2)
	minor, _ := dist.NewPERT(500, 2000, 8000, dist.DefaultPERTShape)
	major, _ := dist.NewLogPERT(1e4, 5e4, 1e6, dist.DefaultPERTShape)
	mostlyMinor, _ := dist.NewConstant(0.8)
	return variate.Graph{
		ID: "scenario",
		Sources: []variate.Source{
			{ID: "threat", Frequency: rate, To: "control"},
		},
		Nodes: []variate.Node{
			{ID: "control", Outcomes: []variate.Outcome{
				{To: "blocked", P: block},
				{To: "impact"},
			}},
			{ID: "blocked"},
			{ID: "impact", Outcomes: []variate.Outcome{
				{To: "minor", P: mostlyMinor},
				{To: "major"},
			}},
			{ID: "minor", Functions: []variate.Function{{Name: "response", Category: "response", Model: minor}}},
			{ID: "major", Functions: []variate.Function{{Name: "loss", Category: "loss", Model: major}}},
		},
	}
}

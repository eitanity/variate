package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

const model = "testdata/site.yaml"

func invoke(t *testing.T, args ...string) (stdout, stderr string, code int) {
	t.Helper()
	var out, errb bytes.Buffer
	code = run(args, &out, &errb)
	return out.String(), errb.String(), code
}

func TestVersionAndUsage(t *testing.T) {
	out, _, code := invoke(t, "version")
	if code != 0 || strings.TrimSpace(out) == "" {
		t.Errorf("version: %q %d", out, code)
	}
	_, errOut, code := invoke(t)
	if code != 1 || !strings.Contains(errOut, "usage") {
		t.Errorf("no args: %q %d", errOut, code)
	}
	_, errOut, code = invoke(t, "bogus")
	if code != 1 || !strings.Contains(errOut, "unknown command") {
		t.Errorf("bogus: %q", errOut)
	}
	_, errOut, code = invoke(t, "run")
	if code != 1 || !strings.Contains(errOut, "-f model file is required") {
		t.Errorf("missing -f: %q", errOut)
	}
	_, errOut, code = invoke(t, "run", "-f", model, "--format", "xml")
	if code != 1 || !strings.Contains(errOut, "--format") {
		t.Errorf("bad format: %q", errOut)
	}
	_, errOut, code = invoke(t, "run", "-f", model, "--min-hits", "lots")
	if code != 1 || !strings.Contains(errOut, "whole number") {
		t.Errorf("bad setting: %q", errOut)
	}
	_, errOut, code = invoke(t, "run", "-f", "nope.yaml")
	if code != 1 || !strings.Contains(errOut, "open model") {
		t.Errorf("missing file: %q", errOut)
	}
}

func TestPlan(t *testing.T) {
	out, _, code := invoke(t, "plan", "-f", model, "--min-hits", "50")
	if code != 0 {
		t.Fatalf("plan failed: %s", out)
	}
	for _, want := range []string{"criterion", "min-hits", "pump_a_repair.assess→new_pump", "binding:", "expected rounds per period: 4", "estimated time"} {
		if !strings.Contains(out, want) {
			t.Errorf("plan output lacks %q:\n%s", want, out)
		}
	}
}

// jsonSummary is the shape the test decodes the JSON report into; the
// writer builds it by hand from typed values, so decoding it here is
// the check that the two agree.
type jsonSummary struct {
	Seed       uint64                     `json:"seed"`
	Periods    int64                      `json:"periods"`
	Converged  bool                       `json:"converged"`
	Binding    string                     `json:"binding"`
	Total      jsonStat                   `json:"total"`
	ByCategory map[string]jsonStat        `json:"by_category"`
	BySource   map[string]jsonStat        `json:"by_source"`
	EntriesPer map[string]jsonStat        `json:"entries_per"`
	Reach      map[string]map[string]prop `json:"reach"`
	Outcomes   map[string]prop            `json:"outcomes"`
	Plan       []map[string]any           `json:"plan"`
	Exceedance []map[string]float64       `json:"exceedance"`
	Analytic   *struct {
		Mean      float64 `json:"mean"`
		Transform struct {
			Applicable bool                 `json:"applicable"`
			Quantiles  map[string][]float64 `json:"quantiles"`
		} `json:"transform"`
	} `json:"analytic"`
}

type jsonStat struct {
	N    int64   `json:"n"`
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P999 float64 `json:"p999"`
}

type prop struct {
	P      float64 `json:"p"`
	StdErr float64 `json:"stderr"`
}

// TestRunJSON: the JSON report decodes and matches a direct run with the
// same seed.
func TestRunJSON(t *testing.T) {
	out, _, code := invoke(t, "run", "-f", model, "-n", "3000", "--seed", "5", "--format", "json", "--analytic")
	if code != 0 {
		t.Fatalf("run failed: %s", out)
	}
	var got jsonSummary
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("decode: %v\n%s", err, out)
	}
	m, err := config.Load(model)
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods, m.Config.Seed, m.Config.Transform = 3000, 5, true
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(t.Context(), 5)
	if err != nil {
		t.Fatal(err)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	if got.Seed != 5 || got.Periods != r.Periods || got.Converged != r.Converged || got.Binding != r.Binding {
		t.Errorf("header %+v vs %+v", got, r)
	}
	if got.Total.Mean != sum.Total.Mean || got.Total.P50 != sum.Total.P50 || got.Total.N != sum.Total.N {
		t.Errorf("total %+v vs %+v", got.Total, sum.Total)
	}
	for name, s := range sum.ByCategory {
		if got.ByCategory[name].Mean != s.Mean {
			t.Errorf("category %s %g vs %g", name, got.ByCategory[name].Mean, s.Mean)
		}
	}
	for src, nodes := range sum.ReachProb {
		for node, p := range nodes {
			if g := got.Reach[src][node]; g.P != p.P && !(math.IsNaN(p.P) && g.P == 0) {
				t.Errorf("reach %s %s %g vs %g", src, node, g.P, p.P)
			}
		}
	}
	if len(got.Plan) != len(sum.Plan) || len(got.Exceedance) != len(sum.Exceedance) {
		t.Errorf("plan %d/%d exceedance %d/%d", len(got.Plan), len(sum.Plan), len(got.Exceedance), len(sum.Exceedance))
	}
	e, _ := c.Analytic()
	if got.Analytic == nil || got.Analytic.Mean != e.Mean || !got.Analytic.Transform.Applicable {
		t.Errorf("analytic %+v", got.Analytic)
	}
	if q := got.Analytic.Transform.Quantiles["p99"]; len(q) != 2 || q[0] > q[1] {
		t.Errorf("p99 bracket %v", q)
	}
	for _, pt := range got.Exceedance {
		if pt["lower"] > pt["p"]+5*pt["stderr"]+1e-9 || pt["upper"] < pt["p"]-5*pt["stderr"]-1e-9 {
			t.Errorf("exceedance point %v outside the bounds", pt)
		}
	}
}

// TestRunCSVAndFiles: the CSV summary and the exceedance file parse, and
// --out writes the summary to a file.
func TestRunCSVAndFiles(t *testing.T) {
	dir := t.TempDir()
	outFile := filepath.Join(dir, "summary.csv")
	ecFile := filepath.Join(dir, "ec.csv")
	stdout, _, code := invoke(t, "run", "-f", model, "-n", "2000", "--seed", "2", "--format", "csv", "--out", outFile, "--exceedance", ecFile, "--analytic")
	if code != 0 || stdout != "" {
		t.Fatalf("code %d stdout %q", code, stdout)
	}
	for _, f := range []string{outFile, ecFile} {
		data, err := os.ReadFile(f) //nolint:gosec // test temp dir
		if err != nil {
			t.Fatal(err)
		}
		rows, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if len(rows) < 3 {
			t.Errorf("%s: %d rows", f, len(rows))
		}
		if f == ecFile && (len(rows[0]) != 5 || rows[0][3] != "lower") {
			t.Errorf("exceedance header %v", rows[0])
		}
		if f == outFile && (rows[0][0] != "quantity" || rows[1][0] != "total") {
			t.Errorf("summary rows %v %v", rows[0], rows[1])
		}
	}
	table, _, code := invoke(t, "run", "-f", model, "-n", "500", "--seed", "2", "--node-hist")
	if code != 0 || !strings.Contains(table, "seed 2  periods 500") || !strings.Contains(table, "entries new_pump") {
		t.Errorf("table:\n%s", table)
	}
}

type jsonRound struct {
	Seq    uint64 `json:"seq"`
	Period int64  `json:"period"`
	Source string `json:"source"`
	Leaf   string `json:"leaf"`
	Value  float64
	Steps  []struct {
		Node    string  `json:"node"`
		Outcome *string `json:"outcome"`
	} `json:"steps"`
}

// TestStream: rounds stream to stdout as JSON Lines in sequence order,
// the summary goes to stderr, and the other formats write to a file.
func TestStream(t *testing.T) {
	out, errOut, code := invoke(t, "stream", "-f", model, "-n", "50", "--seed", "3", "--watch", "new_pump")
	if code != 0 {
		t.Fatalf("stream failed: %s", errOut)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) < 10 || !strings.Contains(errOut, "seed 3") {
		t.Fatalf("%d lines, summary %q", len(lines), errOut)
	}
	for i, line := range lines {
		var r jsonRound
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("line %d: %v", i, err)
		}
		if r.Seq != uint64(i) || r.Leaf != "new_pump" || r.Steps[len(r.Steps)-1].Outcome != nil {
			t.Errorf("round %+v", r)
		}
	}
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "rounds.csv")
	binFile := filepath.Join(dir, "rounds.bin")
	if _, errOut, code := invoke(t, "run", "-f", model, "-n", "50", "--seed", "3", "--rounds", "csv", "--rounds-out", csvFile, "--out", filepath.Join(dir, "s.txt")); code != 0 {
		t.Fatalf("csv rounds: %s", errOut)
	}
	data, _ := os.ReadFile(csvFile) //nolint:gosec // test temp dir
	rows, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
	if err != nil || len(rows) < 2 || rows[0][7] != "path" || !strings.Contains(rows[1][7], ">") {
		t.Errorf("csv rounds: %v %v", err, rows[:min(2, len(rows))])
	}
	if _, errOut, code := invoke(t, "run", "-f", model, "-n", "50", "--seed", "3", "--rounds", "bin", "--rounds-out", binFile, "--out", filepath.Join(dir, "s.txt")); code != 0 {
		t.Fatalf("bin rounds: %s", errOut)
	}
	if info, err := os.Stat(binFile); err != nil || info.Size()%8 != 0 || info.Size() < 56 {
		t.Errorf("bin rounds: %v", err)
	}
	if _, errOut, code := invoke(t, "run", "-f", model, "-n", "5", "--rounds", "csv"); code != 1 || !strings.Contains(errOut, "--rounds-out") {
		t.Errorf("rounds without a file: %q", errOut)
	}
}

// TestPlanJSONAndSeries: plan --format json gives the rows and figures
// as JSON, and run --series --format json carries the per-period
// columns.
func TestPlanJSONAndSeries(t *testing.T) {
	model := filepath.Join("..", "..", "examples", "blocks.yaml")
	out, _, code := invoke(t, "plan", "-f", model, "--format", "json")
	if code != 0 {
		t.Fatalf("plan exit %d", code)
	}
	var plan struct {
		Plan []struct {
			Criterion string `json:"criterion"`
			Periods   int64  `json:"periods"`
		} `json:"plan"`
		Periods       int64   `json:"periods"`
		Binding       string  `json:"binding"`
		Mean          float64 `json:"mean"`
		EstimatedTime string  `json:"estimated_time"`
		Blocks        []struct {
			Source string  `json:"source"`
			MTBF   float64 `json:"mtbf"`
		} `json:"blocks"`
	}
	if err := json.Unmarshal([]byte(out), &plan); err != nil {
		t.Fatalf("decode: %v\n%s", err, out)
	}
	if len(plan.Plan) == 0 || plan.Periods <= 0 || plan.Binding == "" || plan.Mean <= 0 || plan.EstimatedTime == "" || len(plan.Blocks) != 1 || plan.Blocks[0].Source != "plant_down" {
		t.Errorf("plan %+v", plan)
	}
	out, _, code = invoke(t, "run", "-f", filepath.Join("..", "..", "examples", "compound.yaml"), "-n", "300", "--seed", "2", "--series", "--format", "json")
	if code != 0 {
		t.Fatalf("run exit %d", code)
	}
	var run struct {
		Series map[string][]float64 `json:"series"`
	}
	if err := json.Unmarshal([]byte(out), &run); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(run.Series["total"]) != 300 || len(run.Series["category:money"]) != 300 || len(run.Series["source:incident"]) != 300 {
		t.Errorf("series columns %v", len(run.Series["total"]))
	}
}

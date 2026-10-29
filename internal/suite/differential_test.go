package suite

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/report"
)

// corpusModel is one model of the differential corpus.
type corpusModel struct {
	name string
	g    variate.Graph
	path string         // file on disk, for the CLI; empty for generated graphs
	cfg  variate.Config // the file's own settings, which the CLI applies too
}

// corpus is every example model, every demo model at its default knobs,
// and three random DAGs.
func corpus(t *testing.T) []corpusModel {
	t.Helper()
	var out []corpusModel
	files, _ := filepath.Glob("../../examples/*.yaml")
	for _, f := range files {
		m, err := config.Load(f)
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		out = append(out, corpusModel{name: filepath.Base(f), g: m.Graph, path: f, cfg: m.Config})
	}
	out = append(out, demoModels(t)...)
	for seed := uint64(1); seed <= 3; seed++ {
		out = append(out, corpusModel{name: "random" + strconv.FormatUint(seed, 10), g: genGraph(t, seed, seed == 2)})
	}
	return out
}

// demoModels substitutes the default knobs of web/models/index.json into
// the templates and writes the result to a temp file for the CLI.
func demoModels(t *testing.T) []corpusModel {
	t.Helper()
	data, err := os.ReadFile("../../web/models/index.json")
	if err != nil {
		t.Fatal(err)
	}
	var index []struct {
		ID    string `json:"id"`
		File  string `json:"file"`
		Knobs []struct {
			Key   string  `json:"key"`
			Value float64 `json:"value"`
		} `json:"knobs"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	out := make([]corpusModel, 0, len(index))
	for _, dm := range index {
		text, err := os.ReadFile(filepath.Join("../../web/models", dm.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		yaml := string(text)
		for _, k := range dm.Knobs {
			yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(k.Value, 'g', -1, 64))
		}
		path := filepath.Join(dir, "demo-"+dm.File)
		if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil { //nolint:gosec // test temp dir
			t.Fatal(err)
		}
		m, err := config.Parse([]byte(yaml), "")
		if err != nil {
			t.Fatalf("%s: %v", dm.ID, err)
		}
		out = append(out, corpusModel{name: "demo " + dm.ID, g: m.Graph, path: path, cfg: m.Config})
	}
	return out
}

// libraryJSON runs the model through the library and assembles the
// report the way every entry point does.
func libraryJSON(t *testing.T, g variate.Graph, cfg variate.Config, seed uint64) ([]byte, *variate.Result) {
	t.Helper()
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := report.New(c, w, r, true)
	if err != nil {
		t.Fatal(err)
	}
	return rep.JSON(), r
}

// TestDifferentialWorkers: the report is byte-identical for 1, 3 and 32
// workers on every model of the corpus.
func TestDifferentialWorkers(t *testing.T) {
	for _, m := range corpus(t) {
		var first []byte
		for _, workers := range []int{1, 3, 32} {
			cfg := variate.Config{Periods: 2000, Workers: workers, BlockPeriods: 50}
			got, _ := libraryJSON(t, m.g, cfg, 7)
			if first == nil {
				first = got
				continue
			}
			if !bytes.Equal(got, first) {
				t.Errorf("%s: %d workers differ from 1 worker:\n%s", m.name, workers, firstDiff(first, got))
			}
		}
	}
}

// TestDifferentialSimulate: the one-shot Simulate agrees with Run in
// every result field a report is built from.
func TestDifferentialSimulate(t *testing.T) {
	for _, m := range corpus(t) {
		cfg := variate.Config{Periods: 1000, Seed: 9, BlockPeriods: 50}
		_, r := libraryJSON(t, m.g, cfg, 9)
		c, _, err := m.g.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		s, err := c.Simulate(context.Background(), cfg)
		if err != nil {
			t.Fatal(err)
		}
		if s.Periods != r.Periods || s.Steps != r.Steps || s.Converged != r.Converged || s.Binding != r.Binding || s.TailPeriods != r.TailPeriods {
			t.Errorf("%s: header differs: %+v vs %+v", m.name, s.Periods, r.Periods)
		}
		if s.Total.Mean() != r.Total.Mean() || s.Total.Max != r.Total.Max || s.Total.Zero != r.Total.Zero {
			t.Errorf("%s: totals differ: %g vs %g", m.name, s.Total.Mean(), r.Total.Mean())
		}
		for k := range r.Events {
			if s.Events[k] != r.Events[k] {
				t.Errorf("%s: events differ for source %d", m.name, k)
			}
			for v := range r.Reach[k] {
				if s.Reach[k][v] != r.Reach[k][v] {
					t.Errorf("%s: reach differs at %d/%d", m.name, k, v)
				}
			}
		}
	}
}

// TestDifferentialCLI: variate run --format json --analytic on each
// model file gives the library's report byte for byte.
func TestDifferentialCLI(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	bin := filepath.Join(t.TempDir(), "variate")
	build := exec.Command("go", "build", "-o", bin, "../../cmd/variate") //nolint:gosec // builds the CLI under test
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	for _, m := range corpus(t) {
		if m.path == "" {
			continue
		}
		// The exact transform at compile is slow on models with an
		// uncertain rate beside a per-period probability, so --analytic
		// is compared on the compound model only.
		analytic := m.name == "compound.yaml"
		// The file's own settings apply on both sides; the flags override
		// the same fields on the command line as here.
		cfg := m.cfg
		cfg.Periods, cfg.Seed, cfg.BlockPeriods, cfg.Transform = 2000, 5, 50, analytic
		c, _, err := m.g.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		r, err := w.Run(context.Background(), 5)
		if err != nil {
			t.Fatal(err)
		}
		rep, _ := report.New(c, w, r, analytic)
		want := rep.JSON()
		w.Close()
		args := []string{"run", "-f", m.path, "-n", "2000", "--seed", "5", "--block-periods", "50", "--format", "json"}
		if analytic {
			args = append(args, "--analytic")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		cmd := exec.CommandContext(ctx, bin, args...) //nolint:gosec // test binary
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		err = cmd.Run()
		cancel()
		if err != nil {
			t.Fatalf("%s: %v\n%s", m.name, err, stderr.String())
		}
		if got := stdout.Bytes(); !bytes.Equal(got, want) {
			t.Errorf("%s: CLI differs from library:\n%s", m.name, firstDiff(want, got))
		}
	}
}

// TestDifferentialWasmAssembly: the report assembled as the wasm entry
// point does (with the drawing histograms and warnings) is the CLI's
// report with three keys added, and nothing else changed.
func TestDifferentialWasmAssembly(t *testing.T) {
	m, err := config.Load("../../examples/compound.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg := variate.Config{Periods: 2000, Seed: 3, Transform: true}
	c, warnings, err := m.Graph.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	plain, _ := report.New(c, w, r, true)
	extended, _ := report.New(c, w, r, true)
	top := extended.Sum.Total.P99 * 1.05
	extended.Hist = report.LinearHistogram(w, top, 50, c.LatticeUnit())
	extended.LogHist = report.LogHistogram(w, top/1e4, top, 50)
	extended.Warnings = warnings
	var a, b map[string]json.RawMessage
	if err := json.Unmarshal(plain.JSON(), &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(extended.JSON(), &b); err != nil {
		t.Fatal(err)
	}
	for k, v := range a {
		if !bytes.Equal(v, b[k]) {
			t.Errorf("key %s differs", k)
		}
	}
	if len(b) != len(a)+3 || b["histogram"] == nil || b["log_histogram"] == nil || b["warnings"] == nil {
		t.Errorf("extended report keys %d vs %d", len(b), len(a))
	}
}

// firstDiff shows the first differing region of two byte strings.
func firstDiff(a, b []byte) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	lo := int(math.Max(0, float64(i-60)))
	return "at byte " + strconv.Itoa(i) + ":\n  a: " + string(a[lo:min(len(a), i+80)]) + "\n  b: " + string(b[lo:min(len(b), i+80)])
}

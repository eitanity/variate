package suite

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/report"
)

// Golden outputs: the JSON report of every example and demo model at a
// fixed seed and length, frozen under testdata/golden. Any byte of
// difference fails, so every behaviour change shows up in review.
// Regenerate with: go test -run TestGolden -update  (or make golden).

var update = flag.Bool("update", false, "rewrite the golden files")

func TestGoldenReports(t *testing.T) {
	if err := os.MkdirAll("testdata/golden", 0o750); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, m := range corpus(t) {
		if m.path == "" {
			continue // random graphs are covered by their own tests
		}
		name := strings.TrimSuffix(strings.ReplaceAll(m.name, " ", "-"), ".yaml")
		if seen[name] {
			t.Fatalf("duplicate corpus name %s", name)
		}
		seen[name] = true
		// The exact transform is frozen for the compound model only; it
		// costs minutes on models pairing an uncertain rate with a
		// per-period probability.
		analytic := name == "compound"
		cfg := lengthFor(m.g, variate.Config{Periods: 1000, Seed: 11, BlockPeriods: 50, Transform: analytic})
		c, warnings, err := m.g.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		r, err := w.Run(context.Background(), 11)
		if err != nil {
			t.Fatal(err)
		}
		rep, err := report.New(c, w, r, true)
		if err != nil {
			t.Fatal(err)
		}
		rep.Warnings = warnings
		got := rep.JSON()
		w.Close()
		path := filepath.Join("testdata", "golden", name+".json")
		if *update {
			if err := os.WriteFile(path, got, 0o600); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(path) //nolint:gosec // repository file
		if err != nil {
			t.Fatalf("%s: no golden file; run with -update: %v", name, err)
		}
		if bytes.Equal(got, want) {
			continue
		}
		// The golden files are generated on amd64. Other architectures
		// agree to the last unit or two in values the math library
		// derives (Go's Exp, Log and Pow use CPU-specific instructions
		// on amd64 and arm64), so they are compared as numbers there.
		if runtime.GOARCH != "amd64" {
			if reason := reportsClose(want, got, goldenTolerance); reason != "" {
				t.Errorf("%s: report differs from the golden file beyond the cross-architecture tolerance: %s", name, reason)
			}
			continue
		}
		t.Errorf("%s: report differs from the golden file; if the change is intended, regenerate with -update\n%s", name, firstDiff(want, got))
	}
}

// goldenTolerance is the relative difference allowed between a golden
// report and one from another architecture: the transform's bounds, an
// FFT over millions of points, carry the math library's last-unit
// differences to about 1e-11.
const goldenTolerance = 1e-9

// reportsClose compares two JSON reports value by value, numbers to a
// relative tolerance, and names the first field that differs.
func reportsClose(want, got []byte, tol float64) string {
	var a, b any
	if err := json.Unmarshal(want, &a); err != nil {
		return "golden file: " + err.Error()
	}
	if err := json.Unmarshal(got, &b); err != nil {
		return "report: " + err.Error()
	}
	return closeValues(a, b, "", tol)
}

// roundingSpread reports a standard error or deviation that is rounding
// noise on both architectures: the spread of a quantity the same in every
// repeat up to its last bits, as a sum of spans covering a whole period.
func roundingSpread(path string, x, y float64) bool {
	spread := strings.HasSuffix(path, ".stderr") || strings.HasSuffix(path, ".sd") || strings.Contains(path, ".stderr[") || strings.Contains(path, ".sd[")
	return spread && math.Abs(x) < 1e-15 && math.Abs(y) < 1e-15
}

// closeScale is what a value's difference is measured against: the
// larger of the two, or 1 for an exceedance's bounds, probabilities from
// the transform whose rounding is a share of the whole distribution, so
// a tail's small probability does not magnify it.
func closeScale(path string, x, y float64) float64 {
	if strings.Contains(path, ".exceedance[") && (strings.HasSuffix(path, ".lower") || strings.HasSuffix(path, ".upper")) {
		return 1
	}
	return math.Max(math.Abs(x), math.Abs(y))
}

func closeValues(a, b any, path string, tol float64) string {
	switch x := a.(type) {
	case map[string]any:
		y, ok := b.(map[string]any)
		if !ok || len(x) != len(y) {
			return path + ": objects differ in shape"
		}
		for k, v := range x {
			if r := closeValues(v, y[k], path+"."+k, tol); r != "" {
				return r
			}
		}
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return path + ": arrays differ in length"
		}
		for i := range x {
			if r := closeValues(x[i], y[i], path+"["+strconv.Itoa(i)+"]", tol); r != "" {
				return r
			}
		}
	case float64:
		y, ok := b.(float64)
		if !ok {
			return path + ": number vs " + typeName(b)
		}
		if x != y && math.Abs(x-y) > tol*closeScale(path, x, y) && !roundingSpread(path, x, y) {
			return path + ": " + strconv.FormatFloat(x, 'g', -1, 64) + " vs " + strconv.FormatFloat(y, 'g', -1, 64)
		}
	default:
		if a != b {
			return path + ": values differ"
		}
	}
	return ""
}

func typeName(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case string:
		return "string"
	case bool:
		return "bool"
	case float64:
		return "number"
	}
	return "object"
}

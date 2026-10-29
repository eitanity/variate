package suite

import (
	"context"
	"strings"
	"testing"

	"code.eitanity.com/variate"
)

// The compile and workspace errors no other test reached.
func TestCompileErrorPaths(t *testing.T) {
	cases := map[string]struct {
		g    variate.Graph
		want string
	}{
		"duplicate source id": {variate.Graph{
			Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "a"}, {ID: "s", Frequency: C(1), To: "a"}},
			Nodes:   []variate.Node{{ID: "a"}},
		}, "empty or duplicate id"},
		"empty source id": {variate.Graph{
			Sources: []variate.Source{{ID: "", Frequency: C(1), To: "a"}},
			Nodes:   []variate.Node{{ID: "a"}},
		}, "empty or duplicate id"},
		"fixed probabilities exceed 1": {variate.Graph{
			Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "a"}},
			Nodes: []variate.Node{
				{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(0.7)}, {To: "c", P: C(0.7)}, {To: "d"}}},
				{ID: "b"}, {ID: "c"}, {ID: "d"},
			},
		}, "exceed 1"},
		"fixed probabilities not 1": {variate.Graph{
			Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "a"}},
			Nodes: []variate.Node{
				{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(0.7)}, {To: "c", P: C(0.7)}}},
				{ID: "b"}, {ID: "c"},
			},
		}, "sum to 1.4"},
	}
	for name, c := range cases {
		_, _, err := c.g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: want %q, got %v", name, c.want, err)
		}
	}
}

func TestWorkspaceConfigErrors(t *testing.T) {
	m, _, err := Diamond().Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	for name, cfg := range map[string]variate.Config{
		"negative periods":       {Periods: -1},
		"series without periods": {Series: true},
	} {
		if _, err := m.NewWorkspace(cfg); err == nil || !strings.Contains(err.Error(), "config:") {
			t.Errorf("%s: got %v", name, err)
		}
	}
	// More than 2^32 tasks: one-period blocks up to the default cap.
	m2, _, err := Diamond().Compile(variate.Config{MaxPeriods: 1 << 40, AllowUnconverged: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m2.NewWorkspace(variate.Config{MaxPeriods: 1 << 40, BlockPeriods: 1, AllowUnconverged: true}); err == nil || !strings.Contains(err.Error(), "2^32 tasks") {
		t.Errorf("too many tasks: got %v", err)
	}
}

// TestWorkspaceRecordFileCannotBeCreated: an unusable temp directory
// fails NewWorkspace naming the record file, and only when an observer
// asks for one.
func TestWorkspaceRecordFileCannotBeCreated(t *testing.T) {
	t.Setenv("TMPDIR", "/nonexistent/variate-tmp")
	cfg := variate.Config{Periods: 10, Observer: &variate.Counter{}}
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewWorkspace(cfg); err == nil || !strings.Contains(err.Error(), "record file") {
		t.Errorf("got %v", err)
	}
	cfg.Observer = nil
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatalf("without an observer: %v", err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 1); err != nil {
		t.Error(err)
	}
}

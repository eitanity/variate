package variate

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestFaultRecordFileClosed: a slot file that cannot be written fails the
// run naming the record file, and Close still removes the files.
func TestFaultRecordFileClosed(t *testing.T) {
	cfg := Config{Periods: 300, BlockPeriods: 1, Workers: 8, RecordBuffer: 8}
	m, _, err := diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Observer = NewCounter(m)
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(w.slots))
	for _, sl := range w.slots {
		names = append(names, sl.recFile.Name())
		if err := sl.recFile.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := w.Run(context.Background(), 1); err == nil || !strings.Contains(err.Error(), recordFileStage) {
		t.Errorf("got %v", err)
	}
	w.Close()
	for _, name := range names {
		if _, err := os.Stat(name); err == nil {
			t.Errorf("%s left behind", name)
		}
	}
}

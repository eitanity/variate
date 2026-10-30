package suite

import (
	"context"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"code.eitanity.com/variate"
)

// Fault injection: observers that fail, stop, panic or cancel at a
// seeded round, cancellation from outside at a seeded moment, record
// files that fail, and the scratch and record buffers at their limits,
// all with many workers and one-period blocks. After every fault the
// run ends in the documented state, the workspace runs again cleanly,
// and no goroutine or temp file is left behind.

// faultObserver misbehaves once it has seen k rounds.
type faultObserver struct {
	k      int
	kind   string // "error", "stop", "panic", "cancel"
	cancel context.CancelFunc
	seen   int
}

var errInjected = errors.New("injected")

func (f *faultObserver) Observe(*variate.Round) error {
	f.seen++
	if f.seen-1 != f.k {
		return nil
	}
	switch f.kind {
	case "error":
		return errInjected
	case "stop":
		return variate.ErrStop
	case "panic":
		var none variate.Observer
		return none.Observe(nil) // nil interface: panics
	case "cancel":
		f.cancel()
	}
	return nil
}

// leakGuard records goroutines and temp files before a test and checks
// nothing is left after Close.
type leakGuard struct {
	goroutines int
	files      int
}

func guard(t *testing.T) leakGuard {
	t.Helper()
	// Record files go to a directory of the test's own, so that other
	// processes writing theirs to the shared one cannot move the count.
	t.Setenv("TMPDIR", t.TempDir())
	return leakGuard{goroutines: runtime.NumGoroutine(), files: tempRecordFiles(t)}
}

func (g leakGuard) check(t *testing.T, what string) {
	t.Helper()
	// Goroutines park asynchronously after Close; allow a moment.
	for i := 0; i < 100 && runtime.NumGoroutine() > g.goroutines; i++ {
		time.Sleep(10 * time.Millisecond) //nolint:forbidigo // waiting for goroutines to exit
	}
	if n := runtime.NumGoroutine(); n > g.goroutines {
		t.Errorf("%s: %d goroutines before, %d after Close", what, g.goroutines, n)
	}
	if n := tempRecordFiles(t); n != g.files {
		t.Errorf("%s: %d record files before, %d after Close", what, g.files, n)
	}
}

func tempRecordFiles(t *testing.T) int {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(os.TempDir(), "variate-rounds-*"))
	if err != nil {
		t.Fatal(err)
	}
	return len(files)
}

// faultConfig: 32 workers, one-period blocks, small batches.
func faultConfig(periods int64, obs variate.Observer) variate.Config {
	return variate.Config{Periods: periods, BlockPeriods: 1, Batch: 4, Workers: 32, Observer: obs}
}

func TestFaultObserverAtRandomRounds(t *testing.T) {
	for seed := uint64(1); seed <= 8; seed++ {
		r := rand.New(rand.NewPCG(seed, 3))
		kind := []string{"error", "stop", "panic"}[seed%3]
		obs := &faultObserver{k: r.IntN(600), kind: kind}
		what := kind + " at round " + strconv.Itoa(obs.k)
		cfg := faultConfig(400, obs)
		m, _, err := Diamond().Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		g := guard(t)
		w, err := m.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		res, err := w.Run(context.Background(), seed)
		switch kind {
		case "error":
			if !errors.Is(err, errInjected) || !strings.Contains(err.Error(), "observer") {
				t.Errorf("%s: got %v", what, err)
			}
		case "panic":
			if err == nil || !strings.Contains(err.Error(), "panicked") {
				t.Errorf("%s: got %v", what, err)
			}
		case "stop":
			if err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			if !res.Stopped || res.Converged || res.Binding != "stopped" || obs.seen != obs.k+1 || res.Periods == 0 || res.Periods > 400 {
				t.Errorf("%s: stopped %v converged %v binding %s seen %d periods %d", what, res.Stopped, res.Converged, res.Binding, obs.seen, res.Periods)
			}
		}
		// Usable afterwards: a clean run delivers every round.
		obs.k, obs.seen = -1, 0
		res, err = w.Run(context.Background(), seed)
		if err != nil {
			t.Fatalf("%s: second run: %v", what, err)
		}
		var events uint64
		for _, e := range res.Events {
			events += e
		}
		if obs.seen != int(events) || res.Periods != 400 || res.Stopped { //nolint:gosec // small counts
			t.Errorf("%s: second run delivered %d of %d rounds, periods %d", what, obs.seen, events, res.Periods)
		}
		w.Close()
		g.check(t, what)
	}
}

func TestFaultCancelAtRandomPoints(t *testing.T) {
	for seed := uint64(1); seed <= 6; seed++ {
		r := rand.New(rand.NewPCG(seed, 4))
		ctx, cancel := context.WithCancel(context.Background())
		obs := &faultObserver{k: r.IntN(800), kind: "cancel", cancel: cancel}
		cfg := faultConfig(2000, obs)
		m, _, err := Diamond().Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		g := guard(t)
		w, err := m.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Run(ctx, seed); err == nil || !errors.Is(err, context.Canceled) {
			t.Errorf("cancel from the observer at %d: got %v", obs.k, err)
		}
		cancel()
		// From outside, at a seeded moment during a long run.
		ctx2, cancel2 := context.WithCancel(context.Background())
		delay := time.Duration(r.IntN(5000)) * time.Microsecond
		time.AfterFunc(delay, cancel2)
		obs.k = -1
		if _, err := w.Run(ctx2, seed); err == nil || !errors.Is(err, context.Canceled) {
			t.Errorf("cancel after %v: got %v", delay, err)
		}
		cancel2()
		if _, err := w.Run(context.Background(), seed); err != nil {
			t.Errorf("usable after cancel: %v", err)
		}
		w.Close()
		g.check(t, "cancel seed "+strconv.FormatUint(seed, 10))
	}
}

// TestFaultLimits: the period scratch at one record fails loudly; the
// record buffer at a few bytes spills to the slot file and every round
// still arrives, with 32 workers and one-period blocks.
func TestFaultLimits(t *testing.T) {
	g := guard(t)
	full := &faultObserver{k: -1}
	cfg := faultConfig(50, full)
	cfg.RoundScratch = 1
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Run(context.Background(), 1); err == nil || !strings.Contains(err.Error(), "round scratch full") {
		t.Errorf("scratch of one: got %v", err)
	}
	w.Close()
	spill := &faultObserver{k: -1}
	cfg = faultConfig(300, spill)
	cfg.RecordBuffer = 8
	m, _, err = Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err = m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := w.Run(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	var events uint64
	for _, e := range res.Events {
		events += e
	}
	if spill.seen != int(events) { //nolint:gosec // small counts
		t.Errorf("spilled run delivered %d of %d rounds", spill.seen, events)
	}
	w.Close()
	g.check(t, "limits")
}

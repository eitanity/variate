package variate

import (
	"context"
	"errors"
	"testing"
	"time"
)

// fakeClock advances only when slept on.
type fakeClock struct {
	now   time.Time
	slept time.Duration
}

func (c *fakeClock) Now() time.Time        { return c.now }
func (c *fakeClock) Sleep(d time.Duration) { c.now = c.now.Add(d); c.slept += d }

// cancelAfter cancels the run on its k-th round.
type cancelAfter struct {
	k, seen int
	cancel  context.CancelFunc
}

func (o *cancelAfter) Observe(*Round) error {
	o.seen++
	if o.seen == o.k {
		o.cancel()
	}
	return nil
}

// TestPacedCancel: a paced run under a fake clock is cancelled by the
// observer and ends promptly.
func TestPacedCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cfg := Config{Periods: 200, BlockPeriods: 5, Pace: time.Second, Workers: 4, Observer: &cancelAfter{k: 10, cancel: cancel}}
	m, _ := mustCompile(t, diamond(), cfg)
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	w.clock = &fakeClock{now: time.Unix(1000, 0)}
	done := make(chan error, 1)
	go func() { _, err := w.Run(ctx, 1); done <- err }()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("got %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("paced run did not end after cancellation")
	}
	cancel()
}

type stampingObserver struct {
	clock *fakeClock
	seen  []stamped
}

type stamped struct {
	arrival float64
	at      time.Time
}

func (o *stampingObserver) Observe(r *Round) error {
	o.seen = append(o.seen, stamped{r.Arrival, o.clock.Now()})
	return nil
}

// TestStreamPacing: with a fake clock, no round is delivered before
// start + Arrival·Pace.
func TestStreamPacing(t *testing.T) {
	fc := &fakeClock{now: time.Unix(1000, 0)}
	obs := &stampingObserver{clock: fc}
	cfg := Config{Periods: 20, BlockPeriods: 5, Pace: time.Second, Workers: 2, Observer: obs}
	m, _ := mustCompile(t, diamond(), cfg)
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	w.clock = fc
	start := fc.Now()
	if _, err := w.Run(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	if len(obs.seen) == 0 || fc.slept == 0 {
		t.Fatalf("delivered %d, slept %v", len(obs.seen), fc.slept)
	}
	for _, s := range obs.seen {
		if s.at.Before(start.Add(time.Duration(s.arrival * float64(time.Second)))) {
			t.Fatalf("round at %g delivered early at %v", s.arrival, s.at.Sub(start))
		}
	}
}

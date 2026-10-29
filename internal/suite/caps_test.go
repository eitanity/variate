package suite

import (
	"context"
	"strings"
	"testing"

	"code.eitanity.com/variate"
)

// TestCapErlangWork: a rate whose count mass function would be too long,
// constant or uncertain, is refused at compile with the rate named.
func TestCapErlangWork(t *testing.T) {
	g := counting(C(3e8), 2, nil)
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "mass function") {
		t.Errorf("constant rate: %v", err)
	}
	g = counting(Pert(5e6, 1e7, 2e7), 2, nil)
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "quadrature nodes") {
		t.Errorf("uncertain rate: %v", err)
	}
	if _, _, err := counting(C(1e5), 2, nil).Compile(variate.Config{}); err != nil {
		t.Errorf("a rate of 1e5 should compile: %v", err)
	}
}

// TestCapArrivalsPerPeriod: the expected-work model counts the arrivals
// a renewal or chain source draws beyond its rounds, so the plan's caps
// see them; and a source that draws more than the period cap fails the
// run with its name.
func TestCapArrivalsPerPeriod(t *testing.T) {
	g := counting(C(1000), 100, nil)
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := c.Analytic()
	if e.ArrivalsPerPeriod != 1000 || e.RoundsPerPeriod != 10 {
		t.Errorf("arrivals %g rounds %g", e.ArrivalsPerPeriod, e.RoundsPerPeriod)
	}
	ch := chainModel(variate.Chain{K: 1, N: 2, Fail: 3, Repair: 40})
	cc, _, _ := ch.Compile(variate.Config{})
	ec, _ := cc.Analytic()
	if !(ec.ArrivalsPerPeriod > ec.RoundsPerPeriod) {
		t.Errorf("chain arrivals %g should exceed its rounds %g", ec.ArrivalsPerPeriod, ec.RoundsPerPeriod)
	}
	// 2e7 arrivals in a period, above the cap, drawn for every 1000th.
	big := counting(C(2e7), 1000, nil)
	cfg := variate.Config{Periods: 2, Workers: 1}
	cb, _, err := big.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := cb.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 1); err == nil || !strings.Contains(err.Error(), "drew more than") {
		t.Errorf("run beyond the period cap: %v", err)
	}
}

// TestCapTransformBytes: a pass that would hold more than MaxBytes is
// refused with the size named, for the transform and the entries
// transform alike.
func TestCapTransformBytes(t *testing.T) {
	m := mustCompile(t, Diamond())
	tr, err := m.Transform(variate.TransformConfig{MaxBytes: 1 << 20})
	if err != nil || tr.Applicable || !strings.Contains(tr.Reason, "would hold") {
		t.Errorf("transform: applicable %v reason %q err %v", tr.Applicable, tr.Reason, err)
	}
	cnt, err := m.TransformEntries(variate.TransformConfig{MaxBytes: 1 << 12})
	if err != nil || cnt.Applicable || !strings.Contains(cnt.Reason, "would hold") {
		t.Errorf("entries: applicable %v reason %q err %v", cnt.Applicable, cnt.Reason, err)
	}
	ok, err := m.Transform(variate.TransformConfig{Tol: 1e-2})
	if err != nil || !ok.Applicable {
		t.Errorf("the default budget should allow the diamond: %v %q", err, ok.Reason)
	}
}

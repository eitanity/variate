package main

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// demoModel mirrors web/models/index.json.
type demoModel struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	File  string `json:"file"`
	Knobs []struct {
		Key   string  `json:"key"`
		At    string  `json:"at"`
		Min   float64 `json:"min"`
		Max   float64 `json:"max"`
		Value float64 `json:"value"`
	} `json:"knobs"`
}

// TestDemoModels: every demo model, with its knobs at their defaults
// and at their extremes, parses and compiles, and every placeholder has
// a knob.
func TestDemoModels(t *testing.T) {
	dir := filepath.Join("..", "..", "web", "models")
	data, err := os.ReadFile(filepath.Join(dir, "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	if len(models) == 0 {
		t.Fatal("no models")
	}
	for _, m := range models {
		text, err := os.ReadFile(filepath.Join(dir, m.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		for _, which := range []string{"value", "min", "max"} {
			yaml := string(text)
			for _, k := range m.Knobs {
				v := k.Value
				switch which {
				case "min":
					v = k.Min
				case "max":
					v = k.Max
				}
				yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(v, 'g', -1, 64))
			}
			if strings.Contains(yaml, "{{") {
				t.Errorf("%s: placeholder without a knob", m.ID)
			}
			model, err := config.Parse([]byte(yaml), "")
			if err != nil {
				t.Errorf("%s at %s: %v", m.ID, which, err)
				continue
			}
			if _, _, err := model.Graph.Compile(variate.Config{AllowUnconverged: true}); err != nil {
				t.Errorf("%s at %s: compile: %v", m.ID, which, err)
			}
		}
	}
}

// TestHeavyDemoInfiniteVariance: the heavy-tails demo with the open
// Pareto's alpha at its default of 2.5 has a finite variance; at its
// lowest, 1.1, the variance is infinite and the model warns of it.
func TestHeavyDemoInfiniteVariance(t *testing.T) {
	dir := filepath.Join("..", "..", "web", "models")
	data, err := os.ReadFile(filepath.Join(dir, "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	for _, m := range models {
		if m.ID != "heavy" {
			continue
		}
		text, err := os.ReadFile(filepath.Join(dir, m.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		for _, low := range []bool{false, true} {
			yaml := string(text)
			for _, k := range m.Knobs {
				v := k.Value
				if low && k.Key == "open_alpha" {
					v = k.Min
				}
				yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(v, 'g', -1, 64))
			}
			model, err := config.Parse([]byte(yaml), "")
			if err != nil {
				t.Fatal(err)
			}
			_, ws, err := model.Graph.Compile(variate.Config{AllowUnconverged: true})
			if err != nil {
				t.Fatal(err)
			}
			warned := false
			for _, w := range ws {
				warned = warned || (w.Code == "infinite-variance" && w.Subject == "open_loss:loss")
			}
			if warned != low {
				t.Errorf("open alpha at its lowest %v: infinite-variance warning %v", low, warned)
			}
		}
		return
	}
	t.Fatal("no heavy demo")
}

// TestBacklogDemoLongRun: the backlog demo with one crew and half an
// order a month, a load of two thirds, has exact long-run figures by
// Pollaczek–Khinchine; with two crews and lognormal work it has none.
func TestBacklogDemoLongRun(t *testing.T) {
	for _, crews := range []float64{1, 2} {
		g := demoGraph(t, "backlog", map[string]float64{"crews": crews, "orders": 0.5})
		m, _, err := g.Compile(variate.Config{Repeats: 10})
		if err != nil {
			t.Fatal(err)
		}
		lr := m.QueueLongRun(0)
		if (lr.Method == "Pollaczek–Khinchine") != (crews == 1) {
			t.Errorf("crews %g: long-run method %q, reason %q", crews, lr.Method, lr.Reason)
		}
	}
}

// demoGraph is demo id's model with its knobs at their defaults but for
// those set.
func demoGraph(t *testing.T, id string, set map[string]float64) variate.Graph {
	t.Helper()
	dir := filepath.Join("..", "..", "web", "models")
	data, err := os.ReadFile(filepath.Join(dir, "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	for _, m := range models {
		if m.ID != id {
			continue
		}
		text, err := os.ReadFile(filepath.Join(dir, m.File)) //nolint:gosec // repository file
		if err != nil {
			t.Fatal(err)
		}
		yaml := string(text)
		for _, k := range m.Knobs {
			v, ok := set[k.Key]
			if !ok {
				v = k.Value
			}
			yaml = strings.ReplaceAll(yaml, "{{"+k.Key+"}}", strconv.FormatFloat(v, 'g', -1, 64))
		}
		model, err := config.Parse([]byte(yaml), dir)
		if err != nil {
			t.Fatal(err)
		}
		return model.Graph
	}
	t.Fatalf("no demo %s", id)
	return variate.Graph{}
}

// TestDemoAtKnobs: a knob that is a value of a quantity, not of the
// model, names a quantity the model has, and the run reports the chance
// of reaching it at the knob's default; the budget knob of the pump
// station is one.
func TestDemoAtKnobs(t *testing.T) {
	found := false
	for _, m := range demoModels(t) {
		for _, k := range m.Knobs {
			if k.At == "" {
				continue
			}
			found = found || (m.ID == "pump_station" && k.Key == "budget")
			lv, err := config.ParseLevel(k.At + "=" + strconv.FormatFloat(k.Value, 'g', -1, 64))
			if err != nil {
				t.Fatal(err)
			}
			cfg := variate.Config{Periods: 200, Seed: 1, AllowUnconverged: true, At: []variate.Level{lv}}
			c, _, err := demoGraph(t, m.ID, nil).Compile(cfg)
			if err != nil {
				t.Fatalf("%s %s: %v", m.ID, k.Key, err)
			}
			w, err := c.NewWorkspace(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Run(context.Background(), 1); err != nil {
				t.Fatal(err)
			}
			var s variate.Summary
			w.Summarize(&s)
			w.Close()
			if len(s.At) != 1 || s.At[0].N != 200 {
				t.Errorf("%s %s: at %+v, want one value counted over 200 periods", m.ID, k.Key, s.At)
			}
		}
	}
	if !found {
		t.Error("the pump station has no budget knob")
	}
}

// demoModels reads web/models/index.json.
func demoModels(t *testing.T) []demoModel {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "web", "models", "index.json")) //nolint:gosec // repository file
	if err != nil {
		t.Fatal(err)
	}
	var models []demoModel
	if err := json.Unmarshal(data, &models); err != nil {
		t.Fatal(err)
	}
	return models
}

// TestDemoTailWhen: the page's tail chart asks the run what the periods
// at or above each of its return periods are made of; at the pump
// station's defaults the 1-in-100 periods are 1% of the run, at a level
// with an error, and the sources' shares of their total sum to 1, each
// with an error.
func TestDemoTailWhen(t *testing.T) {
	asked := []string{"total>=1/10", "total>=1/100", "total>=1/1000"}
	when := make([]variate.When, 0, len(asked))
	for _, s := range asked {
		c, err := config.ParseWhen(s)
		if err != nil {
			t.Fatal(err)
		}
		when = append(when, c)
	}
	cfg := variate.Config{Periods: 20000, Seed: 1, AllowUnconverged: true, When: when}
	c, _, err := demoGraph(t, "pump_station", nil).Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	if len(s.When) != 3 {
		t.Fatalf("%d conditions, want 3", len(s.When))
	}
	st := s.When[1]
	if st.Reason != "" || st.Met != 200 || !(st.Lower.StdErr > 0) {
		t.Fatalf("1 in 100: %d periods, level %v ± %v, %q", st.Met, st.Lower.X, st.Lower.StdErr, st.Reason)
	}
	sum := 0.0
	for _, p := range st.Parts {
		if p.Kind == "source" {
			sum += p.Share
			if !(p.ShareStdErr >= 0) {
				t.Errorf("%s: share error %v", p.Name, p.ShareStdErr)
			}
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sources' shares sum to %v", sum)
	}
}

// TestDemoTotals: the factory line's named totals are reported, cost
// is the period total, the hours stay out of it, and since the crew's
// cost and hours are both rates over the same days down, labour is
// the hours at the crew's cost per hour in every period; net is the
// revenue less the cost, and doubling the margin makes a losing year
// rarer; doubling the crew's hours a day doubles worked and leaves the
// total alone.
func TestDemoTotals(t *testing.T) {
	run := func(set map[string]float64) (*variate.Compiled, *variate.Result, variate.Summary) {
		cfg := variate.Config{Periods: 5000, Seed: 1, AllowUnconverged: true, Series: true}
		c, _, err := demoGraph(t, "line", set).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		r, err := w.Run(context.Background(), 1)
		if err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		return c, r, s
	}
	c, r, s := run(nil)
	if cost := s.Totals["cost"]; math.Abs(cost.Mean-s.Total.Mean) > 1e-12*s.Total.Mean || len(s.Totals) != 4 {
		t.Fatalf("totals %v, total %v", s.Totals, s.Total.Mean)
	}
	cat := map[string]int{}
	for i, name := range c.Categories() {
		cat[name] = i
	}
	net := r.Series.NamedColumn(3, make([]float64, r.Series.Periods()))
	for p := range r.Series.Periods() {
		hours, labour := r.Series.Category(cat["hours"], p)+r.Series.Category(cat["fitting"], p), r.Series.Category(cat["labour"], p)
		if math.Abs(labour-hours*1500/24) > 1e-6*math.Max(1, labour) {
			t.Fatalf("period %d: labour %v for %v hours", p, labour, hours)
		}
		if want := r.Series.Category(cat["revenue"], p) - r.Series.Total(p); math.Abs(net[p]-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Fatalf("period %d: net %v, revenue less cost %v", p, net[p], want)
		}
	}
	if len(s.AtMostZero) != 1 || !(s.AtMostZero[0].P > 0.05 && s.AtMostZero[0].P < 0.95) {
		t.Errorf("P(net ≤ 0) %+v", s.AtMostZero)
	}
	if _, _, s3 := run(map[string]float64{"margin": 8000}); !(s3.AtMostZero[0].P < s.AtMostZero[0].P/4) {
		t.Errorf("P(net ≤ 0) %v at a margin of 8000, %v at 4000", s3.AtMostZero[0].P, s.AtMostZero[0].P)
	}
	_, _, s2 := run(map[string]float64{"crew_hours": 48})
	crew, crew2 := s.Totals["worked"].Mean-s.ByCategory["fitting"].Mean, s2.Totals["worked"].Mean-s2.ByCategory["fitting"].Mean
	if math.Abs(s2.Total.Mean-s.Total.Mean) > 1e-12*s.Total.Mean || math.Abs(crew2-2*crew) > 1e-9*crew {
		t.Errorf("crew hours 48: total %v crew %v, against %v and %v", s2.Total.Mean, crew2, s.Total.Mean, crew)
	}
}

// TestDemoDiscount: the started team at 1% a month simulates the same
// requests as without a discount, at the same seed, and each month's
// cost, simulated and exact, is (1.01)^-month times the undiscounted.
func TestDemoDiscount(t *testing.T) {
	run := func(rate float64) (*variate.Result, *variate.Expectation) {
		cfg := variate.Config{Repeats: 2000, Seed: 1, AllowUnconverged: true}
		c, _, err := demoGraph(t, "startup", map[string]float64{"discount": rate}).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		e, err := c.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		r, err := c.Simulate(context.Background(), cfg)
		if err != nil {
			t.Fatal(err)
		}
		return r, e
	}
	plain, pe := run(0)
	disc, de := run(0.01)
	for k := range plain.ByPeriod.Periods() {
		f := math.Pow(1.01, -float64(k+1))
		got, want := disc.ByPeriod.Total(k).Mean, f*plain.ByPeriod.Total(k).Mean
		if math.Abs(got-want) > 1e-9*want || math.Abs(de.ByPeriod[k][0]-f*pe.ByPeriod[k][0]) > 1e-9*want {
			t.Errorf("month %d: %v, want %v; exact %v, want %v", k+1, got, want, de.ByPeriod[k][0], f*pe.ByPeriod[k][0])
		}
		if disc.ByPeriod.Entries(0, k).Mean != plain.ByPeriod.Entries(0, k).Mean {
			t.Errorf("month %d: the requests differ", k+1)
		}
	}
}

// TestDemoPrices: the factory line's prices leave the exact means of
// the parts and the revenue as they are, at mean 1, and a wider spread
// widens both and the correlation between them across the years.
func TestDemoPrices(t *testing.T) {
	run := func(sd float64) (*variate.Compiled, *variate.Expectation, *variate.Result) {
		cfg := variate.Config{Periods: 20000, Seed: 1, AllowUnconverged: true, Series: true}
		c, _, err := demoGraph(t, "line", map[string]float64{"prices": sd}).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		e, err := c.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		r, err := c.Simulate(context.Background(), cfg)
		if err != nil {
			t.Fatal(err)
		}
		return c, e, r
	}
	corr := func(c *variate.Compiled, r *variate.Result) (float64, float64) {
		cat := map[string]int{}
		for i, name := range c.Categories() {
			cat[name] = i
		}
		n := float64(r.Series.Periods())
		var sx, sy, sxx, syy, sxy float64
		for p := range r.Series.Periods() {
			x, y := r.Series.Category(cat["parts"], p), r.Series.Category(cat["revenue"], p)
			sx, sy, sxx, syy, sxy = sx+x, sy+y, sxx+x*x, syy+y*y, sxy+x*y
		}
		vx := sxx/n - sx*sx/n/n
		return (sxy/n - sx*sy/n/n) / math.Sqrt(vx*(syy/n-sy*sy/n/n)), math.Sqrt(vx)
	}
	cn, en, rn := run(0.01)
	cw, ew, rw := run(0.4)
	for _, name := range []string{"parts", "revenue"} {
		i := slices.Index(cn.Categories(), name)
		if math.Abs(en.MeanByCategory[i]-ew.MeanByCategory[i]) > 1e-9*en.MeanByCategory[i] {
			t.Errorf("%s: exact mean %v narrow, %v wide", name, en.MeanByCategory[i], ew.MeanByCategory[i])
		}
	}
	rhoN, sdN := corr(cn, rn)
	rhoW, sdW := corr(cw, rw)
	if !(rhoN < 0.05 && rhoW > 0.3 && sdW > 1.2*sdN) {
		t.Errorf("parts and revenue: correlation %v narrow, %v wide; parts sd %v and %v", rhoN, rhoW, sdN, sdW)
	}
}

// TestDemoInflation: one year's prices on every repair's labour and
// parts give the exact distribution at the browser's tolerance, mixed
// over the prices; a wider spread leaves the exact mean as it is and
// moves the 99th percentile out.
func TestDemoInflation(t *testing.T) {
	q99 := func(sd float64) (float64, float64) {
		c, _, err := demoGraph(t, "inflation", map[string]float64{"prices_sd": sd}).Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		tr, err := c.Transform(variate.TransformConfig{Tol: 0.01})
		if err != nil {
			t.Fatal(err)
		}
		if !tr.Applicable || tr.Passes < 2 {
			t.Fatalf("spread %v: applicable %v, %d passes: %s", sd, tr.Applicable, tr.Passes, tr.Reason)
		}
		e, _ := c.Analytic()
		_, hi := tr.Total.Quantile(0.99)
		return e.Mean, hi
	}
	mn, qn := q99(0.05)
	mw, qw := q99(0.3)
	if math.Abs(mn-mw) > 1e-9*mn || !(qw > 1.03*qn) {
		t.Errorf("mean %v narrow, %v wide; 99th percentile %v and %v", mn, mw, qn, qw)
	}
}

// TestDemoFitting: the factory line's fitting labour is its hours times
// 62.5, so more fitting hours raise the exact labour 62.5 times as much
// as the hours, and the simulated labour is within its error.
func TestDemoFitting(t *testing.T) {
	run := func(h float64) (*variate.Compiled, *variate.Expectation, variate.Summary) {
		cfg := variate.Config{Periods: 20000, Seed: 1, AllowUnconverged: true}
		c, _, err := demoGraph(t, "line", map[string]float64{"fitting": h}).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		e, err := c.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		if _, err := w.Run(context.Background(), 1); err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		return c, e, s
	}
	cl, el, _ := run(4)
	_, eh, sh := run(30)
	labour, hours := slices.Index(cl.Categories(), "labour"), slices.Index(cl.Categories(), "fitting")
	dl := eh.MeanByCategory[labour] - el.MeanByCategory[labour]
	dh := eh.MeanByCategory[hours] - el.MeanByCategory[hours]
	if !(dh > 0) || math.Abs(dl/dh-62.5) > 1e-9 {
		t.Errorf("labour rises %v for %v hours, want 62.5 each", dl, dh)
	}
	if got := sh.ByCategory["labour"]; math.Abs(got.Mean-eh.MeanByCategory[labour]) > 4*got.StdErr {
		t.Errorf("labour %v ± %v, exact %v", got.Mean, got.StdErr, eh.MeanByCategory[labour])
	}
}

// TestDemoTogether: the factory line's spares and downtime moving
// together leave the exact means as they are and widen the simulated
// year's cost; the exact standard deviation, with the downtime's
// holding, is the simulated within its error apart, at ranks 0.5 and
// 0.95, where the downtime crossing the year's end bends the figures in
// the group's normal, and in lockstep.
func TestDemoTogether(t *testing.T) {
	run := func(rank float64) (*variate.Expectation, variate.Summary) {
		cfg := variate.Config{Periods: 20000, Seed: 1, AllowUnconverged: true}
		c, _, err := demoGraph(t, "line", map[string]float64{"together": rank}).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		e, err := c.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		if _, err := w.Run(context.Background(), 1); err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		return e, s
	}
	e0, s0 := run(0)
	eh, sh := run(0.5)
	en, sn := run(0.95)
	e1, s1 := run(1)
	for _, c := range []struct {
		e *variate.Expectation
		s variate.Summary
	}{{e0, s0}, {eh, sh}, {en, sn}, {e1, s1}} {
		sd, se := math.Sqrt(c.e.Var), c.s.Total.StdDev/math.Sqrt(2*float64(c.s.Total.N))
		if !(math.Abs(c.s.Total.StdDev-sd) <= 4*se) {
			t.Errorf("exact sd %v, simulated %v ± %v", sd, c.s.Total.StdDev, se)
		}
	}
	if math.Abs(e1.Mean-e0.Mean) > 1e-9*e0.Mean {
		t.Errorf("exact mean %v in lockstep, %v apart", e1.Mean, e0.Mean)
	}
	if !(s1.Total.StdDev > 1.03*s0.Total.StdDev) {
		t.Errorf("sd %v in lockstep, %v apart", s1.Total.StdDev, s0.Total.StdDev)
	}
	t.Logf("sd %v in lockstep, %v apart", s1.Total.StdDev, s0.Total.StdDev)
}

// TestDemoLongRun: the long-run backlog with one crew and half an order
// a month starts its line exactly in its long-run state, so its mean
// wait over 200,000 months is Pollaczek–Khinchine's within four
// standard errors, and the start looks back some months; with two crews
// it has no closed form but runs.
func TestDemoLongRun(t *testing.T) {
	for _, crews := range []float64{1, 2} {
		cfg := variate.Config{Periods: 200_000, Seed: 1, AllowUnconverged: true}
		c, _, err := demoGraph(t, "longrun", map[string]float64{"crews": crews, "orders": 0.5}).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		w.Close()
		q := s.Queues["backlog"]
		lr := q.LongRun
		if (lr.Method == "Pollaczek–Khinchine") != (crews == 1) {
			t.Errorf("crews %g: long-run method %q, reason %q", crews, lr.Method, lr.Reason)
		}
		if crews == 1 && math.Abs(q.Wait.Mean-lr.Wait) > 4*q.Wait.StdErr {
			t.Errorf("one crew: mean wait %v ± %v, Pollaczek–Khinchine %v", q.Wait.Mean, q.Wait.StdErr, lr.Wait)
		}
		if !(q.LookBack > 0) {
			t.Errorf("crews %g: look-back %v", crews, q.LookBack)
		}
		t.Logf("crews %g: wait %.4f ± %.4f, long run %.4f, look-back %.1f", crews, q.Wait.Mean, q.Wait.StdErr, lr.Wait, q.LookBack)
	}
}

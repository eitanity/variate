package suite

import (
	"math"
	"math/rand/v2"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// Engine invariants: a refusal is never a self-check, a random graph
// compiles or fails with a documented error, and no expectation is NaN
// where the expectation is defined as a number.

// compileErrorForm is the shape of every documented compile error: the
// kind of thing at fault (node, source, outcome, function, probability,
// holding, frequency, value, rate, graph, config, watch), its quoted
// name, and the message.
var compileErrorForm = regexp.MustCompile(`^[a-z ]+ "[^"]*": .+`)

// heavyGraph is a random DAG whose values are heavy-tailed: lognormals
// with p90/median of 4 to 20, Pareto with α in (1, 2], and GPD.
func heavyGraph(t *testing.T, seed uint64) variate.Graph {
	t.Helper()
	g := genGraph(t, seed, false)
	r := rand.New(rand.NewPCG(seed, 99))
	for i := range g.Nodes {
		for f := range g.Nodes[i].Functions {
			fn := &g.Nodes[i].Functions[f]
			fn.Scope = variate.PerEvent
			var d dist.Distribution
			var err error
			switch r.IntN(3) {
			case 0:
				median := math.Pow(10, 2*r.Float64())
				d, err = dist.NewLognormalFromMedianP90(median, median*(4+16*r.Float64()))
			case 1:
				d, err = dist.NewPareto(1+9*r.Float64(), 1.05+0.95*r.Float64())
			default:
				d, err = dist.NewGPD(0, 1+9*r.Float64(), 0.3+0.6*r.Float64())
			}
			if err != nil {
				t.Fatal(err)
			}
			fn.Model = d
		}
	}
	return g
}

// transformable adapts a random graph to the transform's domain: no
// period-scoped values, at most three period-scoped probabilities.
func transformable(g variate.Graph) variate.Graph {
	scoped := 0
	for i := range g.Nodes {
		for f := range g.Nodes[i].Functions {
			g.Nodes[i].Functions[f].Scope = variate.PerEvent
		}
		for j := range g.Nodes[i].Outcomes {
			o := &g.Nodes[i].Outcomes[j]
			if _, ok := o.P.(dist.Beta); ok {
				scoped++
				if scoped > 3 {
					o.P = C(o.P.Mean())
				}
			}
		}
	}
	return g
}

// invariantCorpus: the differential corpus plus random and heavy-tailed
// random graphs; seeds says how many of each. Discretising a heavy-tailed
// function costs about 0.2 s per pipeline, so the fast suite keeps two
// seeds and the stat suite takes more.
func invariantCorpus(t *testing.T, seeds int) []corpusModel {
	t.Helper()
	out := corpus(t)
	for seed := uint64(1); seed <= uint64(seeds); seed++ { //nolint:gosec // a small count
		out = append(out, corpusModel{name: "heavy" + strconv.FormatUint(seed, 10), g: heavyGraph(t, seed)})
		out = append(out, corpusModel{name: "random" + strconv.FormatUint(seed+10, 10), g: genGraph(t, seed+10, false)})
	}
	return out
}

// TestInvariantNoSelfCheckRefusal: Transform and TransformEntries never
// refuse with a self-check reason. Every check runs on the first grid of
// the first pass, so a loose tolerance on the smallest grid exercises
// them all at a fraction of a real transform's cost.
func TestInvariantNoSelfCheckRefusal(t *testing.T) {
	SkipUnderRace(t)
	// The sequential models pair an uncertain rate with a per-period
	// probability, which costs the transform about a minute each on the
	// smallest grid; the stat run covers them.
	var fast []corpusModel
	for _, m := range invariantCorpus(t, 1) {
		if !strings.Contains(m.name, "sequential") {
			fast = append(fast, m)
		}
	}
	checkNoSelfCheckRefusal(t, fast)
}

func checkNoSelfCheckRefusal(t *testing.T, models []corpusModel) {
	t.Helper()
	cfg := variate.TransformConfig{Tol: 0.25, MaxGrid: 1 << 16}
	for _, m := range models {
		start := time.Now() //nolint:forbidigo // timing a test
		g := transformable(m.g)
		c, _, err := g.Compile(variate.Config{AllowUnconverged: true})
		if err != nil {
			t.Fatalf("%s: %v", m.name, err)
		}
		tr, err := c.Transform(cfg)
		if err != nil {
			t.Fatalf("%s: %v", m.name, err)
		}
		if !tr.Applicable && strings.HasPrefix(tr.Reason, "self-check") {
			t.Errorf("%s: transform self-check refusal: %s", m.name, tr.Reason)
		}
		cs, err := c.TransformEntries(cfg)
		if err != nil {
			t.Fatalf("%s: %v", m.name, err)
		}
		if !cs.Applicable && strings.HasPrefix(cs.Reason, "self-check") {
			t.Errorf("%s: entries self-check refusal: %s", m.name, cs.Reason)
		}
		t.Logf("%s: %v; transform %v (%s); entries %v (%s)", m.name, time.Since(start).Round(time.Millisecond), tr.Applicable, tr.Reason, cs.Applicable, cs.Reason)
	}
}

// TestInvariantCompileOrDocumentedError: graphs from the fuzz builder
// compile or fail in the documented form, and a compiled model's
// expectations are numbers wherever they are defined as one.
func TestInvariantCompileOrDocumentedError(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 5))
	compiled, refused := 0, 0
	for i := 0; i < 2000; i++ {
		data := make([]byte, 4+r.IntN(60))
		for j := range data {
			data[j] = byte(r.IntN(256)) //nolint:gosec // below 256
		}
		g := graphFromBytes(data)
		c, _, err := g.Compile(variate.Config{AllowUnconverged: true})
		if err != nil {
			refused++
			if !compileErrorForm.MatchString(err.Error()) {
				t.Errorf("undocumented error form: %v", err)
			}
			continue
		}
		compiled++
		e, err := c.Analytic()
		if err != nil {
			t.Fatalf("analytic after compile: %v", err)
		}
		checkNumbers(t, "fuzz graph "+strconv.Itoa(i), c, e)
	}
	t.Logf("%d compiled, %d refused", compiled, refused)
	if compiled < 100 {
		t.Errorf("only %d of 2000 graphs compiled; the generator is too strict to test anything", compiled)
	}
	for _, m := range invariantCorpus(t, 4) {
		c, e := exactOf(t, m.g)
		checkNumbers(t, m.name, c, e)
	}
}

// checkNumbers: the fields that are always numbers must not be NaN; Var
// and its breakdowns may be NaN only with delays, a repair source or
// above the pair budget, and are then NaN together; P(T = 0) only with a
// repair source.
func checkNumbers(t *testing.T, what string, c *variate.Compiled, e *variate.Expectation) {
	t.Helper()
	bad := func(name string, x float64) {
		if math.IsNaN(x) {
			t.Errorf("%s: %s is NaN", what, name)
		}
	}
	bad("Mean", e.Mean)
	bad("StepsPerPeriod", e.StepsPerPeriod)
	bad("RoundsPerPeriod", e.RoundsPerPeriod)
	bad("DrawsMean", e.DrawsMean)
	if !c.HasRepair() {
		bad("PZero", e.PZero())
	}
	for v, x := range e.EntriesPer {
		bad("EntriesPer "+c.NodeIDs()[v], x)
		bad("Contrib "+c.NodeIDs()[v], e.Contrib[v])
	}
	for j, x := range e.OutcomeRate {
		bad("OutcomeRate "+c.OutcomeNames()[j], x)
	}
	for k := range e.Reach {
		bad("MeanBySource", e.MeanBySource[k])
		for v, x := range e.Reach[k] {
			bad("Reach "+c.SourceIDs()[k]+" "+c.NodeIDs()[v], x)
		}
	}
	for i, x := range e.MeanByCategory {
		bad("MeanByCategory "+c.Categories()[i], x)
	}
	if !c.HasDelays() && !c.HasRepair() && math.IsNaN(e.Var) {
		t.Errorf("%s: Var is NaN without delays", what)
	}
	if c.HasRepair() && (!math.IsNaN(e.Var) || !math.IsNaN(e.PZero())) {
		t.Errorf("%s: a repair source cannot claim Var %g or PZero %g", what, e.Var, e.PZero())
	}
	if !c.HasRepair() && (e.PZero() < 0 || e.PZero() > 1+1e-12) {
		t.Errorf("%s: PZero %g outside [0, 1]", what, e.PZero())
	}
}

// TestInvariantSequentialPZero: a sequential node whose outcome
// probabilities sum to just above 1 in floating point, behind a frequency
// heavy enough that e^(λ·ε) overflows at its far nodes, still has
// P(T = 0) = 1 when nothing is booked.
func TestInvariantSequentialPZero(t *testing.T) {
	par, _ := dist.NewPareto(0.01, 1.2)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: par, To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Mode: variate.Sequential, Outcomes: []variate.Outcome{{To: "b", P: C(0.08)}, {To: "b", P: C(0.45)}, {To: "b"}}},
			{ID: "b"},
		},
	}
	c, e := exactOf(t, g)
	checkNumbers(t, "sequential", c, e)
	if e.PZero() != 1 {
		t.Errorf("PZero %v, want 1", e.PZero())
	}
}

// TestInvariantInfiniteMomentsStayNumbers: the zero-times-infinity cases
// the fuzzer found. A per-time rate with infinite variance on a node with
// no holding time books nothing, and a source whose rate is 0 or an
// outcome whose probability is 0 contributes nothing, however heavy the
// values behind it; the variance must be a number or +Inf, never NaN.
func TestInvariantInfiniteMomentsStayNumbers(t *testing.T) {
	par, _ := dist.NewPareto(1, 1.5) // infinite variance
	rateOnLeafless := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{{Name: "r", Category: "k", Model: par, Basis: variate.PerTime}}, Next: "b"},
			{ID: "b", Functions: []variate.Function{Fn("x", "k", C(3))}},
		},
	}
	zeroRate := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(0), To: "a"}, {ID: "t", Frequency: C(1), To: "b"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("x", "k", par)}},
			{ID: "b", Functions: []variate.Function{Fn("y", "k", C(3))}},
		},
	}
	exp1, _ := dist.NewExponential(1)
	zeroOutcome := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("x", "k", exp1)}, Outcomes: []variate.Outcome{{To: "h", P: C(0)}, {To: "b"}}},
			{ID: "h", Functions: []variate.Function{Fn("y", "k", par)}},
			{ID: "b"},
		},
	}
	for name, g := range map[string]variate.Graph{"rate without holding": rateOnLeafless, "zero-rate source": zeroRate, "zero-probability outcome": zeroOutcome} {
		c, e := exactOf(t, g)
		checkNumbers(t, name, c, e)
		if name == "rate without holding" && (e.Var != 18 || e.Mean != 6) {
			t.Errorf("%s: mean %g var %g, want 6 and 18", name, e.Mean, e.Var)
		}
		if name == "zero-rate source" && (e.Var != 9 || e.Mean != 3) {
			t.Errorf("%s: mean %g var %g, want 3 and 9", name, e.Mean, e.Var)
		}
		if name == "zero-probability outcome" && (e.Var != 4 || e.Mean != 2) {
			t.Errorf("%s: mean %g var %g, want 2 and 4", name, e.Mean, e.Var)
		}
	}
}

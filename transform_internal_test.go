package variate

import (
	"context"
	"math"
	"math/big"
	"math/cmplx"
	"os"
	"sort"
	"strings"
	"testing"

	"code.eitanity.com/variate/dist"
)

func compiled(t *testing.T, g Graph) *Compiled {
	t.Helper()
	m, _, err := g.Compile(Config{})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func compoundOf(freq, sev dist.Distribution) Graph {
	return Graph{
		Sources: []Source{{ID: "s", Frequency: freq, To: "a"}},
		Nodes:   []Node{{ID: "a", Functions: []Function{fn("x", "k", sev)}}},
	}
}

// TestTransformTiltChoice: the chosen tilt costs no more than
// θ·M·h = 0, 10, 20 and 40.
func TestTransformTiltChoice(t *testing.T) {
	e, _ := dist.NewExponential(2)
	tr := newTransformer(compiled(t, compoundOf(c(3), e)), TransformConfig{})
	const m = 1 << 16
	h, _ := tr.firstSpan(math.Sqrt(tr.tailP/(1-tr.tailP)), m)
	kStar := m / 2
	theta := tr.chooseTilt(h, m, kStar)
	cost := func(theta float64) float64 {
		return tr.aliasBound(h, m, theta) + tr.roundoffAt(h, m, theta, 0, kStar)
	}
	best := cost(theta)
	for _, tmh := range []float64{0, 10, 20, 40} {
		if other := cost(tmh / (float64(m) * h)); other < best*(1-1e-9) {
			t.Errorf("θMh=%g costs %g, chosen θMh=%g costs %g", tmh, other, theta*float64(m)*h, best)
		}
	}
	if theta == 0 {
		t.Error("no tilt chosen")
	}
}

// TestTransformGridSizing: the first pass's x* is at or above the
// simulated TailP quantile, and the second pass's span divides the
// first's exactly.
func TestTransformGridSizing(t *testing.T) {
	skipUnderRace(t)
	g := diamond()
	m := compiled(t, g)
	tr := newTransformer(m, TransformConfig{})
	cc := math.Sqrt(tr.tailP / (1 - tr.tailP))
	first, reason := tr.converge(minGrid, func(n int) (float64, bool) { return tr.firstSpan(cc, n) }, 0)
	if first == nil {
		t.Fatal(reason)
	}
	_, xStar := first.bounds(first.total).Quantile(tr.tailP)
	out := &Transformed{}
	if err := tr.run(out); err != nil || !out.Applicable {
		t.Fatalf("%v %s", err, out.Reason)
	}
	ratio := first.h / out.H
	if j := math.Log2(ratio); j < 0 || j != math.Floor(j) {
		t.Errorf("spans %g and %g are not nested", first.h, out.H)
	}
	const periods = 20000
	w, err := m.NewWorkspace(Config{Periods: periods, Series: true})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	totals := r.Series.TotalColumn(make([]float64, periods))
	sort.Float64s(totals)
	q := totals[int(tr.tailP*periods)]
	// The closed-form Cantelli x*(h) of the first pass bounds the quantile
	// outright; the pass's own quantile bound must also hold, by a score
	// test on the periods beyond it.
	cantelli := tr.e.Mean + cc*math.Sqrt(tr.e.Var) + first.h*(tr.e.DrawsMean+cc*math.Sqrt(tr.e.DrawsVar))
	if cantelli < q {
		t.Errorf("Cantelli x* %g below the simulated %g quantile %g", cantelli, tr.tailP, q)
	}
	beyond := 0
	for _, v := range totals {
		if v >= xStar {
			beyond++
		}
	}
	if z := (float64(beyond) - periods*(1-tr.tailP)) / math.Sqrt(periods*(1-tr.tailP)*tr.tailP); z > 4.5 {
		t.Errorf("%d periods beyond x* %g, z=%.2f", beyond, xStar, z)
	}
	_, hi := out.Total.Quantile(tr.tailP)
	if hi > xStar {
		t.Errorf("second pass quantile bound %g above x* %g", hi, xStar)
	}
}

// TestTransformBranchCut: the Gamma closed form with non-integer shape
// agrees with direct quadrature over λ to 1e-12 absolutely.
func TestTransformBranchCut(t *testing.T) {
	gamma, _ := dist.NewGamma(2.5, 0.8)
	e, _ := dist.NewExponential(1)
	tr := newTransformer(compiled(t, compoundOf(gamma, e)), TransformConfig{})
	const n = 256
	gc := make([]complex128, n)
	for j := range gc {
		// Points over the whole unit disc, including the negative axis.
		rad := float64(j%16) / 15
		gc[j] = cmplx.Rect(rad, 2*math.Pi*float64(j)/n)
	}
	closed := make([]complex128, n)
	if _, reason := tr.compound(0, gc, closed, 1); reason != "" {
		t.Fatal(reason)
	}
	x := make([]float64, 1<<13)
	w := make([]float64, 1<<13)
	quantileNodes(gamma, x, w)
	direct := make([]complex128, n)
	evalMixture(x, w, gc, direct, 1)
	for j := range gc {
		if d := cmplx.Abs(closed[j] - direct[j]); d > 1e-12 {
			t.Errorf("w=%v: closed %v direct %v diff %g", gc[j], closed[j], direct[j], d)
		}
	}
}

// TestTransformLambdaQuadrature: a wide PERT frequency converges within
// the cap, and ε_q bounds the difference from a 4·n reference at every
// point. The long suite widens the span to 2000.
func TestTransformLambdaQuadrature(t *testing.T) {
	span := 100.0
	switch {
	case os.Getenv("VARIATE_STAT") == "1":
		span = 2000
	case !testing.Short():
		span = 400
	}
	pertF, err := dist.NewPERT(0, span/4, span, dist.DefaultPERTShape)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := dist.NewExponential(1)
	tr := newTransformer(compiled(t, compoundOf(pertF, e)), TransformConfig{})
	const n = 1 << 12
	gc := unitRoots(n)
	for j := range gc {
		gc[j] *= 0.999
	}
	out := make([]complex128, n)
	eps, reason := tr.compound(0, gc, out, 1)
	if reason != "" {
		t.Fatal(reason)
	}
	if tr.quadN < quadStart {
		t.Errorf("only %d nodes", tr.quadN)
	}
	x := make([]float64, 4*tr.quadN)
	w := make([]float64, 4*tr.quadN)
	pertF.Nodes(x, w)
	ref := make([]complex128, n)
	evalMixture(x, w, gc, ref, 1)
	// Summing the reference's 4·n terms of modulus at most 1 rounds by up
	// to 4·n·u, which the round-off estimate carries rather than ε_q.
	if d := maxAbsDiff(out, ref); d > eps+float64(len(x))*unitRoundoff {
		t.Errorf("difference from the 4n reference %g exceeds ε_q %g", d, eps)
	}
}

// TestTransformQuadratureWorkCap: a lognormal frequency with σ = 4 needs
// thousands of nodes; on a large grid that exceeds the work cap and is
// refused with that reason, after only the sample phase.
func TestTransformQuadratureWorkCap(t *testing.T) {
	ln, _ := dist.NewLognormal(-8, 4)
	e, _ := dist.NewExponential(1)
	tr := newTransformer(compiled(t, compoundOf(ln, e)), TransformConfig{})
	const n = 1 << 19
	gc := make([]complex128, n)
	for j := range gc {
		gc[j] = cmplx.Rect(0.999, 2*math.Pi*float64(j)/n)
	}
	_, reason := tr.compound(0, gc, make([]complex128, n), 1)
	if !strings.Contains(reason, "work cap") {
		t.Errorf("reason %q", reason)
	}
}

// TestTransformWrapAround: with a Pareto severity (α = 1.5, infinite
// variance), the result on a grid 16 times longer differs by at most
// α_up, and without the tilt the wrap-around alone would be refused.
func TestTransformWrapAround(t *testing.T) {
	par, _ := dist.NewPareto(1, 1.5)
	tr := newTransformer(compiled(t, compoundOf(c(2), par)), TransformConfig{})
	const m = 1 << 12
	h, _ := tr.firstSpan(0, m)
	short, reason := tr.pass(h, m, 0)
	if short == nil {
		t.Fatal(reason)
	}
	long, reason := tr.pass(h, 16*m, 0)
	if long == nil {
		t.Fatal(reason)
	}
	// The longer grid wraps less; the shorter grid's upper curve may
	// fall below it by at most its own α (and the round-off of both).
	for k := 0; k < m; k++ {
		if d := long.total.up[k] - short.total.up[k]; d > short.alias+short.rho[k]+long.rho[k] {
			t.Errorf("k=%d: long %g short %g differ by %g > α %g", k, long.total.up[k], short.total.up[k], d, short.alias)
		}
	}
	if untilted := tr.aliasBound(h, m, 0); untilted <= tr.cfg.Tol*(1-tr.tailP) {
		t.Errorf("α without tilt %g would not be refused", untilted)
	}
	zero := *short
	zero.alias = tr.aliasBound(h, m, 0)
	if zero.withinTol(float64(m)*h/2, tr.cfg.Tol, tr.tailP) {
		t.Error("the untilted bracket passed the tolerance")
	}
}

// ---- extended precision reference pipeline ----

const bigPrec = 128

func bf(x float64) *big.Float { return new(big.Float).SetPrec(bigPrec).SetFloat64(x) }

func bAdd(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(bigPrec).Add(a, b) }
func bSub(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(bigPrec).Sub(a, b) }
func bMul(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(bigPrec).Mul(a, b) }
func bQuo(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(bigPrec).Quo(a, b) }

// bigAtanInv is atan(1/n) by its series.
func bigAtanInv(n int64) *big.Float {
	sum := bf(0)
	term := bQuo(bf(1), bf(float64(n)))
	n2 := bf(float64(n * n))
	for k := 0; k < 200; k++ {
		part := bQuo(term, bf(float64(2*k+1)))
		if k%2 == 1 {
			part.Neg(part)
		}
		sum = bAdd(sum, part)
		term = bQuo(term, n2)
	}
	return sum
}

// bigPi is Machin's formula.
func bigPi() *big.Float {
	return bSub(bMul(bf(16), bigAtanInv(5)), bMul(bf(4), bigAtanInv(239)))
}

// bigExp halves the argument into (−1/16, 1/16), sums the series and
// squares back.
func bigExp(x *big.Float) *big.Float {
	y := new(big.Float).SetPrec(bigPrec).Set(x)
	s := 0
	for ax, _ := y.Float64(); math.Abs(ax) > 1.0/16; ax, _ = y.Float64() {
		y = bQuo(y, bf(2))
		s++
	}
	sum := bf(1)
	term := bf(1)
	for k := 1; k < 40; k++ {
		term = bQuo(bMul(term, y), bf(float64(k)))
		sum = bAdd(sum, term)
	}
	for ; s > 0; s-- {
		sum = bMul(sum, sum)
	}
	return sum
}

// bigSinCos halves the angle into (−1/16, 1/16), sums the series and
// doubles back.
func bigSinCos(x *big.Float) (sin, cos *big.Float) {
	y := new(big.Float).SetPrec(bigPrec).Set(x)
	s := 0
	for ax, _ := y.Float64(); math.Abs(ax) > 1.0/16; ax, _ = y.Float64() {
		y = bQuo(y, bf(2))
		s++
	}
	y2 := bMul(y, y)
	sin, cos = bf(0), bf(0)
	ts, tc := new(big.Float).Set(y), bf(1)
	for k := 0; k < 30; k++ {
		sin = bAdd(sin, ts)
		cos = bAdd(cos, tc)
		ts = bQuo(bMul(ts, y2), bf(float64((2*k+2)*(2*k+3))))
		ts.Neg(ts)
		tc = bQuo(bMul(tc, y2), bf(float64((2*k+1)*(2*k+2))))
		tc.Neg(tc)
	}
	for ; s > 0; s-- {
		sin, cos = bMul(bMul(bf(2), sin), cos), bSub(bMul(cos, cos), bMul(sin, sin))
	}
	return sin, cos
}

type bigC struct{ re, im *big.Float }

func cAdd(a, b bigC) bigC { return bigC{bAdd(a.re, b.re), bAdd(a.im, b.im)} }
func cSub(a, b bigC) bigC { return bigC{bSub(a.re, b.re), bSub(a.im, b.im)} }
func cMul(a, b bigC) bigC {
	return bigC{bSub(bMul(a.re, b.re), bMul(a.im, b.im)), bAdd(bMul(a.re, b.im), bMul(a.im, b.re))}
}

// cExp is exp(z).
func cExp(z bigC) bigC {
	mag := bigExp(z.re)
	s, co := bigSinCos(z.im)
	return bigC{bMul(mag, co), bMul(mag, s)}
}

// bigFFT is the forward transform Σ_k x_k e^(−2πi·jk/M) (sign −1) or the
// unnormalised inverse (sign +1), iterative radix 2.
func bigFFT(x []bigC, sign float64) []bigC {
	m := len(x)
	out := make([]bigC, m)
	bits := 0
	for 1<<bits < m {
		bits++
	}
	for i := range x {
		rev := 0
		for b := 0; b < bits; b++ {
			if i&(1<<b) != 0 {
				rev |= 1 << (bits - 1 - b)
			}
		}
		out[rev] = x[i]
	}
	twoPi := bMul(bf(2), bigPi())
	for size := 2; size <= m; size *= 2 {
		half := size / 2
		tw := make([]bigC, half)
		for k := range tw {
			s, co := bigSinCos(bQuo(bMul(twoPi, bf(sign*float64(k))), bf(float64(size))))
			tw[k] = bigC{co, s}
		}
		for start := 0; start < m; start += size {
			for k := 0; k < half; k++ {
				a := out[start+k]
				b := cMul(out[start+k+half], tw[k])
				out[start+k] = cAdd(a, b)
				out[start+k+half] = cSub(a, b)
			}
		}
	}
	return out
}

// bigExceedance recomputes the upper pipeline of a compound-Poisson
// model with a constant rate in extended precision: the same float64
// lattice masses and fold schedule, then tilt, FFT, P_N, inverse FFT,
// untilt and the cumulative sum.
func bigExceedance(tr *transformer, h float64, m int, theta, lam float64) []float64 {
	d := tr.m.pools[tr.m.fns[0].pool]
	_, hi := d.Support()
	r := math.Exp(-theta * h)
	thetaH := bf(-theta * h)
	tilted := make([]bigC, m)
	for k := range tilted {
		tilted[k] = bigC{bf(0), bf(0)}
	}
	rm := 1.0
	for n := 0; n < maxFolds; n++ {
		base := float64(n * m)
		mass := 0.0
		prev := cdfAt(d, base*h, h, true)
		for k := 0; k < m; k++ {
			x := (base + float64(k)) * h
			if x > hi+h {
				break
			}
			next := cdfAt(d, x+h, h, true)
			f := math.Max(0, next-prev)
			prev = next
			mass += f
			if f > 0 {
				weight := bMul(bf(f), bigExp(bMul(thetaH, bf(base+float64(k)))))
				tilted[k].re = bAdd(tilted[k].re, weight)
			}
		}
		rm *= math.Pow(r, float64(m))
		dropped := survival(d, (base+float64(m))*h)
		if rm*dropped < unitRoundoff || mass == 0 && n > 0 {
			break
		}
	}
	gf := bigFFT(tilted, -1)
	lamB := bf(lam)
	for j := range gf {
		gf[j] = cExp(bigC{bMul(lamB, bSub(gf[j].re, bf(1))), bMul(lamB, gf[j].im)})
	}
	masses := bigFFT(gf, 1)
	ec := make([]float64, m)
	cdf := bf(0)
	mB := bf(float64(m))
	for k := range ec {
		v, _ := bSub(bf(1), cdf).Float64()
		ec[k] = v
		mk := bQuo(bQuo(masses[k].re, mB), bigExp(bMul(thetaH, bf(float64(k)))))
		cdf = bAdd(cdf, mk)
	}
	return ec
}

// TestTransformRoundoffExtendedPrecision: for grids up to 2^14 the float64
// pipeline is within ρ(x) of the extended-precision one at every point.
func TestTransformRoundoffExtendedPrecision(t *testing.T) {
	skipUnderRace(t)
	e, _ := dist.NewExponential(2)
	ln, _ := dist.NewLognormal(1, 0.5)
	p, _ := dist.NewPERT(1, 2, 4, dist.DefaultPERTShape)
	models := []struct {
		name string
		lam  float64
		sev  dist.Distribution
	}{{"exponential", 3, e}, {"lognormal", 2, ln}, {"pert", 5, p}}
	grids := []int{1 << 10, 1 << 12}
	if !testing.Short() {
		grids = append(grids, 1<<14)
	}
	for _, mdl := range models {
		tr := newTransformer(compiled(t, compoundOf(c(mdl.lam), mdl.sev)), TransformConfig{})
		for _, m := range grids {
			h, ok := tr.firstSpan(math.Sqrt(tr.tailP/(1-tr.tailP)), m)
			if !ok {
				t.Fatalf("%s: no span at %d", mdl.name, m)
			}
			theta := tr.chooseTilt(h, m, m/2)
			up, reason := tr.newPipeline(h, m, math.Exp(-theta*h), true)
			if reason != "" {
				t.Fatal(reason)
			}
			g, reason := up.graph()
			if reason != "" {
				t.Fatal(reason)
			}
			ec := up.invert(g.total)
			ref := bigExceedance(tr, h, m, theta, mdl.lam)
			worst, worstRatio := 0.0, 0.0
			for k := range ec {
				d := math.Abs(ec[k] - ref[k])
				rho := tr.roundoffAt(h, m, theta, 0, k)
				worst = math.Max(worst, d)
				worstRatio = math.Max(worstRatio, d/rho)
				if d > rho {
					t.Errorf("%s M=%d k=%d: float64 %.17g reference %.17g differ by %g > ρ %g", mdl.name, m, k, ec[k], ref[k], d, rho)
					break
				}
			}
			t.Logf("%s M=%d θMh=%.2f: worst error %.3g, worst error/ρ %.3g", mdl.name, m, theta*float64(m)*h, worst, worstRatio)
		}
	}
}

// TestUnitRootsSymmetric: the root table is exactly conjugate-symmetric,
// so mirroring a quadrature's second half equals computing it.
func TestUnitRootsSymmetric(t *testing.T) {
	m := 1 << 10
	roots := unitRoots(m)
	for j := 1; j < m; j++ {
		if roots[m-j] != cmplx.Conj(roots[j]) {
			t.Fatalf("root %d is not the conjugate of root %d", m-j, j)
		}
	}
	x := []float64{0.5, 1.5, 3}
	w := []float64{0.2, 0.5, 0.3}
	gc := make([]complex128, m)
	for j := range gc {
		gc[j] = 1 - 0.3*(1-roots[j])
	}
	full := make([]complex128, m)
	evalMixture(x, w, gc, full, 1)
	half := make([]complex128, m)
	evalMixture(x, w, gc[:m/2+1], half[:m/2+1], 1)
	mirror(half)
	for j := range full {
		if full[j] != half[j] {
			t.Fatalf("point %d: %v vs %v", j, full[j], half[j])
		}
	}
}

// TestAtomUnit: the atoms are the positive constants and the caps a
// capped value has mass at, not the upper end of a bounded value.
func TestAtomUnit(t *testing.T) {
	pert, _ := dist.NewPERT(1, 2, 3, dist.DefaultPERTShape)
	u, _ := dist.NewUniform(1, 3.5)
	e, _ := dist.NewExponential(1)
	capped, _ := dist.NewCapped(e, 6)
	g := compoundOf(c(1), c(4))
	g.Nodes[0].Functions = append(g.Nodes[0].Functions, fn("p", "k", pert), fn("u", "k", u))
	if got := compiled(t, g).atomUnit(); got != 4 {
		t.Errorf("constant 4 beside bounded values: unit %g", got)
	}
	g.Nodes[0].Functions = append(g.Nodes[0].Functions, fn("cap", "k", capped))
	if got := compiled(t, g).atomUnit(); got != 2 {
		t.Errorf("constant 4 and a cap at 6: unit %g", got)
	}
}

// skipUnderRace skips a test of single-threaded code under the race
// detector, where its instrumentation costs minutes and detects nothing;
// make race says so with VARIATE_RACE=1.
func skipUnderRace(t *testing.T) {
	t.Helper()
	if os.Getenv("VARIATE_RACE") == "1" {
		t.Skip("single-threaded; skipped under the race detector")
	}
}

// TestTransformOrder: the recursion's order puts each node after the
// nodes its outcomes lead to and right after them where it can, counts
// the uses of each node, and the recursion holds only the sources'
// targets when it returns.
func TestTransformOrder(t *testing.T) {
	g := Graph{
		Sources: []Source{
			{ID: "a", Frequency: c(1), To: "a1"},
			{ID: "b", Frequency: c(1), To: "b1"},
		},
		Nodes: []Node{
			{ID: "a1", Outcomes: []Outcome{{To: "a2"}}},
			{ID: "a2", Outcomes: []Outcome{{To: "shared"}}},
			{ID: "b1", Outcomes: []Outcome{{To: "shared", P: c(0.5)}, {To: "b2"}}},
			{ID: "b2"},
			{ID: "shared", Functions: []Function{fn("f", "c", c(1))}},
			{ID: "orphan"},
		},
	}
	m := compiled(t, g)
	order, uses := m.transformOrder()
	ids := make([]string, len(order))
	for i, v := range order {
		ids[i] = m.nodeIDs[v]
	}
	if want := "shared a2 a1 b2 b1"; strings.Join(ids, " ") != want {
		t.Errorf("order %q, want %q", strings.Join(ids, " "), want)
	}
	byID := map[string]int32{}
	for v, id := range m.nodeIDs {
		byID[id] = uses[v]
	}
	if byID["shared"] != 2 || byID["a1"] != 1 || byID["b1"] != 1 || byID["b2"] != 1 || byID["orphan"] != 0 {
		t.Errorf("uses %v", byID)
	}
	tr := newTransformer(m, TransformConfig{})
	p, reason := tr.newPipeline(1, minGrid, 1, false)
	if reason != "" {
		t.Fatal(reason)
	}
	total, reason := p.recurse(-1, nil)
	if reason != "" {
		t.Fatal(reason)
	}
	if total[0] != 1 {
		t.Errorf("generating function at z = 1 is %v", total[0])
	}
	// Every node's buffer came back to the pool, and few were ever needed:
	// the chain a1 → a2 → shared is the deepest, plus the scratch.
	if len(p.free) == 0 || len(p.free) > 4 {
		t.Errorf("%d buffers in the pool after the recursion", len(p.free))
	}
}

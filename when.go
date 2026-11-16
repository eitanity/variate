package variate

import (
	"errors"
	"math"
	"slices"
	"strconv"
)

// A condition of Config.When is one or two half-lines of its quantity:
// the periods at or above From, less those at or above To. A half-line
// at a value sums, as the run goes, every quantity over the periods it
// holds; one at a return period keeps the rows of the run's worst periods
// and, at the end, finds its level among them and sums those at or
// above it. Every figure is a ratio of those sums, so its error is the
// delta method's: over the periods, or, with delays, over the batch
// groups.

// cwhen is one of Config.When compiled: the row of its quantity, and the
// half-lines of its bounds, to −1 without a To.
type cwhen struct {
	When
	row      int
	from, to int
}

// whenSpare is the rows a return-period half-line keeps beyond twice the
// share of periods its level holds, so that early in a run, and for a
// level held by few periods, the periods at its level are all kept.
const whenSpare = 64

// compileWhens checks Config.When: each quantity the model has, each
// bound a finite value or a return period of at least 1, and a To above
// its From where both are values or both return periods.
func (m *Compiled) compileWhens(cond []When) error {
	l := newLayout(m)
	m.whens = make([]cwhen, 0, len(cond))
	for _, c := range cond {
		subject := c.Kind
		if c.Name != "" {
			subject += ":" + c.Name
		}
		row, ok := m.quantityRow(l, c.Kind, c.Name)
		if !ok {
			return modelError("config", "When", "no quantity "+subject+"; name one as kind:name, as category:labour or trigger:overrun, or total")
		}
		if err := checkBound(c.From); err != nil {
			return modelError("config", "When", subject+": "+err.Error())
		}
		if t := c.To; t != nil {
			if err := checkBound(*t); err != nil {
				return modelError("config", "When", subject+": "+err.Error())
			}
			if (t.OneIn == 0) == (c.From.OneIn == 0) && !boundAbove(*t, c.From) {
				return modelError("config", "When", subject+": the upper bound must be above the lower")
			}
		}
		m.whens = append(m.whens, cwhen{When: c, row: row, from: -1, to: -1})
	}
	return nil
}

func checkBound(b Bound) error {
	switch {
	case b.OneIn != 0 && (!(b.OneIn >= 1) || math.IsInf(b.OneIn, 1)):
		return errors.New("a return period is 1/N with N at least 1")
	case b.OneIn == 0 && (math.IsNaN(b.X) || math.IsInf(b.X, 0)):
		return errors.New("the value must be a finite number")
	}
	return nil
}

// boundAbove reports whether a is above b, both values or both return
// periods.
func boundAbove(a, b Bound) bool {
	if a.OneIn != 0 {
		return a.OneIn > b.OneIn
	}
	return a.X > b.X
}

// Whens returns Config.When as compiled.
func (m *Compiled) Whens() []When {
	out := make([]When, len(m.whens))
	for i, c := range m.whens {
		out[i] = c.When
	}
	return out
}

// condSums are a set of periods' sums over the columns [0, cw) of the
// row: their count, each column's sum, sum of squares and sum of
// products with the total; with delays, per batch group, the count and
// each column's sum.
type condSums struct {
	n          float64
	s1, s2, xt []float64
	gn, gs     []float64 // [group], [group×cw]; nil without delays
}

func newCondSums(cw int, groups bool) condSums {
	s := condSums{s1: make([]float64, cw), s2: make([]float64, cw), xt: make([]float64, cw)}
	if groups {
		s.gn = make([]float64, batchGroups)
		s.gs = make([]float64, batchGroups*cw)
	}
	return s
}

func (s *condSums) reset() {
	s.n = 0
	clear(s.s1)
	clear(s.s2)
	clear(s.xt)
	clear(s.gn)
	clear(s.gs)
}

// add adds one period's row, in batch group g.
func (s *condSums) add(row []float64, g int) {
	s.n++
	t := row[0]
	cw := len(s.s1)
	for j, q := range row[:cw] {
		s.s1[j] += q
		s.s2[j] += q * q
		s.xt[j] += q * t
	}
	if s.gn != nil {
		s.gn[g]++
		gs := s.gs[g*cw : (g+1)*cw]
		for j, q := range row[:cw] {
			gs[j] += q
		}
	}
}

// set makes s a − b, or a alone where b is nil.
func (s *condSums) set(a, b *condSums) {
	s.n = a.n
	copy(s.s1, a.s1)
	copy(s.s2, a.s2)
	copy(s.xt, a.xt)
	copy(s.gn, a.gn)
	copy(s.gs, a.gs)
	if b == nil {
		return
	}
	s.n -= b.n
	sub(s.s1, b.s1)
	sub(s.s2, b.s2)
	sub(s.xt, b.xt)
	sub(s.gn, b.gn)
	sub(s.gs, b.gs)
}

func sub(a, b []float64) {
	for i, x := range b {
		a[i] -= x
	}
}

func addTo(a, b []float64) {
	for i, x := range b {
		a[i] += x
	}
}

// half is a half-line of a condition: the periods whose quantity at row
// is at least x, or, with oneIn set, at least the level the run estimates
// is reached once in oneIn periods.
type half struct {
	row    int
	x      float64
	oneIn  float64
	slotAt int       // a value's sums within a slot's
	sums   condSums  // a value's sums over the run; a return period's at its level
	heap   rowHeap   // a return period's worst periods
	level  Estimate  // the level, with its error for a return period
	g      []float64 // a return period's columns' means at its level
	reason string    // why a return period's level has no figures
}

// allocWhen lays out the half-lines of Config.When.
func (w *Workspace) allocWhen() {
	cw := w.l.layerAt
	w.halves = w.halves[:0]
	w.slotWhen = 0
	add := func(row int, b Bound) int {
		h := half{row: row, x: b.X, oneIn: b.OneIn, sums: newCondSums(cw, w.delays)}
		if b.OneIn == 0 {
			h.slotAt = w.slotWhen
			w.slotWhen += 1 + 3*cw
		} else {
			h.heap = rowHeap{cw: cw, floor: math.Inf(-1)}
			h.g = make([]float64, cw)
			w.anyOneIn = true
		}
		w.halves = append(w.halves, h)
		return len(w.halves) - 1
	}
	for i := range w.m.whens {
		c := &w.m.whens[i]
		c.from = add(c.row, c.From)
		if c.To != nil {
			c.to = add(c.row, *c.To)
		}
	}
	if len(w.halves) > 0 && w.delays {
		w.whenGN = make([]int64, batchGroups)
	}
	w.whenPartsTmpl = w.whenPartNames()
}

// sizeWhen shares the memory MaxMemory leaves once the workspace is
// planned among the return-period half-lines' rows.
func (w *Workspace) sizeWhen(limit int64) {
	n := int64(0)
	for i := range w.halves {
		if w.halves[i].oneIn > 0 {
			n++
		}
	}
	if n == 0 {
		return
	}
	per := max(limit-w.footprint, 0) / n / (8*int64(w.l.layerAt) + 24)
	for i := range w.halves {
		w.halves[i].heap.max = int(min(per, math.MaxInt32))
	}
}

// whenBytes is what Config.When adds to a slot, and to the workspace
// besides the rows its return periods keep.
func (w *Workspace) whenBytes() (slotB, globalB int64) {
	if len(w.m.whens) == 0 {
		return 0, 0
	}
	cw := int64(w.l.layerAt)
	slotB = 8 * (int64(w.slotWhen) + cw)
	globalB = int64(len(w.halves)) * 8 * (3*cw + batchGroups*(cw+1))
	return slotB, globalB
}

// resetWhen empties the conditions' sums and kept rows before a run.
func (w *Workspace) resetWhen() {
	w.whenN = 0
	clear(w.whenGN)
	for i := range w.halves {
		h := &w.halves[i]
		h.sums.reset()
		h.heap.reset()
	}
}

// addWhen adds a closed row to the sums of every value half-line it
// reaches; in a started run, it keeps the horizon's row for the return
// periods, which in steady state read the block's rows at the merge.
func (w *Workspace) addWhen(sl *slot, row []float64) {
	if len(w.halves) == 0 {
		return
	}
	cw := w.l.layerAt
	if w.started && w.anyOneIn {
		copy(sl.hrow, row[:cw])
	}
	t := row[0]
	for i := range w.halves {
		h := &w.halves[i]
		if h.oneIn > 0 || row[h.row] < h.x {
			continue
		}
		s := sl.when[h.slotAt : h.slotAt+1+3*cw]
		s[0]++
		for j, q := range row[:cw] {
			s[1+j] += q
			s[1+cw+j] += q * q
			s[1+2*cw+j] += q * t
		}
	}
}

// mergeWhen adds a finalised slot to the conditions, in task order: a
// value half-line's sums, and a return period's candidates for its worst
// periods.
func (w *Workspace) mergeWhen(sl *slot) {
	if len(w.halves) == 0 {
		return
	}
	g := int(sl.task % batchGroups)
	w.whenN += sl.closed
	if w.whenGN != nil {
		w.whenGN[g] += sl.closed
	}
	cw := w.l.layerAt
	for i := range w.halves {
		h := &w.halves[i]
		if h.oneIn > 0 {
			w.offerRows(h, sl)
			continue
		}
		s := sl.when[h.slotAt : h.slotAt+1+3*cw]
		h.sums.n += s[0]
		addTo(h.sums.s1, s[1:1+cw])
		addTo(h.sums.s2, s[1+cw:1+2*cw])
		addTo(h.sums.xt, s[1+2*cw:1+3*cw])
		if h.sums.gn != nil {
			h.sums.gn[g] += s[0]
			addTo(h.sums.gs[g*cw:(g+1)*cw], s[1:1+cw])
		}
	}
}

// offerRows offers a slot's closed rows to a return period's worst
// periods, which keep twice the share of the periods so far that its
// level holds, and whenSpare more.
func (w *Workspace) offerRows(h *half, sl *slot) {
	target := int(math.Ceil(2*float64(w.whenN)/h.oneIn)) + whenSpare
	cw := w.l.layerAt
	// bound: the slot's closed rows
	for r := int64(0); r < sl.closed; r++ {
		row := sl.hrow
		if !w.started {
			row = sl.rowAt(w.l, sl.a+r)[:cw]
		}
		h.heap.offer(row[h.row], sl.task, row, target)
	}
}

// rowHeap keeps the rows of the periods with the largest values of a
// quantity, at most max: a min-heap of store places by value, so the
// least kept is the one a larger period replaces. floor is the largest
// value it let go.
type rowHeap struct {
	cw, max int
	idx     []int32   // the heap of store places
	key     []float64 // by store place: the quantity
	task    []int64   // by store place: the task, for its batch group
	rows    []float64 // by store place: the row's columns [0, cw)
	floor   float64
	order   []int32 // scratch: the store places by value, descending
}

func (h *rowHeap) reset() {
	h.idx, h.key, h.task, h.rows = h.idx[:0], h.key[:0], h.task[:0], h.rows[:0]
	h.floor = math.Inf(-1)
}

// offer keeps the row if fewer than target rows are kept, or if it is
// above the least kept, which it then replaces.
func (h *rowHeap) offer(q float64, task int64, row []float64, target int) {
	if len(h.idx) < min(target, h.max) {
		h.idx = append(h.idx, int32(len(h.key))) //nolint:gosec // at most max, an int32
		h.key = append(h.key, q)
		h.task = append(h.task, task)
		h.rows = append(h.rows, row...)
		h.up(len(h.idx) - 1)
		return
	}
	if len(h.idx) == 0 || q <= h.key[h.idx[0]] {
		h.floor = math.Max(h.floor, q)
		return
	}
	s := h.idx[0]
	h.floor = math.Max(h.floor, h.key[s])
	h.key[s], h.task[s] = q, task
	copy(h.rows[int(s)*h.cw:], row)
	h.down(0)
}

func (h *rowHeap) less(i, j int) bool { return h.key[h.idx[i]] < h.key[h.idx[j]] }

func (h *rowHeap) up(i int) {
	// bound: the heap's depth
	for i > 0 {
		p := (i - 1) / 2
		if !h.less(i, p) {
			return
		}
		h.idx[i], h.idx[p] = h.idx[p], h.idx[i]
		i = p
	}
}

func (h *rowHeap) down(i int) {
	n := len(h.idx)
	// bound: the heap's depth
	for {
		c := 2*i + 1
		if c >= n {
			return
		}
		if c+1 < n && h.less(c+1, c) {
			c++
		}
		if !h.less(c, i) {
			return
		}
		h.idx[i], h.idx[c] = h.idx[c], h.idx[i]
		i = c
	}
}

// sorted orders the kept rows by value, largest first.
func (h *rowHeap) sorted() []int32 {
	h.order = append(h.order[:0], h.idx...)
	slices.SortFunc(h.order, func(a, b int32) int {
		switch ka, kb := h.key[a], h.key[b]; {
		case ka > kb:
			return -1
		case ka < kb:
			return 1
		}
		return 0
	})
	return h.order
}

// Estimate is a level with its standard error: 0 for a value given, and
// for a return period's level from the order statistics about it.
type Estimate struct{ X, StdErr float64 }

// resolveLevel finds a return period's level among its kept rows, the
// ⌈n/oneIn⌉-th largest of the n periods, and sums the rows at or above
// it. Its error is the width of the order statistics' 95% interval
// about that rank over 2·1.96, widened with delays by the batch groups'
// inflation of the share at or above it. Where the run let go a period
// at or above the level, or kept too few, it gives the reason.
func (w *Workspace) resolveLevel(h *half) {
	h.sums.reset()
	h.reason = ""
	n := float64(w.whenN)
	m := int(math.Ceil(n / h.oneIn))
	order := h.heap.sorted()
	if m < 1 || m > len(order) {
		h.level = Estimate{X: math.NaN(), StdErr: math.NaN()}
		h.reason = "the rows kept do not reach the level: raise MaxMemory"
		return
	}
	k := &h.heap
	x := k.key[order[m-1]]
	h.x = x
	if !(x > k.floor) {
		h.level = Estimate{X: x, StdErr: math.NaN()}
		h.reason = "periods at the level were not all kept: raise MaxMemory"
		return
	}
	cw := w.l.layerAt
	// bound: the rows kept
	for _, s := range order {
		if k.key[s] < x {
			break
		}
		h.sums.add(k.rows[int(s)*cw:(int(s)+1)*cw], int(k.task[s]%batchGroups))
	}
	p := 1 / h.oneIn
	d := quantileZ * w.whenInflation(&h.sums) * math.Sqrt(n*p*(1-p))
	hi, lo := int(math.Floor(float64(m)-d)), int(math.Ceil(float64(m)+d))
	se := math.NaN()
	if lo <= len(order) && k.key[order[lo-1]] > k.floor {
		se = (k.key[order[max(hi, 1)-1]] - k.key[order[lo-1]]) / (2 * quantileZ)
	}
	h.level = Estimate{X: x, StdErr: se}
	h.atLevel(order, max(hi, 1), min(lo, len(order)))
}

// atLevel sets g to the mean of each column over the kept rows of ranks
// lo to hi, about the level: the column's mean where the quantity is at
// it.
func (h *half) atLevel(order []int32, lo, hi int) {
	clear(h.g)
	k := &h.heap
	for _, s := range order[lo-1 : hi] {
		addTo(h.g, k.rows[int(s)*k.cw:(int(s)+1)*k.cw])
	}
	for j := range h.g {
		h.g[j] /= float64(hi - lo + 1)
	}
}

// whenInflation is the ratio of the batch-means error of the share of
// periods in s to its binomial error: 1 without delays, NaN where the
// groups cannot measure it.
func (w *Workspace) whenInflation(s *condSums) float64 {
	if s.gn == nil {
		return 1
	}
	n := float64(w.whenN)
	p := s.n / n
	if p == 0 || p == 1 {
		return 1
	}
	batch := ratioErr(w.whenGN, func(g int) float64 { return s.gn[g] }, func(g int) float64 { return float64(w.whenGN[g]) })
	return batch / math.Sqrt(p*(1-p)/n)
}

// WhenStat is one of Config.When's conditions: of the N periods
// (repeats, in a started run), the Met that meet it, their share P, and
// the level of each bound, a return period's estimated with its error;
// and over those periods each quantity's mean and share of their total.
// A return period's figures are at its estimated level, and their
// errors take in the level's own. Reason says why there are no figures,
// "" where there are.
type WhenStat struct {
	When
	N, Met    int64
	P, StdErr float64
	Lower     Estimate // the level of From
	Upper     Estimate // the level of To, with one
	Reason    string
	Parts     []WhenPart
}

// WhenPart is a quantity over the periods meeting a condition: its mean
// in them and its share of their total, each with its standard error;
// the share is NaN for the total and for a node's entries.
type WhenPart struct {
	Kind, Name         string
	Mean, MeanStdErr   float64
	Share, ShareStdErr float64
}

// whenPartNames lists the quantities a condition reports: the total,
// the categories and sources, each node's entries, and the contribution
// of each node that books, the breakdowns as the run keeps them.
func (w *Workspace) whenPartNames() []WhenPart {
	parts := []WhenPart{{Kind: totalSubject}}
	for _, kind := range [4]string{kindCategory, kindSource, kindEntries, kindContrib} {
		names, _ := w.m.quantityNames(w.l, kind)
		for v, n := range names {
			if kind != kindContrib || w.hasFn[v] {
				parts = append(parts, WhenPart{Kind: kind, Name: n})
			}
		}
	}
	return parts
}

// summarizeWhen fills each condition's figures.
func (w *Workspace) summarizeWhen(dst *Summary) {
	dst.When = dst.When[:0]
	if len(w.m.whens) == 0 {
		return
	}
	for i := range w.halves {
		if h := &w.halves[i]; h.oneIn > 0 {
			w.resolveLevel(h)
		} else {
			h.level = Estimate{X: h.x}
		}
	}
	for i := range w.m.whens {
		dst.When = append(dst.When, w.whenStat(&w.m.whens[i]))
	}
}

// whenStat is a condition's figures, from its half-lines' sums.
func (w *Workspace) whenStat(c *cwhen) WhenStat {
	from := &w.halves[c.from]
	st := WhenStat{When: c.When, N: w.whenN, Lower: from.level}
	st.Reason = from.reason
	b := band{n: float64(w.whenN), runs: w.whenGN, ga: from.adjust()}
	if c.to >= 0 {
		h := &w.halves[c.to]
		st.Upper = h.level
		if st.Reason == "" {
			st.Reason = h.reason
		}
		b.b, b.gb = &h.sums, h.adjust()
	}
	if st.Reason != "" || w.whenN == 0 {
		st.P, st.StdErr = math.NaN(), math.NaN()
		return st
	}
	if b.b != nil && !(st.Upper.X > st.Lower.X) {
		st.StdErr = math.NaN() // the band is empty
		return st
	}
	s := &w.whenScratch
	if s.s1 == nil {
		*s = newCondSums(w.l.layerAt, w.delays)
	}
	s.set(&from.sums, b.b)
	b.s = s
	st.Met = int64(s.n)
	st.P, st.StdErr = b.chance()
	st.Parts = w.whenParts(&b)
	return st
}

// adjust is a return period's columns' means at its level, which its
// figures' errors take in, or nil for a value, and for a level the
// periods fix exactly, a value then.
func (h *half) adjust() []float64 {
	if h.oneIn == 0 || h.level.StdErr == 0 {
		return nil
	}
	return h.g
}

// whenParts is each quantity's mean and share over the band's periods.
func (w *Workspace) whenParts(b *band) []WhenPart {
	parts := slices.Clone(w.whenPartsTmpl)
	for i := range parts {
		p := &parts[i]
		j := 0
		if p.Kind != totalSubject {
			j, _ = w.m.quantityRow(w.l, p.Kind, p.Name)
		}
		p.Mean, p.MeanStdErr = b.ratio(j, false)
		p.Share, p.ShareStdErr = math.NaN(), math.NaN()
		if p.Kind == kindCategory || p.Kind == kindSource || p.Kind == kindContrib {
			p.Share, p.ShareStdErr = b.ratio(j, true)
		}
	}
	return parts
}

// band is a condition's periods, A less B for its bounds' half-lines A
// and B (B nil without a To), s their sums, for its figures' errors. A
// figure is a ratio of sums over the band; its error is the delta
// method's on the figure linearised in the periods, Y per period, over
// the n periods, or with delays over the batch groups that ran, runs.
// A bound at a return period fixes the count at or above its level,
// which moves with the level: the linearised figure then carries, for
// each such bound, the column's mean at its level, ga and gb (nil for a
// value), as the error of a mean over a tail beyond a quantile does.
type band struct {
	s, b   *condSums
	ga, gb []float64
	n      float64
	runs   []int64
}

// ratio is column j's mean over the band, or with share its share of
// the total, and its error. For the ratio r = Σq/D, D the count or Σt,
// Y = 1[A∖B](q − σt − μ − kA) + 1[B](kB − kA), with σ = r and μ = 0 for
// a share, σ = 0 and μ = r for a mean, and k = g_j − μ − σ·g_0 at a
// return period's bound, 0 at a value's; the error is √(ΣY² −
// (ΣY)²/n)/|D|.
func (b *band) ratio(j int, share bool) (r, se float64) {
	s := b.s
	d, sigma := s.n, 0.0
	if share {
		d = s.s1[0]
	}
	if d == 0 {
		return math.NaN(), math.NaN()
	}
	r = s.s1[j] / d
	mu := r
	if share {
		mu, sigma = 0, r
	}
	ka, kb := corr(b.ga, j, mu, sigma), corr(b.gb, j, mu, sigma)
	c, nb := mu+ka, 0.0
	if b.b != nil {
		nb = b.b.n
	}
	if s.gn != nil {
		cw := len(s.s1)
		y := func(g int) float64 {
			v := s.gs[g*cw+j] - sigma*s.gs[g*cw] - c*s.gn[g]
			if b.b != nil {
				v += b.b.gn[g] * (kb - ka)
			}
			return v
		}
		return r, ratioErr(b.runs, y, b.periods) * b.n / math.Abs(d)
	}
	sumY2 := s.s2[j] - 2*sigma*s.xt[j] + sigma*sigma*s.s2[0] - 2*c*(s.s1[j]-sigma*s.s1[0]) + c*c*s.n + nb*(kb-ka)*(kb-ka)
	sumY := s.s1[j] - sigma*s.s1[0] - c*s.n + nb*(kb-ka)
	return r, math.Sqrt(math.Max(0, sumY2-sumY*sumY/b.n)) / math.Abs(d)
}

// chance is the share of periods in the band and its error, with Y =
// 1[A∖B](1 − kA) + 1[B](kB − kA), k 1 at a return period's bound and 0
// at a value's: at return periods alone the count is fixed, and the
// error 0.
func (b *band) chance() (p, se float64) {
	s := b.s
	p = s.n / b.n
	ka, kb, nb := 0.0, 0.0, 0.0
	if b.ga != nil {
		ka = 1
	}
	if b.b != nil {
		nb = b.b.n
		if b.gb != nil {
			kb = 1
		}
	}
	if s.gn != nil {
		y := func(g int) float64 {
			v := s.gn[g] * (1 - ka)
			if b.b != nil {
				v += b.b.gn[g] * (kb - ka)
			}
			return v
		}
		return p, ratioErr(b.runs, y, b.periods)
	}
	sumY2 := s.n*(1-ka)*(1-ka) + nb*(kb-ka)*(kb-ka)
	sumY := s.n*(1-ka) + nb*(kb-ka)
	return p, math.Sqrt(math.Max(0, sumY2-sumY*sumY/b.n)) / b.n
}

func (b *band) periods(g int) float64 { return float64(b.runs[g]) }

// corr is a bound's term in the linearised figure: g_j − μ − σ·g_0 at a
// return period's level, 0 at a value.
func corr(g []float64, j int, mu, sigma float64) float64 {
	if g == nil {
		return 0
	}
	return g[j] - mu - sigma*g[0]
}

// WhenName is how a condition names its quantity and bounds, as the
// command line writes it: total>=1/100, category:damage in [5e5,1e6).
func WhenName(c When) string {
	q := c.Kind
	if c.Name != "" {
		q += ":" + c.Name
	}
	if c.To == nil {
		return q + ">=" + boundName(c.From)
	}
	return q + " in [" + boundName(c.From) + "," + boundName(*c.To) + ")"
}

func boundName(b Bound) string {
	if b.OneIn != 0 {
		return "1/" + strconv.FormatFloat(b.OneIn, 'g', -1, 64)
	}
	return strconv.FormatFloat(b.X, 'g', -1, 64)
}

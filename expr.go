package variate

import (
	"math"
	"slices"
	"strconv"
)

// An expression is a function's value computed from the values of its
// node's other on-entry functions: numbers, their names, +, -, * and
// parentheses. It is held expanded, a sum of monomials over the node's
// drawn functions, each draw at most once in a monomial, so the mean and
// second moment of any sum of expressions need only each draw's mean and
// second moment.

const (
	// exprTokensMax caps an expression's tokens.
	exprTokensMax = 64
	// exprDepthMax caps its parentheses' nesting.
	exprDepthMax = 16
	// exprTermsMax caps its monomials once expanded.
	exprTermsMax = 256
)

// monomial is coef times the product of the variables in vars, sorted
// ascending and distinct.
type monomial struct {
	coef float64
	vars []int32
}

// poly is a sum of monomials.
type poly []monomial

// cexpr is a compiled expression: its expansion over the node's drawn
// functions.
type cexpr struct {
	p poly
}

type exprTok struct {
	kind byte // 'n' number, 'a' name, '+', '-', '*', 'u' unary minus, '(', ')'
	num  float64
	name string
}

// tokenize splits an expression into tokens, a '-' after an operator or
// '(' or at the start unary.
func tokenize(s string) ([]exprTok, string) {
	var out []exprTok
	i := 0
	// bound: the expression's bytes
	for i < len(s) {
		c := s[i]
		t, j, msg := exprTok{kind: c}, i+1, ""
		switch {
		case c == ' ' || c == '\t':
			i++
			continue
		case c == '+' || c == '*' || c == '(' || c == ')':
		case c == '-':
			if !afterOperand(out) {
				t.kind = 'u'
			}
		case isNameStart(c):
			j = scanName(s, i)
			t = exprTok{kind: 'a', name: s[i:j]}
		case isDigit(c) || c == '.':
			t, j, msg = scanNumber(s, i)
		default:
			msg = "unexpected " + strconv.QuoteRune(rune(c))
		}
		if msg != "" {
			return nil, msg
		}
		if out, i = append(out, t), j; len(out) > exprTokensMax {
			return nil, "more than " + strconv.Itoa(exprTokensMax) + " tokens"
		}
	}
	return out, ""
}

// afterOperand reports whether the last token ends an operand.
func afterOperand(toks []exprTok) bool {
	if len(toks) == 0 {
		return false
	}
	k := toks[len(toks)-1].kind
	return k == 'n' || k == 'a' || k == ')'
}

func isNameStart(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_'
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// scanName is the end of the name starting at i.
func scanName(s string, i int) int {
	j := i + 1
	// bound: the expression's bytes
	for j < len(s) && (isNameStart(s[j]) || isDigit(s[j])) {
		j++
	}
	return j
}

// scanNumber reads the number starting at i, an exponent's sign
// included.
func scanNumber(s string, i int) (exprTok, int, string) {
	j := i + 1
	// bound: the expression's bytes
	for j < len(s) {
		c, prev := s[j], s[j-1]
		if !isDigit(c) && c != '.' && c != 'e' && c != 'E' && ((c != '-' && c != '+') || (prev != 'e' && prev != 'E')) {
			break
		}
		j++
	}
	v, err := strconv.ParseFloat(s[i:j], 64)
	if err != nil {
		return exprTok{}, j, "bad number " + s[i:j]
	}
	return exprTok{kind: 'n', num: v}, j, ""
}

// exprPrec is an operator's precedence, unary minus binding tightest.
func exprPrec(k byte) int {
	switch k {
	case 'u':
		return 3
	case '*':
		return 2
	}
	return 1
}

// shunt is the shunting-yard state: the postfix output, the operators
// pending, the parentheses open, and whether an operand was last.
type shunt struct {
	out, ops []exprTok
	depth    int
	operand  bool
}

// pop moves pending operators to the output while keep allows.
func (sh *shunt) pop(keep func(exprTok) bool) {
	// bound: the operators pending
	for len(sh.ops) > 0 && keep(sh.ops[len(sh.ops)-1]) {
		sh.out, sh.ops = append(sh.out, sh.ops[len(sh.ops)-1]), sh.ops[:len(sh.ops)-1]
	}
}

// step takes one token, or says why it cannot stand there.
func (sh *shunt) step(t exprTok) string {
	switch t.kind {
	case 'n', 'a':
		if sh.operand {
			return "two operands in a row"
		}
		sh.out, sh.operand = append(sh.out, t), true
	case '(':
		if sh.depth++; sh.depth > exprDepthMax {
			return "parentheses deeper than " + strconv.Itoa(exprDepthMax)
		}
		if sh.operand {
			return "an operand before ("
		}
		sh.ops = append(sh.ops, t)
	case ')':
		if !sh.operand || sh.depth == 0 {
			return "unmatched )"
		}
		sh.depth--
		sh.pop(func(o exprTok) bool { return o.kind != '(' })
		sh.ops = sh.ops[:len(sh.ops)-1]
	case 'u':
		sh.ops = append(sh.ops, t)
	default:
		if !sh.operand {
			return "an operator without its operand"
		}
		sh.pop(func(o exprTok) bool { return o.kind != '(' && exprPrec(o.kind) >= exprPrec(t.kind) })
		sh.ops, sh.operand = append(sh.ops, t), false
	}
	return ""
}

// parseExpr turns an expression into postfix order, checking its shape.
func parseExpr(s string) ([]exprTok, string) {
	toks, msg := tokenize(s)
	if msg != "" {
		return nil, msg
	}
	if len(toks) == 0 {
		return nil, "empty"
	}
	var sh shunt
	// bound: exprTokensMax tokens
	for _, t := range toks {
		if msg := sh.step(t); msg != "" {
			return nil, msg
		}
	}
	if !sh.operand || sh.depth != 0 {
		return nil, "incomplete"
	}
	sh.pop(func(exprTok) bool { return true })
	return sh.out, ""
}

// exprNames lists the names a postfix expression reads.
func exprNames(rpn []exprTok) []string {
	var out []string
	for _, t := range rpn {
		if t.kind == 'a' && !slices.Contains(out, t.name) {
			out = append(out, t.name)
		}
	}
	return out
}

// evalPoly expands a postfix expression, each name its polynomial.
func evalPoly(rpn []exprTok, names map[string]poly) (poly, string) {
	var st []poly
	// bound: exprTokensMax tokens
	for _, t := range rpn {
		var p poly
		msg := ""
		switch t.kind {
		case 'n':
			p = poly{{coef: t.num}}
		case 'a':
			p = names[t.name]
		case 'u':
			st[len(st)-1] = st[len(st)-1].scale(-1)
			continue
		default:
			a, b := st[len(st)-2], st[len(st)-1]
			st = st[:len(st)-2]
			switch t.kind {
			case '+':
				p = a.add(b)
			case '-':
				p = a.add(b.scale(-1))
			default:
				p, msg = a.mul(b)
			}
		}
		if msg != "" {
			return nil, msg
		}
		if len(p) > exprTermsMax {
			return nil, "more than " + strconv.Itoa(exprTermsMax) + " terms once multiplied out"
		}
		st = append(st, p)
	}
	return st[0], ""
}

func (p poly) scale(c float64) poly {
	out := make(poly, 0, len(p))
	for _, t := range p {
		if v := c * t.coef; v != 0 {
			out = append(out, monomial{coef: v, vars: t.vars})
		}
	}
	return out
}

// add is p + q, monomials of the same variables merged and zero ones
// dropped.
func (p poly) add(q poly) poly {
	out := append(poly(nil), p...)
	// bound: q's monomials
	for _, t := range q {
		i := slices.IndexFunc(out, func(u monomial) bool { return slices.Equal(u.vars, t.vars) })
		if i < 0 {
			out = append(out, t)
			continue
		}
		out[i].coef += t.coef
	}
	return slices.DeleteFunc(out, func(u monomial) bool { return u.coef == 0 })
}

// mul is p × q; a variable in both of a pair of monomials would be
// squared, which is refused.
func (p poly) mul(q poly) (poly, string) {
	var out poly
	// bound: p's monomials
	for _, a := range p {
		// bound: q's monomials
		for _, b := range q {
			for _, v := range a.vars {
				if slices.Contains(b.vars, v) {
					return nil, "a draw multiplied by itself"
				}
			}
			vars := append(append([]int32(nil), a.vars...), b.vars...)
			slices.Sort(vars)
			out = out.add(poly{{coef: a.coef * b.coef, vars: vars}})
			if len(out) > exprTermsMax {
				return nil, "more than " + strconv.Itoa(exprTermsMax) + " terms once multiplied out"
			}
		}
	}
	return out, ""
}

// lowest is a lower bound of p with each variable in its [lo, hi], all
// lo ≥ 0: each monomial at its least.
func (p poly) lowest(lo, hi func(int32) float64) float64 {
	sum := 0.0
	for _, t := range p {
		x := t.coef
		for _, v := range t.vars {
			b := lo(v)
			if t.coef < 0 {
				b = hi(v)
			}
			if b == 0 {
				x = 0
				break
			}
			x *= b
		}
		sum += x
	}
	return sum
}

// flattenExprs compiles node i's expressions after its drawn on-entry
// functions, from start, each once every expression it names is: a
// name is a drawn function's value, or an expression's expansion.
func (b *builder) flattenExprs(m *Compiled, node int32, n Node, start int32, cats map[string]int32) error {
	names := map[string]poly{}
	// bound: the node's drawn functions
	for f := start; f < index(len(m.fns)); f++ {
		names[m.fnNames[f][len(n.ID)+1:]] = poly{{coef: 1, vars: []int32{f}}}
	}
	rpns, pending, err := parseExprs(n)
	if err != nil {
		return err
	}
	// bound: one expression or more compiled each pass
	for len(pending) > 0 {
		var left []Function
		for _, f := range pending {
			ready, err := exprReady(n, f, rpns[f.Name], names)
			if err != nil {
				return err
			}
			if !ready {
				left = append(left, f)
				continue
			}
			p, msg := evalPoly(rpns[f.Name], names)
			if msg != "" {
				return modelError("function", n.ID+"."+f.Name, "expression "+f.Expr+": "+msg)
			}
			names[f.Name] = p
			if err := b.addExpr(m, node, n, f, p, cats); err != nil {
				return err
			}
		}
		if len(left) == len(pending) {
			return modelError("function", n.ID+"."+left[0].Name, "expressions name each other in a cycle")
		}
		pending = left
	}
	return nil
}

// parseExprs parses the node's expressions, in postfix order by name.
func parseExprs(n Node) (map[string][]exprTok, []Function, error) {
	rpns := map[string][]exprTok{}
	var pending []Function
	for _, f := range n.Functions {
		if f.Expr == "" {
			continue
		}
		rpn, msg := parseExpr(f.Expr)
		if msg != "" {
			return nil, nil, modelError("function", n.ID+"."+f.Name, "expression "+f.Expr+": "+msg)
		}
		rpns[f.Name] = rpn
		pending = append(pending, f)
	}
	return rpns, pending, nil
}

// exprReady reports whether every name f's expression reads is
// compiled, refusing a name that is not one of the node's drawn
// on-entry functions or expressions without a factor.
func exprReady(n Node, f Function, rpn []exprTok, names map[string]poly) (bool, error) {
	ready := true
	for _, name := range exprNames(rpn) {
		if _, ok := names[name]; ok {
			continue
		}
		i := slices.IndexFunc(n.Functions, func(g Function) bool { return g.Name == name })
		why := ""
		switch {
		case i < 0:
			why = "names " + name + ", not a function of the node"
		case n.Functions[i].Basis == PerTime:
			why = "names " + name + ", a per-time function"
		case n.Threshold != nil && name == n.Threshold.Function:
			why = "names " + name + ", the value the threshold tests"
		case n.Functions[i].Expr != "" && n.Functions[i].Factor != "":
			why = "names " + name + ", an expression with a factor"
		}
		if why != "" {
			return false, modelError("function", n.ID+"."+f.Name, why)
		}
		ready = false
	}
	return ready, nil
}

// addExpr adds the function an expression is: a constant factor scales
// it, a drawn one multiplies each value; it is refused when it may be
// below 0.
func (b *builder) addExpr(m *Compiled, node int32, n Node, f Function, p poly, cats map[string]int32) error {
	factor := noIndex
	if f.Factor != "" {
		i := b.factorIndex[f.Factor]
		switch lo, hi := b.g.Factors[i].Model.Support(); {
		case lo != hi:
			factor = b.factorPool[i]
		case lo != 1:
			p = p.scale(lo)
		}
	}
	if p.lowest(m.varLo, m.varHi) < 0 {
		return modelError("function", n.ID+"."+f.Name, "expression "+f.Expr+" may be below 0")
	}
	m.exprs = append(m.exprs, cexpr{p: p})
	m.fns = append(m.fns, cfn{pool: noIndex, category: categoryOf(cats, f.Category), factor: factor, expr: index(len(m.exprs) - 1)})
	m.fnNames = append(m.fnNames, n.ID+":"+f.Name)
	m.fnNode = append(m.fnNode, node)
	m.fnPerTime = append(m.fnPerTime, false)
	return nil
}

// varLo and varHi bound drawn function f's value: its law's support
// times its factor's.
func (m *Compiled) varLo(f int32) float64 {
	lo, _ := m.pools[m.fns[f].pool].Support()
	if g := m.fns[f].factor; g != noIndex {
		glo, _ := m.pools[g].Support()
		lo *= glo
	}
	return lo
}

func (m *Compiled) varHi(f int32) float64 {
	_, hi := m.pools[m.fns[f].pool].Support()
	if g := m.fns[f].factor; g != noIndex && hi != 0 {
		_, ghi := m.pools[g].Support()
		hi *= ghi
	}
	return hi
}

// eval is expression e's value for the round, from its node's drawn
// values in vals.
func (e *cexpr) eval(vals []float64) float64 {
	sum := 0.0
	for _, t := range e.p {
		x := t.coef
		for _, v := range t.vars {
			x *= vals[v]
		}
		sum += x
	}
	return sum
}

// exprTerms appends expression function f's monomials times sign.
func (m *Compiled) exprTerms(f int32, sign float64, out poly) poly {
	for _, t := range m.exprs[m.fns[f].expr].p {
		out = append(out, monomial{coef: sign * t.coef, vars: t.vars})
	}
	return out
}

// nodePolyMoments is sumMoments for a node with expressions: its values
// under the filter as one polynomial over its draws, an expression's
// drawn factor a variable of its own, encoded -(pool+1).
func (m *Compiled) nodePolyMoments(from, to int32, filt valueFilter) (mean, second, secondTheta float64) {
	var p poly
	// bound: functions of one node
	for f := from; f < to; f++ {
		sign, ok := m.fnSign(f, filt)
		c := m.fns[f]
		switch {
		case !ok:
		case c.expr == noIndex:
			p = append(p, monomial{coef: sign, vars: []int32{f}})
		default:
			n := len(p)
			p = m.exprTerms(f, sign, p)
			if c.factor == noIndex {
				continue
			}
			for i := n; i < len(p); i++ {
				p[i].vars = append([]int32{-c.factor - 1}, p[i].vars...)
			}
		}
	}
	return m.polyMoments(p)
}

// polyMoments is E[P], E[P²] and the Θ-version of E[P²], where two
// values share only the draws made once a period, of a polynomial over
// independent variables, none squared in a monomial:
// E[P²] = Σ_a Σ_b c_a c_b Π_{a∩b} E[X²] Π_{a△b} E[X].
func (m *Compiled) polyMoments(p poly) (mean, second, secondTheta float64) {
	moms := map[int32]varMom{}
	get := func(v int32) varMom {
		if x, ok := moms[v]; ok {
			return x
		}
		var x varMom
		if v < 0 {
			g := m.pools[-v-1]
			x = varMom{mu: g.Mean(), m2: g.Var() + g.Mean()*g.Mean()}
			x.m2t = x.m2
		} else {
			mu, m2, period := m.lawMoments(v)
			x = varMom{mu: mu, m2: m2, m2t: mu * mu}
			if period {
				x.m2t = m2
			}
		}
		moms[v] = x
		return x
	}
	// bound: the polynomial's monomials
	for _, a := range p {
		e := a.coef
		for _, v := range a.vars {
			e *= get(v).mu
		}
		mean += e
		// bound: the polynomial's monomials
		for _, b := range p {
			x, xt := pairMoment(a, b, get)
			second += x
			secondTheta += xt
		}
	}
	return mean, second, secondTheta
}

// pairMoment is E[a·b] of two monomials and its Θ-version: E[X²] for a
// variable in both, E[X] for one in either alone.
func pairMoment(a, b monomial, get func(int32) varMom) (s, st float64) {
	s, st = a.coef*b.coef, a.coef*b.coef
	i, j := 0, 0
	// bound: the two monomials' variables
	for i < len(a.vars) || j < len(b.vars) {
		switch {
		case j == len(b.vars) || i < len(a.vars) && a.vars[i] < b.vars[j]:
			x := get(a.vars[i])
			s, st = s*x.mu, st*x.mu
			i++
		case i == len(a.vars) || b.vars[j] < a.vars[i]:
			x := get(b.vars[j])
			s, st = s*x.mu, st*x.mu
			j++
		default:
			x := get(a.vars[i])
			s, st = s*x.m2, st*x.m2t
			i, j = i+1, j+1
		}
	}
	return s, st
}

// varMom is a variable's E[X], E[X²], and E[X·X'] for two values of a
// period: E[X²] when drawn once a period, E[X]² otherwise.
type varMom struct {
	mu, m2, m2t float64
}

// exprPositive reports whether expression function f's value is above 0
// for certain: no monomial below 0, and one above 0 whose draws are.
func (m *Compiled) exprPositive(f int32) bool {
	c := m.fns[f]
	if c.factor != noIndex && m.pools[c.factor].CDF(0) != 0 {
		return false
	}
	pos := false
	for _, t := range m.exprs[c.expr].p {
		if t.coef < 0 {
			return false
		}
		all := true
		for _, v := range t.vars {
			g := m.fns[v].factor
			all = all && m.pools[m.fns[v].pool].CDF(0) == 0 && (g == noIndex || m.pools[g].CDF(0) == 0)
		}
		pos = pos || all
	}
	return pos
}

// exprsZeroKnown reports whether P(T = 0) can count each expression in
// the total as never 0: every one of them is above 0 for certain.
func (m *Compiled) exprsZeroKnown() bool {
	for f, c := range m.fns {
		if c.expr != noIndex && m.fnInTotal(index(f)) && !m.exprPositive(index(f)) {
			return false
		}
	}
	return true
}

// fnInfVar reports whether function f's value has infinite variance:
// its law's, or for an expression one of its draws'.
func (m *Compiled) fnInfVar(f int32) bool {
	c := m.fns[f]
	if c.expr == noIndex {
		return math.IsInf(m.pools[c.pool].Var(), 1)
	}
	for _, t := range m.exprs[c.expr].p {
		for _, v := range t.vars {
			if math.IsInf(m.pools[m.fns[v].pool].Var(), 1) {
				return true
			}
		}
	}
	return false
}

// fnBooks reports whether function f can book a value other than 0.
func (m *Compiled) fnBooks(f int32) bool {
	c := m.fns[f]
	if c.expr != noIndex {
		return len(m.exprs[c.expr].p) > 0
	}
	d := m.pools[c.pool]
	return d.Mean() != 0 || d.Var() != 0
}

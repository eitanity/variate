package report

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"strconv"

	"code.eitanity.com/variate"
)

// WriteSweepTable writes a sweep as a table, the largest swing first: an
// exact or simulated figure as one number, a bracket as [lo, hi], and a
// simulated difference's standard error beside it.
func WriteSweepTable(w *bufio.Writer, sw *variate.Sweep) {
	pct := strconv.FormatFloat(100*sw.Delta, 'g', -1, 64) + "%"
	fmt.Fprintf(w, "sweep of %s, each input × (1 ± %s)", sw.Figure, pct)
	if sw.Periods > 0 {
		fmt.Fprintf(w, "; simulated rows over %d periods at seed %d", sw.Periods, sw.Seed)
	}
	fmt.Fprintf(w, "\n\n%-44s %-10s %22s %22s %22s %12s %10s %10s\n", "input", "method", "−"+pct, "base", "+"+pct, "swing", "± down", "± up")
	clamped := false
	for _, r := range sw.Rows {
		mark := ""
		if r.Clamped {
			mark, clamped = " *", true
		}
		fmt.Fprintf(w, "%-44s %-10s %22s %22s %22s %12.6g %10s %10s%s\n", r.Input.Name, r.Method,
			rangeText(r.Down), rangeText(r.Base), rangeText(r.Up), r.Swing, seText(r, r.DownSE), seText(r, r.UpSE), mark)
	}
	if clamped {
		fmt.Fprintln(w, "\n* a probability was held at 1, or to its node's sum, at a nudge")
	}
}

func rangeText(r variate.Range) string {
	if r.Lo == r.Hi {
		return strconv.FormatFloat(r.Lo, 'g', 6, 64)
	}
	return "[" + strconv.FormatFloat(r.Lo, 'g', 5, 64) + ", " + strconv.FormatFloat(r.Hi, 'g', 5, 64) + "]"
}

// seText is a simulated row's standard error; exact and bracketed rows
// have none to show.
func seText(r variate.SweepRow, se float64) string {
	switch {
	case r.Method != variate.MethodSimulation:
		return ""
	case math.IsNaN(se):
		return "n/a"
	}
	return strconv.FormatFloat(se, 'g', 3, 64)
}

// SweepJSON is WriteSweepJSON into a byte slice.
func SweepJSON(sw *variate.Sweep) []byte {
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	WriteSweepJSON(w, sw)
	_ = w.Flush() // a bytes.Buffer does not fail
	return buf.Bytes()
}

// WriteSweepJSON writes a sweep as one JSON object; a figure is a lo,
// hi pair, equal when exact or simulated, and a missing standard error
// is null.
func WriteSweepJSON(w *bufio.Writer, sw *variate.Sweep) {
	j := &jw{w: w}
	j.open('{')
	j.fieldStr("figure", sw.Figure)
	j.field("delta", sw.Delta)
	j.fieldUint("seed", sw.Seed)
	j.fieldInt("periods", sw.Periods)
	j.key("rows")
	j.open('[')
	for _, r := range sw.Rows {
		j.sep()
		j.open('{')
		j.fieldStr("input", r.Input.Name)
		j.fieldStr("kind", r.Input.Kind)
		j.fieldStr("method", r.Method)
		j.rangeField("down", r.Down)
		j.rangeField("base", r.Base)
		j.rangeField("up", r.Up)
		j.field("swing", r.Swing)
		j.field("down_se", r.DownSE)
		j.field("up_se", r.UpSE)
		j.fieldBool("clamped", r.Clamped)
		j.closeC('}')
	}
	j.closeC(']')
	j.closeC('}')
	w.WriteByte('\n')
}

func (j *jw) rangeField(k string, r variate.Range) {
	j.key(k)
	j.open('{')
	j.field("lo", r.Lo)
	j.field("hi", r.Hi)
	j.closeC('}')
}

// WriteSweepCSV writes one row per input.
func WriteSweepCSV(w *bufio.Writer, sw *variate.Sweep) {
	fmt.Fprintln(w, "input,kind,method,down_lo,down_hi,base_lo,base_hi,up_lo,up_hi,swing,down_se,up_se,clamped")
	for _, r := range sw.Rows {
		fmt.Fprintf(w, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%v\n", CSVField(r.Input.Name), r.Input.Kind, r.Method,
			csvNum(r.Down.Lo), csvNum(r.Down.Hi), csvNum(r.Base.Lo), csvNum(r.Base.Hi), csvNum(r.Up.Lo), csvNum(r.Up.Hi),
			csvNum(r.Swing), csvCell(r.DownSE), csvCell(r.UpSE), r.Clamped)
	}
}

// csvCell is a number, or empty when there is none.
func csvCell(x float64) string {
	if math.IsNaN(x) {
		return ""
	}
	return csvNum(x)
}

package report

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"strconv"

	"code.eitanity.com/variate"
)

// WriteSweepTable writes a sweep as a table, the largest swing first:
// each input's range, the figure at its low end, as it is and at its
// high end, an exact or simulated figure as one number and a bracket as
// [lo, hi], and a simulated difference's standard error beside it.
func WriteSweepTable(w *bufio.Writer, sw *variate.Sweep) {
	if t := PhaseText(sw.Phase); t != "" {
		fmt.Fprintln(w, t)
	}
	if sw.Screen > 0 {
		pct := strconv.FormatFloat(100*sw.Screen, 'g', -1, 64) + "%"
		fmt.Fprintf(w, "screen of %s, each input × (1 ± %s)", sw.Figure, pct)
	} else {
		fmt.Fprintf(w, "sweep of %s, each input with a range to its two ends", sw.Figure)
	}
	if sw.Periods > 0 {
		fmt.Fprintf(w, "; simulated rows over %d periods at seed %d", sw.Periods, sw.Seed)
	}
	fmt.Fprintf(w, "\n\n%-44s %-20s %-10s %22s %22s %22s %12s %10s %10s\n", "input", "range", "method", "at low", "base", "at high", "swing", "± low", "± high")
	clamped := false
	for _, r := range sw.Rows {
		mark := ""
		if r.Clamped {
			mark, clamped = " *", true
		}
		fmt.Fprintf(w, "%-44s %-20s %-10s %22s %22s %22s %12.6g %10s %10s%s\n", r.Input.Name, inputRangeText(sw, r), r.Method,
			rangeText(r.AtLow), rangeText(r.Base), rangeText(r.AtHigh), r.Swing, seText(r, r.LowSE), seText(r, r.HighSE), mark)
	}
	if clamped {
		fmt.Fprintln(w, "\n* a probability was held at 1, or to its node's sum, at an end")
	}
}

// inputRangeText is the range an input was moved over: as factors, which
// read the same whatever unit the model wrote the input in, and a
// probability given as probabilities as those.
func inputRangeText(sw *variate.Sweep, r variate.SweepRow) string {
	g := func(x float64) string { return strconv.FormatFloat(x, 'g', 4, 64) }
	if rg := r.Input.Range; sw.Screen == 0 && rg != nil && !rg.Factors && r.Input.Kind == variate.InputProbability {
		return "p " + g(rg.Low) + " – " + g(rg.High)
	}
	return "×" + g(r.Low) + " – ×" + g(r.High)
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
// hi pair, equal when exact or simulated, a missing standard error is
// null, and each row carries the range the model gave its input (null
// under a screen or when there is none) and the factors it was moved by.
func WriteSweepJSON(w *bufio.Writer, sw *variate.Sweep) {
	j := &jw{w: w}
	j.open('{')
	j.phase(sw.Phase)
	j.fieldStr("figure", sw.Figure)
	j.field("screen", sw.Screen)
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
		j.key("range")
		if rg := r.Input.Range; rg != nil && sw.Screen == 0 {
			j.open('{')
			j.field("low", rg.Low)
			j.field("high", rg.High)
			j.fieldBool("factors", rg.Factors)
			j.closeC('}')
		} else {
			j.w.WriteString("null")
		}
		j.field("low_factor", r.Low)
		j.field("high_factor", r.High)
		j.rangeField("at_low", r.AtLow)
		j.rangeField("base", r.Base)
		j.rangeField("at_high", r.AtHigh)
		j.field("swing", r.Swing)
		j.field("low_se", r.LowSE)
		j.field("high_se", r.HighSE)
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
	fmt.Fprintln(w, "input,kind,method,low_factor,high_factor,at_low_lo,at_low_hi,base_lo,base_hi,at_high_lo,at_high_hi,swing,low_se,high_se,clamped")
	for _, r := range sw.Rows {
		fmt.Fprintf(w, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%v\n", CSVField(r.Input.Name), r.Input.Kind, r.Method,
			csvNum(r.Low), csvNum(r.High), csvNum(r.AtLow.Lo), csvNum(r.AtLow.Hi), csvNum(r.Base.Lo), csvNum(r.Base.Hi),
			csvNum(r.AtHigh.Lo), csvNum(r.AtHigh.Hi), csvNum(r.Swing), csvCell(r.LowSE), csvCell(r.HighSE), r.Clamped)
	}
}

// csvCell is a number, or empty when there is none.
func csvCell(x float64) string {
	if math.IsNaN(x) {
		return ""
	}
	return csvNum(x)
}

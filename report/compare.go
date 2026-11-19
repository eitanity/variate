package report

import (
	"bufio"
	"fmt"
	"math"
	"strconv"

	"code.eitanity.com/variate"
)

// WriteCompareTable writes a comparison as a table, one option a row in
// its order, named by names: the figure as it is and with the option,
// the difference, and a simulated difference's standard error.
func WriteCompareTable(w *bufio.Writer, c *variate.Comparison, names []string) {
	fmt.Fprintf(w, "comparison of %s, each option against the model as it is", c.Figure)
	if c.Periods > 0 {
		fmt.Fprintf(w, "; simulated rows over %d periods at seed %d", c.Periods, c.Seed)
	}
	width := len("option")
	for i := range c.Rows {
		width = max(width, len(compareName(names, i)))
	}
	fmt.Fprintf(w, "\n\n%-*s %-10s %22s %22s %22s %10s\n", width, "option", "method", "as is", "option", "difference", "±")
	for i, r := range c.Rows {
		fmt.Fprintf(w, "%-*s %-10s %22s %22s %22s %10s\n", width, compareName(names, i), r.Method,
			rangeText(r.Base), rangeText(r.Value), rangeText(r.Diff), compareSE(r))
	}
}

// compareName is option i's name, or its place when it has none.
func compareName(names []string, i int) string {
	if i < len(names) && names[i] != "" {
		return names[i]
	}
	return "option " + strconv.Itoa(i+1)
}

// compareSE is a simulated row's standard error; exact and bracketed
// rows have none to show.
func compareSE(r variate.CompareRow) string {
	switch {
	case r.Method != variate.MethodSimulation:
		return ""
	case math.IsNaN(r.SE):
		return "n/a"
	}
	return strconv.FormatFloat(r.SE, 'g', 3, 64)
}

// WriteCompareJSON writes a comparison as one JSON object; a figure is
// a lo, hi pair, equal when exact or simulated, and a missing standard
// error is null.
func WriteCompareJSON(w *bufio.Writer, c *variate.Comparison, names []string) {
	j := &jw{w: w}
	j.open('{')
	j.fieldStr("figure", c.Figure)
	j.fieldUint("seed", c.Seed)
	j.fieldInt("periods", c.Periods)
	j.key("rows")
	j.open('[')
	for i, r := range c.Rows {
		j.sep()
		j.open('{')
		j.fieldStr("option", compareName(names, i))
		j.fieldStr("method", r.Method)
		j.rangeField("base", r.Base)
		j.rangeField("value", r.Value)
		j.rangeField("diff", r.Diff)
		j.field("se", r.SE)
		j.closeC('}')
	}
	j.closeC(']')
	j.closeC('}')
	w.WriteByte('\n')
}

// WriteCompareCSV writes one row per option.
func WriteCompareCSV(w *bufio.Writer, c *variate.Comparison, names []string) {
	fmt.Fprintln(w, "option,method,base_lo,base_hi,value_lo,value_hi,diff_lo,diff_hi,se")
	for i, r := range c.Rows {
		fmt.Fprintf(w, "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", CSVField(compareName(names, i)), r.Method,
			csvNum(r.Base.Lo), csvNum(r.Base.Hi), csvNum(r.Value.Lo), csvNum(r.Value.Hi),
			csvNum(r.Diff.Lo), csvNum(r.Diff.Hi), csvCell(r.SE))
	}
}

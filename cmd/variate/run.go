package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/report"
)

// plan loads, compiles and prints the sizing table without running.
func plan(args []string, stdout, stderr io.Writer) error {
	o, err := parseFlags("plan", args, stderr)
	if err != nil {
		return err
	}
	m, err := o.apply()
	if err != nil {
		return err
	}
	c, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	printWarnings(stderr, warnings)
	e, err := c.Analytic()
	if err != nil {
		return fmt.Errorf("analytic: %w", err)
	}
	w := bufio.NewWriter(stdout)
	if o.format == formatJSON {
		periods := c.PlanPeriods(int64(max(m.Config.BlockPeriods, 1)))
		work := float64(periods) * (e.StepsPerPeriod + e.ArrivalsPerPeriod)
		blocks := make([]report.BlockFigure, 0, len(m.Blocks))
		for _, b := range m.Blocks {
			blocks = append(blocks, report.BlockFigure{Source: b.Source, Rate: b.Rate, MTBF: b.MTBF, Unavailability: b.Unavailability})
		}
		report.PlanJSON(w, c, e, m.Config, warnings, estimate(work, float64(periods), m.Config.Workers), blocks)
		return flush(w)
	}
	writePlan(w, c, e, m.Config)
	for _, b := range m.Blocks {
		fmt.Fprintf(w, "block %s: %.6g failures per period, system MTBF %.6g hours, unavailability %.3g\n", b.Source, b.Rate, b.MTBF, b.Unavailability)
	}
	return flush(w)
}

// runModel loads, compiles and runs the model, writing the summary and,
// when asked, the rounds.
func runModel(args []string, stdout, stderr io.Writer, stream bool) error {
	o, err := parseFlags("run", args, stderr)
	if err != nil {
		return err
	}
	m, err := o.apply()
	if err != nil {
		return err
	}
	c, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	printWarnings(stderr, warnings)
	rounds, closeRounds, err := o.roundsWriter(c, stdout, stream)
	if err != nil {
		return err
	}
	if rounds != nil {
		m.Config.Observer = rounds
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		return fmt.Errorf("workspace: %w", err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), m.Config.Seed)
	if err != nil {
		err = fmt.Errorf("run: %w", err)
	}
	if closeErr := closeRounds(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	rep, err := report.New(c, w, r, o.analytic)
	if err != nil {
		return err
	}
	return o.writeOutputs(rep, stdout, stderr, stream)
}

// roundsWriter builds the observer for --rounds: to --rounds-out, or to
// stdout when streaming.
func (o *options) roundsWriter(c *variate.Compiled, stdout io.Writer, stream bool) (variate.Observer, func() error, error) {
	none := func() error { return nil }
	format := o.rounds
	if stream && format == "" {
		format = formatJSONL
	}
	if format == "" {
		return nil, none, nil
	}
	if !stream && o.roundsOut == "" {
		return nil, none, errors.New("--rounds needs --rounds-out unless streaming")
	}
	dst := stdout
	closer := none
	if o.roundsOut != "" {
		f, err := os.Create(o.roundsOut) //nolint:gosec // the output file the user named
		if err != nil {
			return nil, none, fmt.Errorf("rounds file: %w", err)
		}
		dst = f
		closer = func() error { return closeFile(f) }
	}
	return newRoundWriter(format, dst, c), closer, nil
}

// writeOutputs writes the summary in the chosen format to --out (else
// stdout, or stderr when the rounds stream to stdout) and the exceedance
// curve to --exceedance.
func (o *options) writeOutputs(rep *report.Report, stdout, stderr io.Writer, stream bool) error {
	dst := stdout
	if stream && o.roundsOut == "" {
		dst = stderr
	}
	closer := func() error { return nil }
	if o.out != "" {
		f, err := os.Create(o.out) //nolint:gosec // the output file the user named
		if err != nil {
			return fmt.Errorf("output file: %w", err)
		}
		dst, closer = f, func() error { return closeFile(f) }
	}
	bw := bufio.NewWriter(dst)
	switch o.format {
	case formatJSON:
		rep.WriteJSON(bw)
	case formatCSV:
		rep.WriteCSV(bw)
	default:
		rep.WriteTable(bw)
	}
	if err := flush(bw); err != nil {
		return err
	}
	if err := closer(); err != nil {
		return err
	}
	if o.exceedance != "" {
		return rep.WriteExceedanceFile(o.exceedance)
	}
	return nil
}

// flush and closeFile wrap the writer errors with their stage.
func flush(w *bufio.Writer) error {
	if err := w.Flush(); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func closeFile(f *os.File) error {
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", f.Name(), err)
	}
	return nil
}

func printWarnings(stderr io.Writer, warnings []variate.Warning) {
	for _, w := range warnings {
		fmt.Fprintf(stderr, "warning: %s %s: %s\n", w.Code, w.Subject, w.Message)
	}
}

// writePlan prints the sizing table and the expected work.
func writePlan(w io.Writer, c *variate.Compiled, e *variate.Expectation, cfg variate.Config) {
	rows := c.Plan()
	fmt.Fprintf(w, "%-12s %-32s %14s %14s\n", "criterion", "driver", "per period", "periods")
	var need int64
	binding := ""
	for _, row := range rows {
		fmt.Fprintf(w, "%-12s %-32s %14.6g %14d\n", row.Criterion, row.Driver, row.PlannedPerPeriod, row.PeriodsNeeded)
		if row.PeriodsNeeded > need {
			need, binding = row.PeriodsNeeded, row.Criterion+" "+row.Driver
		}
	}
	periods := c.PlanPeriods(int64(max(cfg.BlockPeriods, 1)))
	fmt.Fprintf(w, "\nbinding: %s\nperiods: %d\nexpected rounds per period: %.6g\nexpected steps per period: %.6g\n", binding, periods, e.RoundsPerPeriod, e.StepsPerPeriod)
	work := float64(periods) * (e.StepsPerPeriod + e.ArrivalsPerPeriod)
	fmt.Fprintf(w, "expected steps: %.4g\nestimated time: %s at %d M steps/s and %d M periods/s per core\n", work, estimate(work, float64(periods), cfg.Workers), int(stepsPerSecond/1e6), int(periodsPerSecond/1e6))
	if _, _, err := quantileLine(c); err == nil {
		fmt.Fprintln(w, "transform: applies")
	}
}

// stepsPerSecond and periodsPerSecond are the planning rates per core:
// a walk step, and a period's own cost whether or not anything happens
// in it, which is what a rare-event run spends its time on.
const (
	stepsPerSecond   = 25e6
	periodsPerSecond = 12e6
)

func estimate(steps, periods float64, workers int) string {
	cores := max(workers, 1)
	seconds := (steps/stepsPerSecond + periods/periodsPerSecond) / float64(cores)
	switch {
	case seconds < 1:
		return "under a second"
	case seconds < 120:
		return fmt.Sprintf("%.0f s", seconds)
	case seconds < 7200:
		return fmt.Sprintf("%.1f min", seconds/60)
	default:
		return fmt.Sprintf("%.1f h", seconds/3600)
	}
}

// quantileLine is the transform's quantile bracket, when it was run.
func quantileLine(c *variate.Compiled) (lo, hi []float64, err error) {
	tr := c.Transformed()
	if tr == nil || !tr.Applicable {
		return nil, nil, errors.New("no transform")
	}
	for _, p := range []float64{0.5, 0.9, 0.95, 0.99, 0.999} {
		l, h := tr.Total.Quantile(p)
		lo, hi = append(lo, l), append(hi, h)
	}
	return lo, hi, nil
}

package main

import (
	"bufio"
	"fmt"
	"io"
	"strconv"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/report"
)

// roundWriter writes each delivered round as one JSON Lines or CSV
// record, flushed as it is written so another process can follow.
type roundWriter struct {
	w     *bufio.Writer
	csv   bool
	names []string // nodes
	srcs  []string
	outs  []string
	line  []byte
}

// newRoundWriter returns the observer for a record format.
func newRoundWriter(format string, dst io.Writer, c *variate.Compiled) variate.Observer {
	if format == formatBin {
		return variate.NewRecordWriter(dst, c.LongestPath())
	}
	rw := &roundWriter{w: bufio.NewWriter(dst), csv: format == formatCSV, names: c.NodeIDs(), srcs: c.SourceIDs(), outs: c.OutcomeNames()}
	if rw.csv {
		rw.w.WriteString("seq,period,source,arrival,end,leaf,value,path\n")
	}
	return rw
}

// Observe writes one record and flushes it.
func (rw *roundWriter) Observe(r *variate.Round) error {
	rw.line = rw.line[:0]
	if rw.csv {
		rw.csvRecord(r)
	} else {
		rw.jsonRecord(r)
	}
	rw.line = append(rw.line, '\n')
	if _, err := rw.w.Write(rw.line); err != nil {
		return fmt.Errorf("write round: %w", err)
	}
	if err := rw.w.Flush(); err != nil {
		return fmt.Errorf("write round: %w", err)
	}
	return nil
}

func (rw *roundWriter) jsonRecord(r *variate.Round) {
	b := rw.line
	b = append(b, `{"seq":`...)
	b = strconv.AppendUint(b, r.Seq, 10)
	b = append(b, `,"period":`...)
	b = strconv.AppendInt(b, r.Period, 10)
	b = append(b, `,"source":`...)
	b = appendJSONString(b, rw.srcs[r.Source])
	b = append(b, `,"arrival":`...)
	b = strconv.AppendFloat(b, r.Arrival, 'g', -1, 64)
	b = append(b, `,"end":`...)
	b = strconv.AppendFloat(b, r.End, 'g', -1, 64)
	b = append(b, `,"leaf":`...)
	b = appendJSONString(b, rw.names[r.Leaf])
	b = append(b, `,"value":`...)
	b = strconv.AppendFloat(b, r.Value, 'g', -1, 64)
	b = append(b, `,"steps":[`...)
	for i, s := range r.Steps {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, `{"node":`...)
		b = appendJSONString(b, rw.names[s.Node])
		b = append(b, `,"outcome":`...)
		if s.Outcome < 0 {
			b = append(b, "null"...)
		} else {
			b = appendJSONString(b, rw.outs[s.Outcome])
		}
		b = append(b, `,"t":`...)
		b = strconv.AppendFloat(b, s.T, 'g', -1, 64)
		b = append(b, `,"value":`...)
		b = strconv.AppendFloat(b, s.Value, 'g', -1, 64)
		b = append(b, '}')
	}
	b = append(b, "]}"...)
	rw.line = b
}

func (rw *roundWriter) csvRecord(r *variate.Round) {
	b := rw.line
	b = strconv.AppendUint(b, r.Seq, 10)
	b = append(b, ',')
	b = strconv.AppendInt(b, r.Period, 10)
	b = append(b, ',')
	b = append(b, report.CSVField(rw.srcs[r.Source])...)
	b = append(b, ',')
	b = strconv.AppendFloat(b, r.Arrival, 'g', -1, 64)
	b = append(b, ',')
	b = strconv.AppendFloat(b, r.End, 'g', -1, 64)
	b = append(b, ',')
	b = append(b, report.CSVField(rw.names[r.Leaf])...)
	b = append(b, ',')
	b = strconv.AppendFloat(b, r.Value, 'g', -1, 64)
	b = append(b, ',')
	path := make([]byte, 0, 64)
	for i, s := range r.Steps {
		if i > 0 {
			path = append(path, '>')
		}
		path = append(path, rw.names[s.Node]...)
	}
	b = append(b, report.CSVField(string(path))...)
	rw.line = b
}

// appendJSONString appends s as a JSON string.
func appendJSONString(b []byte, s string) []byte {
	b = append(b, '"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '"' || c == '\\':
			b = append(b, '\\', c)
		case c < 0x20:
			b = append(b, '\\', 'u', '0', '0', hexDigit(c>>4), hexDigit(c&0xf))
		default:
			b = append(b, c)
		}
	}
	return append(b, '"')
}

func hexDigit(x byte) byte {
	if x < 10 {
		return '0' + x
	}
	return 'a' + x - 10
}

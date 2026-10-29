package config

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"code.eitanity.com/variate/dist"
)

// empirical reads { empirical: [x, …] } or { empirical: { file: name } }.
func (l *loader) empirical(v node, s float64) (dist.Distribution, error) {
	var values []float64
	var err error
	if v.isSequence() {
		values, err = v.floats(0)
	} else {
		values, err = l.empiricalFile(v)
	}
	if err != nil {
		return nil, err
	}
	for i := range values {
		values[i] *= s
	}
	d, err := dist.NewEmpirical(values)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

// empiricalFile reads numbers, one per line or comma-separated, from a
// file beside the model.
func (l *loader) empiricalFile(v node) ([]float64, error) {
	m, err := v.mapping("file")
	if err != nil {
		return nil, err
	}
	f, ok := m["file"]
	if !ok {
		return nil, v.errorf("empirical takes a list or { file: name }")
	}
	name, err := f.str()
	if err != nil {
		return nil, err
	}
	path, err := l.resolve(name)
	if err != nil {
		return nil, f.errorf("%v", err)
	}
	fh, err := os.Open(path) //nolint:gosec // resolve confined the path to the model's directory
	if err != nil {
		return nil, f.errorf("cannot open data file %q", name)
	}
	defer closeQuietly(fh)
	data, err := io.ReadAll(io.LimitReader(fh, maxDataSize+1))
	if err != nil {
		return nil, f.errorf("data file %q: %v", name, err)
	}
	if len(data) > maxDataSize {
		return nil, f.errorf("data file %q is larger than 64 MB", name)
	}
	values, err := readNumbers(data)
	if err != nil {
		return nil, f.errorf("data file %q: %v", name, err)
	}
	return values, nil
}

// resolve checks a data file name: relative, without .. segments, and
// not escaping the model's directory through a symlink.
func (l *loader) resolve(name string) (string, error) {
	if l.dir == "" {
		return "", errors.New("data files are not available here")
	}
	if filepath.IsAbs(name) || strings.HasPrefix(name, "/") || strings.HasPrefix(name, "\\") {
		return "", errors.New("data file path must be relative to the model file")
	}
	for _, seg := range strings.Split(filepath.ToSlash(name), "/") {
		if seg == ".." {
			return "", errors.New("data file path must not contain a .. segment")
		}
	}
	path := filepath.Join(l.dir, filepath.FromSlash(name))
	real, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", errors.New("data file does not exist")
	}
	dir, err := filepath.EvalSymlinks(l.dir)
	if err != nil {
		return "", errors.New("model directory does not resolve")
	}
	if real != dir && !strings.HasPrefix(real, dir+string(filepath.Separator)) {
		return "", errors.New("data file resolves outside the model's directory")
	}
	return real, nil
}

// readNumbers parses whitespace-, comma- or semicolon-separated numbers.
func readNumbers(data []byte) ([]float64, error) {
	var out []float64
	for i, field := range strings.FieldsFunc(string(data), func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == ';' || r == '\n' || r == '\r'
	}) {
		x, err := strconv.ParseFloat(field, 64)
		if err != nil {
			return nil, errors.New("entry " + strconv.Itoa(i+1) + " is not a number")
		}
		out = append(out, x)
	}
	if len(out) == 0 {
		return nil, errors.New("no numbers")
	}
	return out, nil
}

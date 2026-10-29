package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const sample = `./walk.go:12:6: moved to heap: row
./compile.go:40:2: leaking param: g
./dist/sampler.go:77:9: &buf escapes to heap
./dist/pert.go:30:3: x does not escape
`

func TestEscapesGate(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "escapes")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	list := filepath.Join(t.TempDir(), "hotpath.txt")
	if err := os.WriteFile(list, []byte("# comment\nwalk.go\ndist/sampler.go\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin, list)
	cmd.Stdin = strings.NewReader(sample)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected failure:\n%s", out)
	}
	if n := strings.Count(string(out), "hot path allocation"); n != 2 {
		t.Fatalf("want 2 findings, got %d:\n%s", n, out)
	}
	if err := os.WriteFile(list, []byte("compile.go\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd = exec.Command(bin, list)
	cmd.Stdin = strings.NewReader(sample)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("unexpected failure: %v\n%s", err, out)
	}
}

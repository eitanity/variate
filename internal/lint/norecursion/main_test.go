package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFindsRecursion(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "norecursion")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin, "./...")
	cmd.Dir = "testdata/rec"
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected failure, got success:\n%s", out)
	}
	s := string(out)
	for _, want := range []string{"rec.Fact", "rec.Ping", "rec.Pong"} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %s in:\n%s", want, s)
		}
	}
	if strings.Contains(s, "rec.Fine") {
		t.Errorf("Fine reported:\n%s", s)
	}
	if !strings.Contains(s, "Scaled") {
		t.Errorf("wrapper not reported without the allowlist:\n%s", s)
	}
	cmd = exec.Command(bin, "-wrappers", "wrappers.txt", "./...")
	cmd.Dir = "testdata/rec"
	out, _ = cmd.CombinedOutput()
	if strings.Contains(string(out), "Scaled") {
		t.Errorf("wrapper reported despite the allowlist:\n%s", out)
	}
}

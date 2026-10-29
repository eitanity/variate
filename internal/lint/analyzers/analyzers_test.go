package analyzers_test

import (
	"testing"

	"code.eitanity.com/variate/internal/lint/analyzers"
	"golang.org/x/tools/go/analysis/analysistest"
)

const mod = "code.eitanity.com/variate"

func TestLoopBound(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), analyzers.LoopBound, mod, mod+"/config")
}

func TestNoFuncType(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), analyzers.NoFuncType, mod+"/dist")
}

func TestNoAny(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), analyzers.NoAny, mod+"/stats", mod+"/config", mod+"/cmd/x")
}

func TestUnordered(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), analyzers.Unordered, mod+"/um")
}

func TestStmts(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), analyzers.Stmts, mod+"/st", mod+"/config")
}

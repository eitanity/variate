package analyzers

import (
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// modulePath is the main module every rule applies to.
const modulePath = "code.eitanity.com/variate"

// enginePkgs are the packages the engine-only checks apply to.
var enginePkgs = [...]string{modulePath, modulePath + "/dist", modulePath + "/stats"}

// inModule reports whether the package being analysed is own code.
func inModule(pass *analysis.Pass) bool {
	p := pass.Pkg.Path()
	return p == modulePath || strings.HasPrefix(p, modulePath+"/")
}

// inEngine reports whether the package is the root package, dist, stats or
// a subpackage of the latter two.
func inEngine(pass *analysis.Pass) bool {
	p := pass.Pkg.Path()
	if p == modulePath {
		return true
	}
	for _, e := range enginePkgs[1:] {
		if p == e || strings.HasPrefix(p, e+"/") {
			return true
		}
	}
	return false
}

// isConfig reports whether the package is config, the one place that may
// use type switches and assertions.
func isConfig(pass *analysis.Pass) bool {
	return pass.Pkg.Path() == modulePath+"/config"
}

// isTest reports whether pos lies in a _test.go file.
func isTest(pass *analysis.Pass, pos token.Pos) bool {
	return strings.HasSuffix(pass.Fset.Position(pos).Filename, "_test.go")
}

// atFile reports whether pos lies in file base of the package pkg.
func atFile(pass *analysis.Pass, pos token.Pos, pkg, base string) bool {
	if pass.Pkg.Path() != pkg {
		return false
	}
	name := pass.Fset.Position(pos).Filename
	return name == base || strings.HasSuffix(name, "/"+base)
}

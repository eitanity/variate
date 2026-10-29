package dist

type table struct {
	hook func() int // want "function-typed struct field"
	n    int
}

var pkgFn func() // want "package-level variable of function type"

var _ = pkgFn
var _ = table{}

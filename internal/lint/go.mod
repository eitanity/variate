// Tool module, so that its dependencies stay out of the main module.
module code.eitanity.com/variate/internal/lint

go 1.27

toolchain go1.27.2

require golang.org/x/tools v0.52.0

require (
	golang.org/x/mod v0.42.0 // indirect
	golang.org/x/sync v0.24.0 // indirect
)

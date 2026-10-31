# `make ci` runs the every-push checks, `make weekly` the weekly set.
# The workflows call these targets. Tool versions are pinned here.

GOLANGCI_LINT_VERSION := v2.14.0
GOIMPORTS_VERSION     := v0.52.0
GOVULNCHECK_VERSION   := latest
TINYGO_VERSION        := 0.42.0
BIN                   := $(abspath bin)
PKGS                  := ./...
HOTPATH               := internal/lint/hotpath.txt

.PHONY: all ci weekly tools build test test32 race alloc lint fmt vet custom-lint tinygo tinygo-version wasm demo golden wasm-smoke wasm-test \
        norecursion escapes fuzz fuzz-long stat bench vuln release clean

all: ci

ci: verify fmt vet lint custom-lint escapes race alloc test32 tinygo vuln

weekly: stat fuzz-long bench

# ---- toolchain ------------------------------------------------------------

tools: $(BIN)/golangci-lint $(BIN)/goimports $(BIN)/govulncheck $(BIN)/variatevet $(BIN)/norecursion $(BIN)/escapes

$(BIN)/golangci-lint:
	GOBIN=$(BIN) go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

$(BIN)/goimports:
	GOBIN=$(BIN) go install golang.org/x/tools/cmd/goimports@$(GOIMPORTS_VERSION)

$(BIN)/govulncheck:
	GOBIN=$(BIN) go install golang.org/x/vuln/cmd/govulncheck@$(GOVULNCHECK_VERSION)

$(BIN)/variatevet $(BIN)/norecursion $(BIN)/escapes: $(wildcard internal/lint/*/*.go internal/lint/*/*/*.go)
	cd internal/lint && go build -o $(BIN)/variatevet ./cmd/variatevet \
	  && go build -o $(BIN)/norecursion ./norecursion \
	  && go build -o $(BIN)/escapes ./escapes

verify:
	go mod verify
	go mod tidy -diff
	cd internal/lint && go mod verify && go mod tidy -diff
	@bad="$$(git ls-files -z | xargs -0 file --mime-type | grep -E 'application/(x-executable|x-pie-executable|x-sharedlib|x-mach-binary|x-dosexec|wasm|x-object|x-archive)' | cut -d: -f1)"; \
	  if [ -n "$$bad" ]; then echo "tracked binary files:"; echo "$$bad"; exit 1; fi

build:
	go build -trimpath $(PKGS)
	GOOS=js GOARCH=wasm go build -trimpath $(PKGS)
	GOOS=wasip1 GOARCH=wasm go build -trimpath $(PKGS)

# Rewrites the frozen reports of the example and demo models after an
# intended behaviour change; review the diff before committing.
golden:
	go test -run TestGoldenReports ./internal/suite -update

# ---- browser demo -----------------------------------------------------------

# The demo's engine is built with TinyGo; its wasm_exec.js is TinyGo's
# own, not Go's.
tinygo-version:
	@tinygo version 2>/dev/null | grep -q "tinygo version $(TINYGO_VERSION) " || { echo "tinygo $(TINYGO_VERSION) is required: https://tinygo.org/getting-started/install/"; exit 1; }

wasm: tinygo-version
	tinygo build -o web/variate.wasm -target wasm -no-debug -opt 2 ./cmd/variate-wasm
	cp "$$(tinygo env TINYGOROOT)/targets/wasm_exec.js" web/wasm_exec.js

# The wasm entry point builds under TinyGo.
tinygo: tinygo-version
	tinygo build -o $(BIN)/variate.tinygo.wasm -target wasm -no-debug -opt 2 ./cmd/variate-wasm

demo: wasm
	go run ./cmd/variate-demo

# Runs the wasm engine under node and compares its report with the CLI's.
wasm-smoke: wasm
	go run ./cmd/variate run -f examples/compound.yaml -n 2000 --seed 5 --block-periods 50 --format json --analytic --node-hist > web/native.json
	node web/smoke.js web/native.json

# The engine's tests as wasm under the wazero runtime (WASI), with the
# golden reports comparing wasm output against the natively generated
# natively generated files. wazero is a tool, not a dependency:
# go install github.com/tetratelabs/wazero/cmd/wazero@latest
WASM_TESTS = TestGolden|TestErlang|TestRepair|TestChain|TestOutage|TestSharedFactor|TestTransformCompound|TestTransformPERT|TestTransformWorkBudget|TestDifferentialWorkers|TestDifferentialSimulate|TestFixedRunBlocks|TestCap
wasm-test:
	PATH="$$PATH:$$(go env GOPATH)/bin:$$(go env GOROOT)/lib/wasm" GOOS=wasip1 GOARCH=wasm GOWASIRUNTIME=wazero go test -count=1 -timeout 60m ./dist ./stats ./config ./report
	PATH="$$PATH:$$(go env GOPATH)/bin:$$(go env GOROOT)/lib/wasm" GOOS=wasip1 GOARCH=wasm GOWASIRUNTIME=wazero go test -count=1 -timeout 60m -run '$(WASM_TESTS)' ./internal/suite ./examples

# ---- formatting, vet, lint ------------------------------------------------

fmt: $(BIN)/goimports
	@out="$$(gofmt -l . ; $(BIN)/goimports -l .)"; if [ -n "$$out" ]; then echo "$$out"; exit 1; fi

vet: $(BIN)/golangci-lint
	go vet $(PKGS)
	cd internal/lint && go vet ./...
	GOOS=js GOARCH=wasm go vet ./cmd/variate-wasm
	GOOS=js GOARCH=wasm $(BIN)/golangci-lint run ./cmd/variate-wasm/
	@! grep -rn --include='*.go' --exclude-dir=bin --exclude-dir=.git '^//go:build' . || { echo "build constraints are not allowed"; exit 1; }
	@test "$$(find . -path ./bin -prune -o -name '*.go' -print | grep -E '_(js|wasm|linux|darwin|windows|amd64|arm64)(_test)?\.go$$')" = "./cmd/variate-wasm/main_js.go" || { echo "only cmd/variate-wasm/main_js.go may be build-constrained by its name"; exit 1; }

lint: $(BIN)/golangci-lint
	$(BIN)/golangci-lint run $(PKGS)

custom-lint: $(BIN)/variatevet $(BIN)/norecursion
	go vet -vettool=$(BIN)/variatevet $(PKGS)
	GOOS=js GOARCH=wasm go vet -vettool=$(BIN)/variatevet ./cmd/variate-wasm
	$(BIN)/norecursion -wrappers internal/lint/wrappers.txt $(PKGS)
	cd internal/lint && go test ./...

norecursion: $(BIN)/norecursion
	$(BIN)/norecursion -wrappers internal/lint/wrappers.txt $(PKGS)

escapes: $(BIN)/escapes
	go build -gcflags=-m . ./dist ./stats 2>&1 | $(BIN)/escapes $(HOTPATH)

# ---- tests ------------------------------------------------------------------

test:
	go test -count=1 $(PKGS)

# The suite with a 32-bit int, the width TinyGo's wasm target has, on
# four workers so that the footprint fits a 32-bit address space.
test32:
	GOARCH=386 GOMAXPROCS=4 go test -count=1 -timeout 60m $(PKGS)

# The race detector over the tests that drive the worker pool: its
# lifecycle, cancellation and faults, the stream the finaliser delivers,
# adaptive stops at batch boundaries and runs across worker counts. The
# root package starts the only goroutines; elsewhere the detector has
# nothing to find.
RACE_TESTS := ^Test(Workspace|Fault|Stream|Adaptive|Engine|FixedRun|PacedCancel|DifferentialWorkers)

race:
	go test -race -count=1 -timeout 60m -run '$(RACE_TESTS)' . ./internal/suite

alloc:
	go test -count=1 -run 'Alloc|Churn' $(PKGS)

fuzz:
	go test -run=^$$ -fuzz=FuzzLoad -fuzztime=60s ./config
	go test -run=^$$ -fuzz=FuzzCompile -fuzztime=60s ./internal/suite

fuzz-long:
	go test -run=^$$ -fuzz=FuzzLoad -fuzztime=30m ./config
	go test -run=^$$ -fuzz=FuzzCompile -fuzztime=30m ./internal/suite

stat:
	VARIATE_STAT=1 go test -count=1 -timeout 6h $(PKGS)

bench:
	go test -run=^$$ -bench . -benchmem -count=10 $(PKGS) | tee bench.out

# ---- vulnerabilities --------------------------------------------------------

vuln: $(BIN)/govulncheck
	$(BIN)/govulncheck $(PKGS)

# ---- release ----------------------------------------------------------------

release:
	CGO_ENABLED=0 go build -trimpath -ldflags=-buildid= -o $(BIN)/variate ./cmd/variate
	cd $(BIN) && sha256sum variate > variate.sha256

clean:
	rm -rf ./bin bench.out

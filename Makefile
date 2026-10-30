# `make ci` runs the every-push checks, `make nightly` the nightly set.
# The workflows call these targets. Tool versions are pinned here.

GOLANGCI_LINT_VERSION := v2.14.0
GOIMPORTS_VERSION     := v0.52.0
GOVULNCHECK_VERSION   := latest
BIN                   := $(abspath bin)
PKGS                  := ./...
HOTPATH               := internal/lint/hotpath.txt

.PHONY: all ci nightly tools build test race alloc lint fmt vet custom-lint wasm demo golden wasm-smoke wasm-test \
        norecursion escapes fuzz fuzz-long stat bench vuln release clean

all: ci

ci: verify fmt vet lint custom-lint escapes race alloc fuzz vuln

nightly: stat fuzz-long bench

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

wasm:
	GOOS=js GOARCH=wasm go build -trimpath -o web/variate.wasm ./cmd/variate-wasm
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" web/wasm_exec.js

demo: wasm
	go run ./cmd/variate-demo

# Runs the wasm engine under node and compares its report with the CLI's.
wasm-smoke: wasm
	go run ./cmd/variate run -f examples/compound.yaml -n 2000 --seed 5 --block-periods 50 --format json --analytic > web/native.json
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

race:
	VARIATE_RACE=1 go test -race -count=1 -timeout 60m $(PKGS)

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

# Benchmark baseline

`baseline.txt` is `go test -bench . -benchmem -count=5` over the engine's
suite, `dist` and `stats` on the development machine (32 cores) at the
end of the performance pass. Compare a new run with

    go test -run '^$' -bench . -benchmem -count=5 ./internal/suite ./dist ./stats > bench.out
    benchstat bench/baseline.txt bench.out

The nightly workflow does the same against the previous night's run and
fails on a regression over 5% in `BenchmarkPERT`, `BenchmarkWalk` or
`BenchmarkChain`.

// Command variate-demo serves the browser demo: the page under web/,
// the engine compiled to WebAssembly, and the example models. Build the
// engine with make wasm first, then go run ./cmd/variate-demo.
package main

import (
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	addr := flag.String("addr", "127.0.0.1:8000", "address to listen on")
	dir := flag.String("dir", "web", "directory holding the page and variate.wasm")
	flag.Parse()
	if _, err := os.Stat(filepath.Join(*dir, "variate.wasm")); err != nil {
		return errors.New("no variate.wasm in " + *dir + ": run make wasm first")
	}
	fmt.Printf("variate demo at http://%s/\n", *addr)
	srv := &http.Server{Addr: *addr, Handler: handler(*dir), ReadHeaderTimeout: 10 * time.Second}
	if err := srv.ListenAndServe(); err != nil {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}

// handler serves the directory with the wasm content type and no caching,
// so a rebuilt engine is picked up on reload.
func handler(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".wasm") {
			w.Header().Set("Content-Type", "application/wasm")
		}
		w.Header().Set("Cache-Control", "no-store")
		files.ServeHTTP(w, r)
	})
}

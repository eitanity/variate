# Mutation testing

`mutate.py` applies one operator at one site of each target file
(boundary `<`/`<=`, sign flips, dropped negations, loosened tolerances),
runs that file's fast test subset with fail-fast, restores the file, and
reports killed and surviving mutants. `recheck.py` reruns the survivors
of `run.log` against a broader subset. Both write to the working tree
while they run, so run nothing else against it meanwhile.

    python3 internal/mutate/mutate.py 20 > run.log
    python3 internal/mutate/recheck.py > recheck.log

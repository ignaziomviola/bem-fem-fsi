# The vendored fluid solver: provenance and how to check it

The panel method in this repository is **not** written here. It is a verbatim
copy of

> https://github.com/ignaziomviola/time-dependent-free-wake-panel-method
> commit `d9d1f657e9191cf57959558685a4f06b3b46923c`
> ("Record the fresh-clone reproduction check", 9 August 2026)

taken at the moment the finite element module was begun, and it is not modified
here in any way. The upstream repository owns the fluid physics, its three
design documents and its verification programme; this one owns the structure
and the coupling.

## Which files are vendored

| file | role |
| --- | --- |
| `thick_panel_wing.py` | the steady physics and the solver API |
| `unsteady_wing.py` | the time loop, prescribed motions, harmonic fit |
| `freewake_kernels.py` | Biot-Savart and the input helpers |
| `make_thick_sample_inputs.py` | mesh and onset generators |
| `test_thick_panel_wing.py` | the steady suite, 14 tests |
| `test_unsteady_wing.py` | the unsteady suite, 20 tests |
| `verify_analytic.py` | cylinder and Joukowski |
| `verify_unsteady.py` | added mass, steady limit, Wagner, Theodorsen |

The fluid solver's own documents are vendored as well, in
[fluid/](fluid/), and are covered by the same manifest:

| file | role |
| --- | --- |
| `docs/fluid/DESIGN.md` | the steady formulation and every sign convention |
| `docs/fluid/UNSTEADY.md` | the time-dependent formulation and the shedding scheme |
| `docs/fluid/ARCHITECTURE.md` | the fluid block boundaries and public signatures |
| `docs/fluid/README.md` | the fluid solver's own usage guide |

They are copied rather than linked for the same reason the code is: this
repository should carry the authority for the physics it runs, not a URL to it.
`docs/fluid/INDEX.md` is written here, is NOT vendored, and records the two
wrinkles of reading those documents from this repository - the README's
relative links assume a different root, and the architecture document describes
this repository as future work.

## Why copy rather than depend

This is the precedent `freewake_kernels.py` already sets upstream, where seven
functions of `panel_wing.py` were duplicated so that a fresh clone runs on its
own, "collected in one module so that the duplication stays visible". The same
reasoning applies one level up. A fresh clone of `bem-fem-fsi` runs the entire
fluid verification programme without a second checkout, a submodule pointer or
a path variable, and the two vendored suites above are part of this
repository's own test run: a divergence in behaviour would fail them here.

The cost is that upstream fixes do not arrive automatically. That is what the
commit hash above is for, and updating is a `git diff` against it.

## The manifest

`fluid_manifest.txt` records the SHA-256 of every vendored file.
`test_vendored.py` recomputes them and fails if any file has been edited. It
checks local integrity, not agreement with upstream - nothing here can reach
GitHub during a test run - but local integrity is the property that matters,
because the one thing that must never happen is a quiet local fix to the fluid
code that makes the coupling results irreproducible against the repository they
are supposed to have come from.

If the fluid code genuinely needs to change, change it upstream, re-vendor, and
update the hash and the commit reference above in the same commit.

## Refreshing the copy

```bash
git clone https://github.com/ignaziomviola/time-dependent-free-wake-panel-method /tmp/upstream
for f in $(cut -d' ' -f3 fluid_manifest.txt); do diff -u "$f" "/tmp/upstream/$f"; done
```

Empty output means the copy is current with whatever `/tmp/upstream` is checked
out at. After copying anything across, regenerate the manifest with

```bash
python3 test_vendored.py --write
```

and run both vendored suites before committing.

# The fluid solver's own documents

The four files beside this one are **verbatim copies** from
https://github.com/ignaziomviola/time-dependent-free-wake-panel-method at the
commit recorded in [../FLUID.md](../FLUID.md). They are the authoritative
specification of the panel method carried into this repository, and they are
not maintained here: their SHA-256 is in `fluid_manifest.txt` and
`test_vendored.py` fails if one is edited. This index is the only file in this
directory written for this repository.

| document | what it owns |
| --- | --- |
| [DESIGN.md](DESIGN.md) | the steady formulation, every sign convention, the steady verification results |
| [UNSTEADY.md](UNSTEADY.md) | the time-dependent formulation, the shedding scheme, the unsteady verification |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the fluid block boundaries, its state contract, its public signatures and its extension hooks |
| [README.md](README.md) | the fluid solver's own usage guide |

Read `DESIGN.md` before changing anything that touches a sign, `UNSTEADY.md`
before anything that touches the wake or dmu/dt, and `ARCHITECTURE.md` before
anything that touches the solver API. All three record why several choices that
look like accidents are not.

## Two wrinkles of reading them here rather than upstream

**Relative links in `README.md` point at a root that is one level up.** It was
written to sit at the top of its own repository, so its `docs/DESIGN.md` links
resolve to `docs/fluid/docs/DESIGN.md` here, which does not exist; the file it
means is `DESIGN.md` beside it. The links between `DESIGN.md`, `UNSTEADY.md`
and `ARCHITECTURE.md` are already relative to each other and resolve correctly.
Nothing is edited to fix this, because a byte-identical copy is worth more than
four working links.

**`ARCHITECTURE.md` describes this repository as future work.** Its extension
(2), "partitioned coupling with a finite element structural code", is what
`fsi_driver.py` and `fem_solid.py` are, and its hooks 8, 9 and 10 are the ones
they use. The status line and the hook table were written before that existed
and still say so. [../ARCHITECTURE.md](../ARCHITECTURE.md) picks up where it
leaves off and its closing section records what the extension actually cost
against what that document predicted.

Extension (3), ventilation and cavitation, is built here and
`docs/VENTILATION.md` owns it. It uses four of the hooks that document left
unused: the mirror-covariant corner-pure kernels, for a free-surface image
realised by mesh doubling; the per-panel doublet/source unknown swap in
`assemble_system`, for the cavity's prescribed pressure and solved thickness;
`ds_wrap`, for the cavity integrals; and the trailing-edge fold's handling of a
prescribed `mu`, for a cavity that reaches the trailing edge. One of that
document's predictions needed amending, and the amendment is recorded in
`docs/ARCHITECTURE.md`: a mirror image cannot be convected, only projected.
Extension (3) remains future work upstream.

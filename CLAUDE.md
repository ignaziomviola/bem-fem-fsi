# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A partitioned fluid-structure interaction code. The fluid is the free-wake
source-doublet panel method of
https://github.com/ignaziomviola/time-dependent-free-wake-panel-method, carried
in **verbatim and never modified here**; the structure is a three-dimensional
finite element solid written here; and the coupling is written here. It builds
extension (2) of that repository's `docs/ARCHITECTURE.md`, which specified a
partitioned coupling with a finite element code and left its hooks unused.

## The vendored fluid code is not maintained here

`thick_panel_wing.py`, `unsteady_wing.py`, `freewake_kernels.py`,
`make_thick_sample_inputs.py`, the four test and verification files that go with
them, and their four documents in `docs/fluid/` are byte-for-byte copies.
`test_vendored.py` checks all twelve against the SHA-256 in
`fluid_manifest.txt` and fails if one is edited.

**Never fix the fluid code here.** Fix it upstream, re-vendor, regenerate the
manifest with `python3 test_vendored.py --write`, and update the commit
reference in `docs/FLUID.md` in the same commit.

Read `docs/fluid/DESIGN.md` before changing anything that touches a sign,
`docs/fluid/UNSTEADY.md` before anything that touches the wake or dmu/dt, and
`docs/fluid/ARCHITECTURE.md` before anything that touches the solver API. They
record why several things that look like accidents are not.
`docs/fluid/INDEX.md` is written here, is excluded from the manifest, and says
how to read those four from this repository rather than from upstream.

## Authoritative documents

`docs/FEM.md` owns the structural formulation, every convention that pins a
sign or a factor, and the structural verification results with measured
numbers. `docs/COUPLING.md` owns the interface operators, the partitioned
schemes, the two resolution constraints and the coupled verification results.
`docs/ARCHITECTURE.md` owns the block boundaries, the state contracts, the
public signatures and the extension hooks, and its final section records what
the coupling extension cost against what the fluid architecture predicted.
Update them when behaviour changes - the tables hold actuals, not targets.

## Commands

```bash
# tests. Counts are 44 / 21 / 4 / 14 / 20; anything less means a suite errored.
python3 test_fem.py                       # structural fast set, ~0.5 s
FEM_SLOW=1 python3 test_fem.py            # + convergence studies, ~5 s
python3 test_fsi.py                       # coupling fast set, ~1 s
FSI_SLOW=1 python3 test_fsi.py            # + coupled physics, ~23 s
python3 test_vendored.py                  # the fluid copies are unmodified
python3 test_vendored.py --write          # regenerate the manifest after a re-vendor
python3 test_thick_panel_wing.py          # vendored steady suite, unchanged
python3 test_unsteady_wing.py             # vendored unsteady suite, unchanged

# one test, one class, or one module - the standard unittest paths
python3 -m unittest test_fem.TestMaterialSeam.test_rotating_the_material_rotates_the_structure -v
python3 -m unittest test_fsi.TestTransfer -v
FSI_SLOW=1 python3 -m unittest test_fsi.TestSlow -v

# verification programmes, which print the tables the documents tabulate
python3 verify_fem.py                     # all nine structural cases, ~30 s
python3 verify_fem.py --case F4           # one case: F1..F9
python3 verify_fsi.py                     # all six coupled cases, ~30 s
python3 verify_fsi.py --case C3           # one case: C1..C6
python3 verify_fsi.py --quick             # the cheap ones only (C1, C2, C6)
python3 verify_analytic.py                # vendored: cylinder and Joukowski
python3 verify_unsteady.py                # vendored: Wagner, Theodorsen (~1 h)

# command-line drivers; each ends in plt.show(), hence MPLBACKEND=Agg in batch
python3 make_thick_sample_inputs.py       # meshes and onset profiles
python3 fem_mesh.py                       # report and check the mesh builders
MPLBACKEND=Agg python3 fem_solid.py       # cantilever, against the beam solution
printf "6.0\n0.12\n5.0\n2.0e8\n1200\n1000\n0.05\n40\n" | MPLBACKEND=Agg python3 fsi_driver.py
```

pytest is **not** installed and must not be introduced: the tests use the
standard library `unittest`, and the slow sets are gated by environment
variables rather than markers. numpy and matplotlib are the only dependencies
and scipy is deliberately absent, which is why the triangular solves and the
generalised eigenproblem are written out in `fem_solid.py`.

`pyflakes` is clean over everything written here and is worth keeping so.

## Architecture

`fem_*` is the structure, `fsi_*` the coupling, and the vendored fluid files
keep their own names. Nothing imports downward:

```
fem_materials  <-  fem_solid  <-  fem_mesh (builders only)
                        ^
thick_panel_wing  <-  fsi_transfer  <-  fsi_driver  ->  unsteady_wing
```

`fem_solid.py` holds the structural physics in ten dependency-ordered sections;
`fsi_transfer.py` is the only module that imports both sides.

The three discipline rules of the upstream module carry over and each is
load-bearing: nothing imports downward; every block is a pure function over
explicit arguments with all mutable data in an explicit state dict, so two
states coexist (the verification programmes build nine independent cases in one
process); and prompts, prints and figures live only under `main()`.

### The four states, and who owns each

This is the part that cannot be read off any single file.

- **`fluid`** - the upstream panel-method state. Its contract is
  `docs/fluid/ARCHITECTURE.md`. `update_points` is the ONLY geometry entry
  point and `solve(wake="frozen")` the only per-step solve.
- **`model`** - structural topology, materials and a cache of the reference
  configuration (`B`, `Ba`, `w*detJ`, and lazily the element tangents). Built
  once by `build_model` and never invalidated by displacement.
- **`sstate`** - the committed structural level `(u, v, a, f, history)`, the
  constraints, the assembled operators and the cached Cholesky factor.
- **`transfer`** - `rows`, `weights`, the weld map and the fluid mesh shape.
  Built ONCE on the two reference configurations by `build_transfer` and never
  rebuilt: that is what makes the coupling residual a fixed function of the
  displacement, which quasi-Newton acceleration requires.

`solve_static` and `step_dynamic` return a **level** dict and mutate nothing;
`commit(sstate, level, f_ext)` is the only thing that advances the state of
record. A strongly coupled step re-solves the same time level several times.

### The coupled step

Spans `fsi_driver.time_march_fsi` and the fluid module, and the order matters:

```
shed(wake, ...)                     ONCE per step, before any subiteration
repeat:
    update_points -> solve(wake="frozen") -> get_loads(dphi_dt)
    -> structural_forces -> step_dynamic -> accelerate
commit the structural level AND the doublet strengths
```

### Diagnostics to reach for when a coupled run misbehaves

`fsi_driver.steps_per_period` and `unsteady_wing.convection_per_step` are the
two resolution measures; `fsi_driver.added_mass_ratio` says whether staggered
coupling could ever work; `fsi_transfer.conservation_report` separates a
transfer defect from a solver defect; `fsi_driver.divergence_pressure` locates
a static instability directly; `fsi_driver.growth_rate` fits a marched
response.

## Deliberate decisions that look like defects

Each was found by a failing gate or is required by one. Reverting any of them
reintroduces a real bug.

- **The internal force is always the integral of B^T sigma, never K u**, even
  though the two agree exactly for a linear material and a test asserts they do
  to 1e-11. The other way puts the constitutive law back inside the element and
  makes a nonlinear material a rewrite.
- **`step_dynamic` is a pure function of the committed level; `commit` is
  separate.** A strongly coupled step re-solves the same time level several
  times with different loads, and an integrator that has advanced its own state
  cannot be asked twice.
- **The Rayleigh damping matrix `C` and the material rate tangent `C_mat` are
  kept apart.** `C` enters equilibrium directly; `C_mat` is used only to build
  the effective dynamic operator, because the viscous stress already reaches
  equilibrium through `internal_force(v=...)`. Adding `C_mat` to `C` counts it
  twice.
- **Both Newton loops measure the residual against the FIRST residual of the
  call, not the applied force, and stop on stagnation.** Free vibration has no
  applied force, and a scale taken from it collapses to zero and runs every
  step silently to the iteration cap. The floor near 1e-9 is round-off in the
  integral, not an unconverged solve.
- **The incompatible-mode gradient uses the Jacobian at the element CENTRE,
  scaled by detJ0/detJ.** Without the Taylor-Beresford-Wilson correction the
  patch test fails on a distorted mesh.
- **The inverse Jacobian is contracted on its second index** in the element
  gradient. The other contraction is identical on a rectangular grid and wrong
  everywhere else; only the distorted patch test sees it.
- **`shed` is called once per time step, outside the coupling subiteration.** It
  prepends a wake row and convects, mutating the wake irreversibly. Everything
  inside the subiteration is `update_points` plus `solve(wake="frozen")`.
- **The doublet history is seeded with the acyclic level and advances only on
  commit.** Without the seed the first step's dmu/dt is zero and the added mass
  lags by one level for the whole march; without the deferral it depends on the
  iteration path.
- **The Aitken factor is not clipped to a small range.** The optimal factor for
  a dominant eigenvalue of 0.98 is 50, so a clip to plus or minus two removes
  exactly the acceleration it exists to supply.
- **The lumping of panel forces reuses `tpw.centroid_weights`.** It is a
  partition of unity reproducing the collocation point, so force and moment are
  conserved exactly, and it is the adjoint of the operator the fluid uses to
  sample body motion.

## Expected results that are not errors

- **A two-step ripple in the lift over a perfectly smooth quasi-steady lift** is
  the second resolution constraint being violated: unresolved structural modes
  ringing undamped, tiny in displacement and large in acceleration, reaching the
  pressure through dmu/dt. Resolve the mode, start from the static equilibrium,
  or lower `rho_inf`. `steps_per_period` reports it and the driver warns.
- **A deforming surface is never `rigid`**, so every coupling subiteration pays
  a full body reassembly. That is the accepted cost contract, not an oversight,
  and it is why the verification cases use 72 to 128 panels.
- **The cantilever is 1 to 2% stiffer than the beam solution**, converging from
  below: the clamped end face restrains Poisson contraction and warping.
- **A cantilever of this section has an EDGEWISE mode between its first and
  second flapwise ones.** Comparing the first three eigenvalues with a flapwise
  beam reference compares the wrong things; select flapwise modes by their
  shape, as `verify_fem.py` and `fem_solid.main()` do.
- **The solid foil has wedge-degenerate elements at its leading and trailing
  edges.** The vertical extent of an aerofoil section is zero there.
- **A solid foil clamped at midspan does not diverge**: its torsional stiffness
  is far above its bending stiffness and bending of a straight unswept wing
  produces no twist. The divergence gate uses an elastically supported wing.
- **A lifting run in still water eventually destabilises**: there is no mean
  flow to carry shed vorticity away, so the sheet piles up at the trailing
  edge. Added-mass work is done without a wake, as the fluid's own sphere gate
  is.
- All the expected results of the upstream documents still apply.

## Extending the materials

This is what the structural side was shaped around, so it is the extension most
likely to be asked for. A new constitutive law is one class in
`fem_materials.py` implementing

```python
response(strain, strain_rate=None, history=None, dt=None, orientation=None)
    -> {"stress", "tangent", "rate_tangent", "history"}
```

with `n_history` internal variables per Gauss point. Nothing above it changes:
`internal_force` already threads `history` and `dt`, both solvers are already
Newton loops on a residual, and the incompatible-mode amplitudes are already
resolved by a separate element-level solve in `element_alpha`, which is where a
local Newton iteration goes. The one real consequence is cost - a material
without `constant_tangent` invalidates the `constant_operators` cache and the
tangent is rebuilt per iteration.

`OrthotropicElastic` and `KelvinVoigt` exist to keep the anisotropy path and
the rate-and-history path exercised rather than promised; without them
`orientation`, `strain_rate`, `dt` and `history` would all be dead arguments.
`test_fem.TestMaterialSeam` gates the model-level plumbing - orientation
broadcast, per-element orientation reaching the damping, and two materials
through `mat_id`.

Kinematics are linear (small displacement, small strain) and there is no shell
element; both are recorded as open items in `docs/FEM.md`.

## Meshes

Fluid meshes are unchanged from upstream: a `thick-wrap-1` archive with
`points` of shape `(nwrap+1, nspan+1, 3)`, sharp closed trailing edge, both
tips pinched. Structural meshes are built by `fem_mesh.py` and need not match:
`solid_foil_mesh` lofts a solid foil between the paired chordwise stations so
its outer surface **is** the wetted surface, and the transfer handles the
general non-matching case and reports the projection offset.

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
`make_thick_sample_inputs.py` and the four test and verification files that go
with them are byte-for-byte copies. `test_vendored.py` checks their SHA-256
against `fluid_manifest.txt` and fails if one is edited. **Never fix the fluid
code here.** Fix it upstream, re-vendor, regenerate the manifest with
`python3 test_vendored.py --write`, and update the commit reference in
`docs/FLUID.md` in the same commit. Those three files' own documents are vendored
alongside them in `docs/fluid/` and are covered by the same manifest: read
`docs/fluid/DESIGN.md` before changing anything that touches a sign,
`docs/fluid/UNSTEADY.md` before anything that touches the wake or dmu/dt, and
`docs/fluid/ARCHITECTURE.md` before anything that touches the solver API. They
record why several things that look like accidents are not.

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
python3 test_fem.py                       # structural fast set, ~0.2 s
FEM_SLOW=1 python3 test_fem.py            # + convergence studies, ~5 s
python3 test_fsi.py                       # coupling fast set, ~1 s
FSI_SLOW=1 python3 test_fsi.py            # + coupled physics, ~23 s
python3 test_vendored.py                  # the fluid copies are unmodified
python3 test_thick_panel_wing.py          # vendored steady suite, unchanged
python3 test_unsteady_wing.py             # vendored unsteady suite, unchanged
python3 verify_fem.py                     # structural tables, ~30 s
python3 verify_fsi.py                     # coupled tables, ~30 s
python3 make_thick_sample_inputs.py       # meshes and onset profiles
printf "6.0\n0.12\n5.0\n2.0e7\n1200\n1000\n0.05\n40\n" | MPLBACKEND=Agg python3 fsi_driver.py
```

pytest is **not** installed and must not be introduced: the tests use the
standard library `unittest`. numpy and matplotlib are the only dependencies and
scipy is deliberately absent, which is why the triangular solves and the
generalised eigenproblem are written out in `fem_solid.py`. `MPLBACKEND=Agg` is
needed only for the command-line paths, which end in `plt.show()`.

## Architecture

`fem_materials.py` is the constitutive seam; `fem_solid.py` holds the
structural physics in ten dependency-ordered sections and the cantilever
driver; `fem_mesh.py` builds structural meshes; `fsi_transfer.py` owns the two
interface operators; `fsi_driver.py` owns the coupling loops and the figures.
`fem_*` is the structure, `fsi_*` the coupling, and the eight vendored fluid
files keep their own names.

The three discipline rules of the upstream module carry over and each is
load-bearing: nothing imports downward; every block is a pure function over
explicit arguments with all mutable data in an explicit state dict, so two
states coexist; and prompts, prints and figures live only under `main()`.

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
  a full body reassembly. That is the accepted cost contract, not an oversight.
- **The cantilever is 1 to 2% stiffer than the beam solution**, converging from
  below: the clamped end face restrains Poisson contraction and warping.
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

## Meshes

Fluid meshes are unchanged from upstream: a `thick-wrap-1` archive with
`points` of shape `(nwrap+1, nspan+1, 3)`, sharp closed trailing edge, both
tips pinched. Structural meshes are built by `fem_mesh.py` and need not match:
`solid_foil_mesh` lofts a solid foil between the paired chordwise stations so
its outer surface **is** the wetted surface, and the transfer handles the
general non-matching case and reports the projection offset.

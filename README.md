# Fluid-structure interaction: a free-wake panel method coupled to a finite element solid

A Python code computing the unsteady loads on a deformable finite wing in
incompressible flow, and the deformation those loads produce. The fluid is a
free-wake source-doublet panel method that sheds one wake strip per time step
and convects it as material fluid, so the added mass is predicted by the
unsteady Bernoulli relation rather than modelled. The structure is a
three-dimensional finite element solid in which the stress and the tangent
modulus are evaluated at each Gauss point by a material object, so anisotropic
and inelastic materials extend the code rather than rewrite it. The two are
coupled by a partitioned scheme with a conservative interface and quasi-Newton
acceleration, strongly enough to run in water, where the added mass exceeds the
structural mass and a staggered scheme has no stable step size.

The fluid solver is the companion repository
[time-dependent-free-wake-panel-method](https://github.com/ignaziomviola/time-dependent-free-wake-panel-method),
carried here verbatim and unmodified so that a fresh clone runs its whole
verification programme on its own; [docs/FLUID.md](docs/FLUID.md) records the
commit and how to check the copy. The steady solver both grew from is
[free-wake-thick-panel-method](https://github.com/ignaziomviola/free-wake-thick-panel-method),
and the thin lifting-surface ancestors are in
[free-wake-lifting-surface](https://github.com/ignaziomviola/free-wake-lifting-surface).

## Requirements

Python 3 with numpy; matplotlib is needed only for the figures and is imported
lazily, so importing any solver as a library pulls in no figure backend. scipy
is deliberately not used, on either side.

```bash
python3 -m pip install --user numpy matplotlib
```

## Quick start

```bash
python3 make_thick_sample_inputs.py     # meshes and onset profiles
python3 fsi_driver.py                   # a flexible wing in a stream
```

The driver prompts for the span, thickness, incidence, the material, the two
densities and the time step; pressing enter accepts the defaults. It reports
the dry natural frequencies, the transfer offset, and then the lift history,
the tip displacement, the energy balance and the coupling effort, and ends with
two figures. A batch run needs the non-interactive backend:

```bash
printf "6.0\n0.12\n5.0\n2.0e7\n1200\n1000\n0.05\n40\n" | MPLBACKEND=Agg python3 fsi_driver.py
```

The structural solver alone runs a cantilever demonstration:

```bash
python3 fem_solid.py
```

and the fluid solvers run exactly as they do upstream:

```bash
python3 thick_panel_wing.py             # steady
python3 unsteady_wing.py                # time-marched, rigid
```

## Using the code as a library

Importing any module creates no prompt, no print and no figure. A coupled time
history takes one call.

```python
import numpy as np
import thick_panel_wing as tpw
import make_thick_sample_inputs as mts
import fem_materials as fmat
import fem_mesh
import fem_solid as fes
import fsi_driver as fsd

points = mts.thick_wing_mesh(n_c=8, nspan=8, span=6.0, root_chord=1.0,
                             taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                             camber=0.0, thickness=0.12)
points = tpw.pitch_mesh(points, np.radians(5.0))
z, u = tpw.load_velocity_profile('uniform_profile.csv')
onset = tpw.make_onset(z, u)

mesh = fem_mesh.solid_foil_mesh(points, n_thick=2)       # solid foil, wetted
model = fes.build_model(mesh['nodes'], mesh['elements'], #   surface and all
                        fmat.IsotropicElastic(2.0e8, 0.3, 1200.0))

fluid, structure, transfer = fsd.init_fsi(points, onset, model,
                                          u_ref=1.0, rho=1025.0)
fes.clamp(structure, mesh['node_sets']['root'])
fes.assemble_operators(structure)

history, fluid, structure = fsd.time_march_fsi(fluid, structure, transfer,
                                               dt=0.05, nsteps=200, rho=1025.0)

history['CL'], history['s']                       # lift against reduced time
history['CL'] - history['CL_quasi_steady']        # the added-mass part of it
history['tip_z'], history['strain_energy']        # the structural response
history['sub'], history['resid']                  # the coupling effort
```

`fluid` is an ordinary panel-method state at the last level, so every fluid
diagnostic applies to it unchanged; `structure` is an ordinary structural state,
so `fes.modes`, `fes.strain_energy` and `fes.plot_deformed` apply to it.

### Static aeroelasticity

```python
fluid, structure, history = fsd.static_aeroelastic(fluid, structure, transfer,
                                                   rho=1025.0)
history['residual']            # the fixed point's contraction, iteration by iteration
```

The wake is relaxed on the first pass and frozen afterwards. Near divergence
the fixed point stops contracting, which is the answer rather than a failure;
`fsd.divergence_pressure` locates it directly as the smallest dynamic pressure
at which the coupled operator loses rank.

### Materials

```python
fmat.IsotropicElastic(e_mod, nu, rho)
fmat.OrthotropicElastic(e1, e2, e3, g12, g23, g31, nu12, nu13, nu23, rho)
fmat.KelvinVoigt(base_material, tau)
```

`build_model` takes a list of materials with a per-element `mat_id`, and a
per-element `orientation` rotating the material axes into the global frame, so
a layered or fibre-reinforced structure needs no new machinery. Writing a
fourth material - plasticity, damage, creep - means implementing one method:

```python
response(strain, strain_rate=None, history=None, dt=None, orientation=None)
    -> {"stress", "tangent", "rate_tangent", "history"}
```

with `n_history` internal variables carried per Gauss point. Nothing above it
changes: the internal force is already an integral of `B^T sigma` with the
material in the loop, and both solvers are already Newton loops on a residual.
The orthotropic and viscoelastic materials exist to keep that from being a
promise - they exercise the anisotropy path and the rate-and-history path, and
both are gated against closed-form solutions.

### Structural meshes

`fem_mesh.solid_foil_mesh(points, n_thick)` lofts a solid foil between the
paired chordwise stations of a fluid wrap mesh, so its outer surface **is** the
wetted surface and the transfer has almost nothing to interpolate.
`fem_mesh.plate_wing_mesh` and `fem_mesh.box_mesh` build plates and boxes. The
meshes need not match the fluid mesh: the transfer is built for non-matching
surfaces and reports the projection offset.

## What the coupling guarantees

- **Force and moment are conserved exactly** when panel loads are lumped onto
  the fluid mesh, because the lumping reuses the fluid's own
  `centroid_weights`, a partition of unity that reproduces the collocation
  point. It is the adjoint of the operator the fluid uses to sample body
  motion, so the two sides agree on the power as well.
- **Virtual work is conserved exactly** across the interface, because motion
  crosses through `H` and loads return through `H^T`.
- **Rigid-body motion transfers exactly**, so a structure merely being carried
  along acquires no spurious strain.
- **The coupling residual is a fixed function of the displacement**: the wake is
  shed once per step, the transfer is built once on the reference
  configurations, and the fluid solve is idempotent at fixed geometry. That is
  what quasi-Newton acceleration needs.

All four are measured at round-off in [docs/COUPLING.md](docs/COUPLING.md).

## Results

| case | result |
| --- | --- |
| transfer conservation | force, moment and virtual work to 1e-16 |
| rigid limit | reproduces the fluid code's own history to 0.8% at E = 2e12 |
| static divergence | three routes agree; the two coupled ones to five figures, and both 1.0% from the classical formula |
| added mass in water | wet frequency from a marched oscillation within 2% of the dmu/dt estimate |
| added-mass instability | staggered coupling diverges at a mass ratio of 0.43; strong coupling holds at 7.1 |
| energy balance | work in equals stored energy to 4.5e-3 |
| structural gates | patch test, rigid-body modes and mass properties exact; cantilever and frequencies within 1-2% of the beam solution |

Full tables, with what each number means and where it does not agree, are in
[docs/FEM.md](docs/FEM.md) and [docs/COUPLING.md](docs/COUPLING.md).

## Two things to know before running a coupled case

**Resolve the structural period, not only the wake.** A coupled march has two
resolution constraints. Below about ten steps per period of the softest
structural mode, unresolved modes ring undamped, and although their
displacement is negligible their acceleration is not - it reaches the pressure
through dmu/dt. The symptom is a two-step ripple in the lift over a perfectly
smooth quasi-steady lift. Resolve the mode, start from the static equilibrium,
or set `rho_inf` below one. The driver warns.

**In water, use strong coupling.** Below a mass ratio of order one a staggered
scheme has no stable step size at all. `coupling="loose"` exists so that this
can be demonstrated; it is not a recommendation.

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
python3 verify_fsi.py --case C3           # one case by name
python3 verify_analytic.py                # vendored: cylinder and Joukowski
python3 verify_unsteady.py                # vendored: Wagner, Theodorsen (~1 h)
python3 fem_mesh.py                       # report and check the mesh builders
```

pytest is not installed and must not be introduced: the tests use the standard
library `unittest`, following the upstream convention.

## Documents

- [docs/FEM.md](docs/FEM.md) - the structural formulation, every convention, and
  the structural verification tables.
- [docs/COUPLING.md](docs/COUPLING.md) - the interface operators, the
  partitioned schemes, the two resolution constraints, and the coupled
  verification tables.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - block boundaries, state
  contracts, public signatures, extension hooks, and what the coupling
  extension cost against what the fluid architecture predicted.
- [docs/FLUID.md](docs/FLUID.md) - provenance of the vendored fluid solver.

The fluid formulation is documented in the upstream repository, in its
`DESIGN.md`, `UNSTEADY.md` and `ARCHITECTURE.md`. Read them there.

## Licence

MIT. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

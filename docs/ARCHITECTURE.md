# Architecture: the structural and coupling modules in blocks

This document owns the block boundaries, the state contracts, the public
signatures and the extension hooks of the four modules written here. The
structural formulation is in [FEM.md](FEM.md) and the coupling formulation in
[COUPLING.md](COUPLING.md); the fluid side is documented in the upstream
repository ([FLUID.md](FLUID.md) records which commit) and is not restated.

It is the counterpart of `docs/ARCHITECTURE.md` upstream, which specified this
coupling as its extension (2) and left three hooks unused. The final section
records what that extension actually cost against what it was designed to cost,
in the same form.

Throughout: N = nwrap·nspan panels and M = (nwrap+1)·(nspan+1) mesh nodes on
the fluid side; Ns structural nodes, Ne elements, ndof = 3·Ns and G = 8 Gauss
points on the structural side. Floats are float64 and index arrays int64.

Last updated 11 August 2026.

## Layout

Flat modules, no package, following the upstream convention and for the same
reason: the vendored fluid suites must run unchanged, and importability for
coupling requires a side-effect-free import rather than a package.

```
fem_materials.py     the constitutive seam
fem_solid.py         the structural physics, ten sections
fem_mesh.py          structural mesh builders
fsi_transfer.py      the two interface operators
fsi_driver.py        the static and time-marched coupling, and the CLI
```

`fem_*` is the structure and `fsi_*` the coupling, so the eight vendored fluid
files at the repository root stay visually distinct from what is maintained
here. Nothing imports downward:

```
fem_materials  <-  fem_solid  <-  fem_mesh (builders only)
                        ^
thick_panel_wing  <-  fsi_transfer  <-  fsi_driver  ->  unsteady_wing
```

The three discipline rules of the upstream module carry over unchanged, and
each is load-bearing here rather than stylistic.

- **Every block is a pure function over explicit arguments**, with all mutable
  data in an explicit state dictionary, so two states coexist. The coupling
  needs this: `case_instability` in `verify_fsi.py` builds a fresh fluid and
  structural state for each of nine runs inside one process.
- **The library path is free of prompts, prints and figures.** `main()` is the
  only place they appear, in each module that has one.
- **Nothing imports downward.** `fem_solid` knows nothing about the fluid;
  `fsi_transfer` is the only module that imports both sides.

## The material seam

This is the extension the repository was asked for, so it is the first thing
specified rather than the last.

```python
class Material:
    name: str
    rho: float               # mass density
    n_history: int           # internal variables per Gauss point
    constant_tangent: bool   # may the assembly evaluate the tangent once?

    def tangent(self, orientation=None) -> (6, 6)
    def response(self, strain, strain_rate=None, history=None, dt=None,
                 orientation=None) -> {"stress", "tangent", "rate_tangent",
                                       "history"}
```

`response` is vectorised over the G Gauss points of the elements sharing a
material: strain `(G, 6)` in, stress `(G, 6)` and tangent `(G, 6, 6)` out.
Voigt order is `[11, 22, 33, 12, 23, 31]` with engineering shear.

Three materials exist, and the second and third are not decoration: they are
what turns "extendable" into a property that is exercised and gated.

| material | what it proves | gate |
| --- | --- | --- |
| `IsotropicElastic` | the verified default | every structural case of `verify_fem.py` |
| `OrthotropicElastic` | the **anisotropy** path: `orientation`, the 21-constant tangent, the rotation `D -> T^T D T` | F6: reduces to isotropy at 2.4e-16, rotates as a tensor at 2.5e-16 |
| `KelvinVoigt` | the **rate and history** path: `strain_rate`, `dt`, `history`, `rate_tangent` | F7: analytical decay to 6e-5 |

Without the second, `orientation` would be an unused argument; without the
third, `strain_rate`, `dt`, `history` and `rate_tangent` would all be dead
code, and dead code in an interface is a promise, not a seam.

**What a plastic material adds.** A `ReturnMappingPlastic(base, yield_stress,
hardening)` implements `response` as a trial elastic stress, a yield check, a
return mapping, and the consistent algorithmic tangent; it writes the plastic
strain and the hardening variable into `history` and sets `n_history` to their
count. Nothing else in the repository changes: `internal_force` already threads
`history` and `dt` through, `solve_static` and `step_dynamic` are already
Newton loops on a residual, and the incompatible-mode amplitudes are already
resolved by an element-level solve which becomes an element-level Newton in
`element_alpha` and nowhere else. The one thing that does change is cost: a
material with a state invalidates `constant_operators`, so the tangent is
reassembled per iteration instead of cached.

## Structural state contract

Geometry splits the same way it does upstream, into topology built once and
metric quantities cached from the reference configuration.

```python
model = {                            # built once by build_model
  "nodes": (Ns, 3), "elements": (Ne, 8) int64, "etype": "hex8",
  "materials": [Material], "mat_id": (Ne,) int64,
  "orientation": (Ne, 3, 3) or None,      # material axes, the anisotropy hook
  "incompatible": bool,
  "faces": (Nf, 4) int64, "face_elem": (Nf,) int64,   # outer surface
  "cache": {                         # the reference configuration, once
     "B": (Ne, G, 6, 24), "Ba": (Ne, G, 6, 9) or None,
     "wdet": (Ne, G), "N": (G, 8), "dNdx": (Ne, G, 8, 3),
     "volume": (Ne,), "gauss": (points, weights),
     "d_all": (Ne, 6, 6) or None,    # lazily filled by constant_operators
     "k_aa_inv": (Ne, 9, 9) or None,
  },
}

sstate = {
  "model": model,
  "u": (Ns, 3), "v": (Ns, 3), "a": (Ns, 3), "f": (Ns, 3),   # the committed level
  "history": (Ne, G, n_history),
  "fixed": (ndof,) bool, "spring": (ndof,) float,
  "rayleigh": (a0, a1), "lumped_mass": bool,
  "K": (ndof, ndof), "M": ..., "C": ..., "C_mat": ...,
  "factor": ..., "factor_key": tuple,
  "operators_version": int, "constraint_version": int,
  "residuals": {"newton": int, "resid": float},
}
```

Two entries need justifying.

**`C` and `C_mat` are separate.** `C` is Rayleigh damping, a modelling device
with no constitutive content, and it enters the equilibrium equation directly.
`C_mat` is the assembled material rate tangent, and it is used **only** to
build the effective dynamic operator, where it is a tangent: the viscous stress
itself reaches equilibrium through `internal_force(v=...)` like any other
stress. Adding `C_mat` to `C` as well would count the viscous stress twice.

**`factor_key` includes both version counters and the step.** The Cholesky
factor of the effective operator is the expensive object; it is reused across
every step and every coupling subiteration and invalidated by a change of dt,
of the integrator parameters, of the operators or of the constraints.

## Public signatures

### fem_solid, sections 3 to 6

```python
gauss_points() -> (pts (8,3), weights (8,))
shape_functions(xi) -> (P, 8)                # trilinear, nodal, partition of unity
shape_derivatives(xi) -> (P, 8, 3)
bubble_derivatives(xi) -> (P, 3, 3)          # the three Wilson bubbles

element_arrays(nodes, elements, incompatible=True) -> cache
    # B, Ba, w*detJ, N, volumes. The incompatible gradient uses the Jacobian at
    # the element CENTRE scaled by detJ0/detJ - the Taylor, Beresford and Wilson
    # correction, without which the patch test fails on a distorted mesh.
    # A non-positive Jacobian at any Gauss point raises, naming the element.
element_stiffness(model, d_all) -> (Ke (Ne,24,24), k_aa_inv)
    # incompatible amplitudes condensed exactly at element level
element_mass(model, lumped=False) -> (Ne, 24, 24)
element_alpha(model, u_e, d_all, k_aa_inv) -> (Ne, 9)
    # the element-level equilibrium of the bubbles; ONE local Newton here is
    # the whole of what a nonlinear material adds to the element
constant_operators(model) -> (d_all, k_aa_inv)      # lazily cached

build_model(nodes, elements, materials, mat_id=None, orientation=None,
            incompatible=True) -> model
surface_faces(elements) -> (faces (Nf,4), owners (Nf,))
    # faces keyed by their node SET, so the collapsed wedges at a foil leading
    # and trailing edge pair correctly instead of appearing on the boundary
mass_properties(model) -> {"mass", "centre", "inertia", "volume"}

assemble_stiffness(model) / assemble_mass(model, lumped) / assemble_rate_damping(model)
internal_force(model, u, v=None, history=None, dt=None)
    # -> (f_int (Ns,3), history, stress (Ne,G,6)). The integral of B^T sigma,
    # ALWAYS, with the material in the loop. Never K u, even where they agree.
strain_energy(model, u) / kinetic_energy(sstate, v=None)
```

### fem_solid, sections 7 to 9

```python
init_state(model, rayleigh=(0,0), lumped_mass=False) -> sstate
clamp(sstate, node_ids, components=(0,1,2)) -> sstate
add_spring(sstate, node_ids, components, stiffness) -> sstate
    # grounded linear springs: enough to build an elastically supported wing -
    # a typical section is grounded springs on a stiff body - without a second
    # element type, and the divergence case is built on it
assemble_operators(sstate) -> sstate

cholesky_factor(a, what) / forward_substitute / backward_substitute / cholesky_solve
generalised_modes(K, M, nmodes=None) -> (omega, phi)

solve_static(sstate, f_ext, tol, max_iter, u0=None) -> level
modes(sstate, nmodes=6) -> (freq_Hz, shapes (Ns,3,m))
integrator_parameters(rho_inf=1.0, alpha_m=None, alpha_f=None) -> dict
step_dynamic(sstate, f_new, dt, par=None, f_old=None, tol, max_iter) -> level
initial_acceleration(sstate, f_ext) -> (Ns, 3)
commit(sstate, level, f_ext=None) -> sstate
```

`step_dynamic` returns a level `{u, v, a, history}` and **mutates nothing**.
That is the single property the strong coupling rests on: the same time level
is re-solved several times with different loads, and an integrator that has
already advanced its own state cannot be asked twice. `commit` is the only
thing that advances the state of record.

Both Newton loops measure their residual against the **first** residual of the
call, not against the applied force, and stop on stagnation. A free-vibration
or self-equilibrated step has no external force, and a scale taken from it
collapses to zero, which runs every step silently to the iteration cap; the
attainable floor is round-off in the integral of B^T sigma on a stiff
structure, near 1e-9 relative, and a tighter target only spins.

### fsi_transfer

```python
project_to_quads(points, corners, clamp=True) -> (weights (P,4), projected, distance)
build_transfer(fluid_state, model, candidates=8) -> transfer
    # ONCE, on the two REFERENCE configurations. rows (M,4), weights (M,4),
    # the fluid mesh shape, the weld map, and the projection offset.
to_fluid(transfer, field) -> (M, 3)            # H
to_structure(transfer, field) -> (Ns, 3)       # H^T
displaced_points(transfer, reference_points, u_struct) -> (nwrap+1, nspan+1, 3)
point_velocities(transfer, v_struct) -> (nwrap+1, nspan+1, 3)
panel_forces_to_nodes(pan, force_panels) -> (M, 3)
structural_forces(transfer, pan, force_panels) -> (Ns, 3)
conservation_report(transfer, pan, force_panels, u_struct=None) -> dict
```

### fsi_driver

```python
FixedPointAccelerator(method='aitken'|'iqn'|'constant', omega, reuse)
init_fsi(points, onset, model, u_ref, rho, unsteady=True, core=None,
         config=None, rayleigh=(0,0), lumped_mass=False) -> (fluid, sstate, transfer)
static_aeroelastic(fluid, sstate, transfer, rho, ...) -> (fluid, sstate, history)
time_march_fsi(fluid, sstate, transfer, dt, nsteps, rho, coupling='strong',
               accel='aitken', ...) -> (history, fluid, sstate)
steps_per_period(sstate, dt, nmodes=1) -> float
added_mass_ratio(fluid, sstate, transfer, rho, dt, amplitude=None) -> dict
aerodynamic_stiffness(fluid, sstate, transfer, rho, nmodes=4, ...) -> dict
divergence_pressure(fluid, sstate, transfer, rho, nmodes=4, ...) -> dict
growth_rate(t, signal, skip=0.25) -> {"rate", "frequency", "peaks"}
plot_fsi_history(history, save=None) -> Figure
```

## The two resolution measures

A coupled march has to resolve two things, and the driver reports and warns on
both.

| measure | source | threshold | symptom of violating it |
| --- | --- | --- | --- |
| chords convected per step | `unsteady_wing.convection_per_step` | 0.5 | the near wake is under-resolved; the indicial response is wrong while the steady limit is not |
| steps per structural period | `fsi_driver.steps_per_period` | 10 | unresolved modes ring undamped at rho_inf = 1, with tiny displacement and large acceleration, and the acceleration reaches the pressure through dmu/dt |

The second was found rather than anticipated, and the way it presents is worth
recording because it does not look like a resolution problem: the lift carries
a two-step ripple while the quasi-steady lift is perfectly smooth. Three cures,
in order of preference: resolve the mode, start from the static aeroelastic
equilibrium instead of from rest, or set `rho_inf` below one.

## Extension hooks: property now, behaviour later

| # | present now | added later | status |
| --- | --- | --- | --- |
| 1 | stress and tangent at the Gauss point through a `Material` | plasticity, damage, creep | seam exercised by `KelvinVoigt` |
| 2 | per-element `orientation` and a full 21-constant tangent | composite lay-ups, fibre directions | exercised by `OrthotropicElastic` |
| 3 | `history` (Ne, G, n_hist) threaded through `internal_force` and committed only with the level | internal variables of any inelastic law | exercised, n_hist = 6 |
| 4 | internal force ALWAYS the integral of B^T sigma | total Lagrangian: change the strain measure and add the geometric stiffness | future |
| 5 | Newton loops on a residual in both solvers, converging in one iteration when linear | nonlinear material or geometry: more iterations, no new structure | future |
| 6 | `element_alpha` is a separate element-level solve | the same solve becomes a local Newton for a nonlinear material | future |
| 7 | element kinematics separated from constitution | shells and beams as new elements sharing the same materials | future |
| 8 | `mat_id` per element | spatially varying and layered materials | present, untested at scale |
| 9 | `step_dynamic` pure; `commit` separate | any implicit coupling scheme | done, and required |
| 10 | transfer built once on the reference configurations | quasi-Newton coupling | done, IQN-ILS |
| 11 | `H` and `H^T` for the two directions | any conservative interface | done |
| 12 | dense operators, factored once against a version stamp | a sparse or iterative solver behind the same four functions | future |

## What the coupling extension actually cost

The upstream architecture predicted this coupling and wrote its driver loop out
on paper, claiming it "closed with no corrections". Against the real structural
side, that claim holds for the **static** loop, which is exactly the loop it
wrote, and needed four amendments for the time-marched one.

1. **`shed` must be outside the subiteration.** The paper loop did not have
   subiterations. `shed` prepends a row and convects, mutating the wake
   irreversibly, so a strongly coupled step that shed once per subiteration
   would grow the wake by the subiteration count. Everything inside the loop is
   `update_points` plus `solve(wake="frozen")`, and `update_points` re-pins wake
   row zero onto the moved trailing edge, which is what lets the newly shed
   strip follow the deforming surface without being shed again.
2. **The doublet history must not advance until the step converges.** dmu/dt,
   and with it the whole added mass, would otherwise depend on the path the
   iteration took rather than on the time level. The history is seeded with the
   acyclic level and committed with the structural level.
3. **A deforming surface is never `rigid`.** The upstream rigid fast path,
   which removes the 0.63 s reassembly at 960 panels, is unavailable to a
   coupled run by construction: every subiteration pays a full reassembly. That
   is the dominant cost of the coupling and it is the accepted contract, not an
   oversight.
4. **The structural spectrum is a second resolution constraint**, of the same
   standing as the wake one, and it has no counterpart upstream because a
   prescribed rigid motion has no spectrum.

One prediction was vindicated more strongly than expected. Because the fluid
consumes the onset callable in exactly two places and every surface block takes
sampled arrays, the deforming surface entered through `update_points(points,
point_velocities)` and nothing downstream knew - the same property that
admitted rigid body motion. And `centroid_weights`, added upstream for a
different purpose entirely (sampling body velocity at collocation points),
turned out to be exactly the conservative load-lumping operator this side
needed, adjoint included. Neither was designed for this.

## Tests

`test_fem.py` (35), `test_fsi.py` (20) and `test_vendored.py` (3), on the
standard-library unittest; pytest is not installed upstream and is not
introduced. The convergence and coupled-physics studies are behind `FEM_SLOW=1`
and `FSI_SLOW=1` rather than markers, following the upstream convention.

`test_vendored.py` checks the SHA-256 of every vendored fluid file, so a local
edit to the fluid solver fails the suite. The two vendored fluid suites run
unchanged and are part of this repository's test run.

## Build order

Each step gated on its own checks before the next started: materials and their
two gates; element, assembly, patch test and rigid-body modes; constraints,
factorisation, statics and eigen; the time integrator; the mesh builders; the
transfer, gated to round-off before any coupled run existed; the static
coupling; the time march; the verification programmes and these documents.

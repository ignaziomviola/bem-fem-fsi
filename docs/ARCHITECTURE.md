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
vent_section.py      sectional cavity physics; imports nothing from here
vent_mesh.py         the doubled surface-piercing strut, and the mirror maps
vent_cavity.py       cavity extent, inception, the regime state machine
vent_loads.py        the cavity pressure and the immersed-half load path
vent_solve.py        the image, the mixed-unknown system, the cavity fixed point
```

`fem_*` is the structure, `fsi_*` the coupling and `vent_*` the ventilation
closure model, so the eight vendored fluid files at the repository root stay
visually distinct from what is maintained here. Nothing imports downward:

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

## The ventilation state contract

`docs/VENTILATION.md` owns the physics; what belongs here is which entries change
and when.

```python
vent = {                            # built once by build_vent
  # parameters, never changed
  "y_fs", "h", "chord", "ar", "u_ref", "fn_h", "dsigma", "g", "gamma_st", "rho",
  "alpha_rad", "alpha_stall", "recovery", "growth_chords", "closure",
  "extent_rule", "length_model", "sub_tol", "sub_max", "sub_omega",
  "x_ref": (3,),                    # FIXED once; see below
  # geometry, built once
  "image": bool, "n_half", "waterline_j", "shape_full", "shape_half",
  "node_mirror": (M,), "panel_mirror": (N,), "node_real": (Mh,),
  "real_mask": (N,), "image_mask": (N,), "wake_col_mirror", "wake_strip_mirror",
  "points_half_ref": (nwrap+1, n_half+1, 3),
  # the TIME LEVEL: advances only in commit_vent
  "regime": 'FW'|'PV'|'FV', "l_c": (nspan,), "detach": (nspan,),
  "weight": (N,), "d_cav", "phi_bar", "step", "time", "transitions", "drift_y",
}
```

**Two entries need justifying.**

`x_ref` is fixed at construction rather than left to `integrate_loads`, which
locates its own moment reference by a max-distance search on the mid-span
section. On a doubled mesh that section is the waterline, and the search re-runs
on every deflected geometry, so the yawing moment would acquire a spurious drift.

Everything a coupling subiteration computes - the pressures, the cavity mask, the
thickness, the closure line, the residual - lives in a `cav` dictionary returned
inside `loads["cav"]` and is discarded. Nothing that changes within a
subiteration is stored in `vent`. That, plus cold-starting the cavity iteration
from the committed length, is what keeps the coupled load a deterministic
function of the displacement and the committed regime, which the quasi-Newton
acceleration requires. The gate is one cheap test: evaluate the load twice from
the same committed state at the same displacement and assert the two nodal force
arrays are bit-identical.

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

init_vent_fsi(points, onset, model, h, chord, ..., image=True, **vent_kw)
                                     -> (fluid, sstate, transfer, vent)
ventilation_margin(fluid, vent, rho) -> dict     # how close to inception
```

Every ventilated path is behind one optional `vent=None` argument on
`static_aeroelastic`, `time_march_fsi`, `added_mass_ratio`,
`aerodynamic_stiffness` and `divergence_pressure`; with it None, the code is the
pre-existing code verbatim, which `test_vent.TestBaseline` asserts bit-for-bit.
Three private helpers carry the whole intrusion: `_reference_points` returns the
array the transfer was built on, `_geometry` lifts a displaced half to the
doubled mesh, and `_solve_fluid` dispatches between the vendored solve and the
image-antisymmetric one.

### vent_section

```python
sigma_cavity(depth, h, fn_h, dsigma=0.0) -> array      # (1.10), (1.12)
froude_h(u, h, g) -> float;  weber(u, chord, rho, gamma) -> float
psi(sigma_c, alpha_2d) -> array
acosta_psi(L) / acosta_length(psi) / acosta_lift_slope(L)      # (1.6), exact
tulin_length(psi, alpha_2d) / tulin_psi(L)                     # (1.5), exact
harwood_length(psi) / harwood_length_low_order(psi)            # (1.7), (1.8)
cavity_length(psi, model='exact'|'fit'|'low', alpha_2d=0.0) -> array
lift_slope(L) -> array                                         # (1.9)
centre_of_pressure(L) -> array                                 # (3.4)
jet_speed(u, sigma_c) / jet_components(u, sigma_c, phi)         # (3.1), (3.2)
closure_angle(depth, x_closure) -> (phi_bar, phi_local)
unstable_closure(phi_bar) -> bool;  regime(D, phi_bar, h) -> str
helmbold(a0, ar) / elliptic_shape(kappa) / lift_weighted_slope(ar)
washout_froude(cl, ar_h) / washout_froude_chain(cl, ar_h)      # (4.5), (4.1-4.4)
breslin_skalak(cl) / washout_margin(fn_h, cl, ar_h)            # (1.2)
```

### vent_mesh

```python
strut_mesh(n_c, nspan_half, h, chord, ..., image=True) -> points
symmetrise_points(points, y_fs) / symmetry_residual(points, y_fs)
mirror_maps(shape_full, y_fs=0.0, image=True) -> maps
half_points(points_full, maps) / complete_points(points_half, maps)
complete_vectors(v_half, maps) / mirror_scalar(field, maps, sign=-1.0)
half_interface(points_half) -> dict          # for build_transfer(iface=...)
symmetrise_wake(wake, maps) -> {"drift_y", "drift_max", "mu_residual"}
strut_solid_mesh(points_half, ...) -> mesh
```

### vent_cavity

```python
build_vent(maps, points_half, h, chord, u_ref, alpha_rad, ...) -> vent
commit_vent(vent, cav, t=None, dt=None) -> vent      # the ONLY mutation
transition(vent, cav) -> (regime, why)               # called only from commit
set_regime(vent, regime) / freeze(vent) / inject(vent, station=None)
panel_depth(pan, y_fs) / suction_side(pan, alpha_rad) / chordwise_frame(pan)
wrap_extent(pan, frame) / wrap_spacing(pan, a, b) / cavity_interior(pan, weight)
mask_from_lengths(pan, frame, lengths, alpha_rad, real_mask, detach)
pressure_target(...) -> (detach, length);  section_target(...) -> length
thickness(...) / closure_thickness(...) / closure_residual(...) / entrainment(...)
separated(pan, cp, alpha_rad, recovery) / air_path(pan, candidate, seed)
waterline_band(pan, vent) / closure_geometry(pan, frame, lengths, vent)
cavity_report(pan, frame, vent, cav) -> dict
```

### vent_loads and vent_solve

```python
cavity_cp(pan, vent) / cavity_speed(pan, vent)
vent_pressure(pan, mu, u_rel, u_ref, vent, weight, dphi_dt, rho) -> dict
immersed_loads(fluid, vent, p_gauge, rho) -> loads      # on s_ref = h*c
structural_forces_half(transfer, vent, pan, force_panels) -> (Ns, 3)
conservation_report_half(...) / depth_loading(pan, vent, p_gauge, rho)

image_sigma(pan, vent, u_rel, sign=-1.0) -> (N,)
mixed_solve(fluid, vent, sys, sigma, unknown_type, mu_fixed) -> (mu, sigma)
solve_wetted(fluid, vent, wake='frozen', sign=-1.0) -> fluid    # replaces tpw.solve
dynamic_mu(pan, frame, vent, weight, mu_wet, u_rel, v_span) -> (mu, v_s)
solve_cavity(fluid, vent, rho, dphi_of_mu, wake, sign, dt) -> (fluid, cav, loads)
relax_wake(fluid, vent, ...) / shed_and_project(fluid, vent, dt, nmax)
antisymmetry_residual(fluid, vent, probe=None) -> dict
washout_margin(vent, cl) / report(fluid, vent, cav, rho) -> dict
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
| 13 | corner-pure, mirror-covariant kernels | free-surface image | done, by mesh doubling and a source-strength sign flip |
| 14 | per-panel doublet/source unknown swap in `assemble_system` | cavity with prescribed pressure and solved thickness | done, `closure='dirichlet'` |
| 15 | `ds_wrap` per panel | cavity-height integrals | done, thickness and entrainment |
| 16 | the trailing-edge fold sends a prescribed `mu` to the right-hand side | a cavity that reaches the trailing edge | done, and needed no change |
| 17 | `build_transfer(iface=...)` | a structure facing only part of the fluid mesh | done, the immersed half |
| 18 | `foil_ribs(pinched=...)` | a wrap mesh with one open span end | done, the waterline root |
| 19 | `dsigma` in the cavity pressure | vaporous cavitation, and the vaporous-to-ventilated transition | property present, behaviour future |
| 20 | `growth_chords` rate limit on the cavity front | a transport equation for the cavity interface | property present, behaviour future |

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

`test_fem.py` (44), `test_fsi.py` (21), `test_vent.py` (87) and
`test_vendored.py` (4), on the standard-library unittest; pytest is not installed
upstream and is not introduced. The convergence, coupled-physics and coupled
ventilation studies are behind `FEM_SLOW=1`, `FSI_SLOW=1` and `VENT_SLOW=1`
rather than markers, following the upstream convention.

`test_vendored.py` checks the SHA-256 of every vendored fluid file, so a local
edit to the fluid solver fails the suite. The two vendored fluid suites run
unchanged and are part of this repository's test run.

## Build order

Each step gated on its own checks before the next started: materials and their
two gates; element, assembly, patch test and rigid-body modes; constraints,
factorisation, statics and eigen; the time integrator; the mesh builders; the
transfer, gated to round-off before any coupled run existed; the static
coupling; the time march; the verification programmes and these documents.

## What the ventilation extension actually cost

The upstream architecture predicted this extension too, and named four hooks for
it. All four held. Two of its predictions needed amending, and one thing it did
not anticipate turned out to be the load-bearing decision.

1. **A mirror image cannot be convected, only projected.** The upstream sketch
   said a free-surface image is "strictly `kernel(pts, mirror_corners(corners,
   z_fs))` with signed factors, plus mirrored segment endpoints", and warned that
   a driver composing image blocks must own the wake relaxation. Both are true,
   but the reason is stronger than the warning: with `phi` antisymmetric the
   perturbation velocity mirrors WITH a sign while the onset does not, so a
   mirror-consistent convection would require the perturbation velocity to
   vanish. The image wake has to be placed rather than advected, and the drift the
   projection discards is the linearised wave elevation - a diagnostic, not an
   error. Nothing upstream could have predicted this, because a rigid wing has no
   image.
2. **The image is cheaper as geometry than as blocks.** Because
   `validate_mesh` and `update_points` both force pinched span ends, an open
   waterline section is impossible; and because `thick_wing_mesh` with unit taper
   already IS the doubled strut, the image needs no new kernel calls at all - only
   a sign flip on the source strengths of the mirror half, through
   `assemble_system`'s `sigma_fixed`. The consequence, which the sketch did not
   note, is that on a doubled mesh the vendored `tpw.solve` returns the RIGID-WALL
   answer bit-for-bit: the free surface is exactly one sign away from the answer a
   caller gets by accident.
3. **The unknown-swap masks were right, and the trailing-edge fold was already
   correct.** `assemble_system` handles a cavity panel that is also a Kutta panel
   without change, sending its prescribed strength to the right-hand side. That
   case - a cavity reaching the trailing edge - is the fully ventilated regime, so
   the hook mattered in the regime that matters most.
4. **`ds_wrap` was documented as the cavity-height integration hook and is
   exactly that**, but the spacing the cavity's prescribed doublet strength must
   be integrated with is the arc-floored one the wrap stencil itself divides by,
   not the panel extent alone and emphatically not the chordwise distance:
   integrating along the chord puts the leading-edge panels badly wrong, because
   there the surface tangent is nearly normal to the chord.

The decision nothing upstream predicted, and the one everything else rests on, is
that **the cavity must be parameterised by a continuous length with a fractional
closure panel rather than by a boolean panel mask.** A boolean mask makes the load
piecewise-constant in the displacement; the cavity then chatters between two
adjacent panels, and IQN-ILS - which upstream provided for exactly this coupling -
fits its least-squares system to a staircase and cannot converge below one panel
width at any tolerance. The same reasoning applied twice more: to the growth-rate
limit, which stops a transition being an instantaneous load step whose structural
response measures the time step rather than the flow, and to the regime, which
advances only on commit because bi-stable states are not functions of the
instantaneous condition at all.

The coupling extension's own cost accounting said the transfer being built once
was what made quasi-Newton work. That is still true, and the doubled mesh nearly
broke it: the image half has no structural counterpart, so the transfer had to be
built on the immersed half through a new `iface` argument, with the image geometry
slaved by a mirror map. Because that map is fixed and linear, the build-once
property survived untouched.

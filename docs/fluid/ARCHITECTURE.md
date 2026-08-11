# Architecture: thick_panel_wing.py in blocks

**Status.** Extension (1), time dependence, is built: it lives in `unsteady_wing.py` and is specified in [UNSTEADY.md](UNSTEADY.md). This document is unchanged apart from the amendments recorded in "What the time extension actually cost" at the end, which report what the paper design got right and what it did not. Extensions (2) and (3) remain future work and their hooks are untouched.

This document specifies the code structure of `thick_panel_wing.py` so that three planned extensions fit later without structural rewrite: (1) time dependence, (2) partitioned coupling with a finite element structural code, and (3) ventilation and cavitation models. The physics is fixed by DESIGN.md and is not restated; this document owns the block boundaries, the state contract and the public signatures. Each extension point below is traceable to a concrete requirement of one of the three extensions, none adds a speculative code path to version one, and the architecture was verified by writing all three future drivers on paper against the public signatures alone — the coupling driver closed with no corrections, the unsteady and cavity drivers with local amendments that are folded in below.

Throughout, N = nwrap·nspan panels, M = (nwrap+1)·(nspan+1) mesh nodes, panel flat index h = i·nspan + j, and the abbreviations of the design document apply: trailing edge (TE), leading edge (LE). Floats are float64, index arrays int64, the unknown-type mask int8.

Last updated 9 August 2026.

## Layout: one module, disciplined sections

The implementation is a single file `thick_panel_wing.py` beside `panel_wing.py`, plus `make_thick_sample_inputs.py` and `test_thick_panel_wing.py`; no package. Every existing code in the repository is a standalone file and the fresh-clone reproduction check assumes flat modules; importability for coupling requires a side-effect-free import, not a package — no prompts, prints, file input/output or figures at module level, matplotlib imported lazily inside the plotting functions, and all mutable data in an explicit state dictionary. The estimated size is 1500 lines, and if a later extension outgrows one file the section boundaries below are the package cut lines, so the layout decision is reversible without interface change.

Sections in dependency order (nothing imports downward):

```
# 1 constants     RHO=1.0 (default only), NWAKE, WAKE_LENGTH_SPANS, FAR_FIELD,
#                 MAX_ITER, OMEGA, TOL, CORE_WING, RC_WAKE_FRACTION, WARP_WARN=0.1
# 2 imports       from freewake_kernels import segment_velocities,
#                 induced_velocity, load_velocity_profile, make_onset,
#                 pitch_mesh, initial_nodes, prompt
# 3 kernels       pure, corner-based, stateless
# 4 geometry      load/validate/build; topology-metric split
# 5 wake          dict + views + node updates
# 6 assembly      blocks, wake influence, folds, masked system
# 7 solver API    init_state, update_points, interface, solve_once, solve
# 8 loads         gradient, velocity, pressure, integration, Trefftz, leakage
# 9 plotting      lazy matplotlib, returns Figure, never show()
# 10 CLI          prompt()/print()/plt.show() live ONLY here, under main()
```

The shared machinery lives in `freewake_kernels.py`, a verbatim copy of seven functions from `panel_wing.py` in the companion repository https://github.com/ignaziomviola/free-wake-lifting-surface, duplicated so that a fresh clone of this repository runs on its own and collected in one module so that the duplication stays visible; the doublet-ring unit check is what would detect a divergence. Two further deviations from the design document's reuse list, recorded here as its amendments: `filament_segments` is not copied — the array wake below supersedes the per-filament dictionaries, and `wake_segments` mirrors its closure invariant instead (one leg per filament along the normalised local onset at the last node, length FAR_FIELD, no far-end spanwise return segment in the velocity view, while the potential view does carry the full closure quad); and `relax_filaments` is re-implemented as `relax_wake_step` because the advection velocity gains the source-panel term. Since `freewake_kernels.py` imports numpy alone, importing the solver pulls in no figure backend; only the command-line path, which ends in `plt.show()`, needs MPLBACKEND=Agg for a batch run. The companion repository carries one matching line, the `load_mesh` guard in `panel_wing.py` rejecting `thick-wrap*` archives.

The onset closure is consumed in exactly two places: the driver-level composition of `u_rel` in `init_state`/`update_points`, and velocity evaluation at arbitrary points (`wake_node_velocity`, the closure legs of `init_wake` and `wake_segments`, `trefftz_drag`). No assembly, solve or loads block ever receives the callable; every surface-side block takes sampled arrays. This single rule is what admits body motion later.

## State contract

### panels

Geometry splits into topology, built once and never invalidated by point motion, and metric, rebuilt whenever points move:

```python
panels = {
  # topology: pure function of (nwrap, nspan)
  "nwrap": int, "nspan": int,
  "panel_nodes": (N, 4) int64,    # node indices of Q1..Q4, anticlockwise about n
  "weld_map":    (M,)   int64,    # canonical node of each coincident group
                                  #   (TE seam rows; mirrored tip-pinch pairs)
  "k_up": (nspan,) int64,         # (nwrap-1)*nspan + j  } sole owners
  "k_low": (nspan,) int64,        # j                    } of TE topology
  "wrap_prev/wrap_next/span_prev/span_next": (N,) int64,  # -1 sentinels at the
                                  #   TE seam and tips; wrap continuous through the LE
  "tip_flag": (N,) bool,          # tip-strip panels (noisy-pressure flag)
  # metric: rebuilt on every geometry update
  "points": (nwrap+1, nspan+1, 3), "corners": (N, 4, 3),
  "centroids": (N, 3), "normals": (N, 3), "areas": (N,),
  "frames": (N, 3, 3),            # rows t_c, t_s, n (right-handed)
  "dual": (N, 2, 3),              # dual basis for the surface gradient
  "ds_wrap": (N,),                # wrap-direction arc extent (cavity integrals later)
  "warp": (N,), "d_box": float,
}
```

Panel numbering is fixed by the wrap format alone, so index h denotes the same material panel at every geometry update and, later, every time level — the property dμ/dt rests on.

### wake

An array form supersedes the per-filament dictionaries of `panel_wing.py`; the view functions re-emit the plain (p1, p2, strengths, cores) arrays that `segment_velocities` and `induced_velocity` consume, and the consistency review confirmed the array views reproduce the design's two-level folding numerically exactly.

```python
wake = {
  "nodes":   (nrows+1, nspan+1, 3),  # row 0 attached to the TE nodes
  "mu":      (nrows,  nspan),        # doublet per wake quad (row, strip)
  "n_bound": int,                    # rows slaved to the current TE jump
  "steps":   (nrows,),               # per-row arc lengths; caller data, never
                                     #   derived from NWAKE inside a block
  "closure": bool,                   # FAR_FIELD closing quad/legs on or off
  "core":    float,                  # rc_wake
}
```

Steady version one: nrows = NWAKE, n_bound = nrows (every row bound, all mu rows equal by construction), closure True, nodes updated by relaxation. Future unsteady: n_bound = 1, `shed` prepends one nodes/mu/steps row per step (all three arrays, keeping nrows = len(steps) = len(mu)), tail rows dropped by axis-0 slicing, closure False with the last downstream spanwise segment emitted as the starting vortex — normative, not optional. Wake quad (r, j) has corners Q1 = nodes[r, j], Q2 = nodes[r+1, j], Q3 = nodes[r+1, j+1], Q4 = nodes[r, j+1], and the kernel's solid angle is averaged over both diagonal splits — deterministic and iteration-fixed, as the design requires, and exactly mirror-symmetric (a single fixed split is not mirror-covariant and broke the spanwise loading symmetry at O(10⁻¹) on warped panels during implementation).

### blocks and state

```python
blocks = {"D": (N, N), "S": (N, N)}   # doublet block with -1/2 already on the
                                      # diagonal (added by assembly, not the kernel);
                                      # source block. Immutable per geometry; the wake
                                      # is folded into a fresh copy at every solve.

state = {
  "u_ref": float, "onset": callable, "lifting": bool,      # config
  "panels": panels,
  "u_rel": (N, 3),                  # onset(centroids) - u_body; u_body = 0 in v1;
                                    #   the ONLY onset sample surface blocks see
  "chord_mean": float,              # mean straight LE-to-TE section chord — NOT the
                                    #   wrap arc length, which is about twice the chord
                                    #   and would silently double the TOL scale
  "span_extent": float, "s_ref": float, "x_near": float,
  "blocks": blocks or None, "blocks_version": int,
  "mu": (N,) or None, "sigma": (N,) or None, "wake": wake or None,
  "geom_version": int, "wake_version": int,
  "residuals": {"wake_disp_chords": float, "iterations": int},
}
```

`mu`, `sigma` and the wake persist across geometry updates as warm starts; two states coexist freely because no module-level data is mutable.

## Public signatures

### Kernels — pure, corner-based, stateless

```python
doublet_potential_matrix(pts, corners, far_diag=None) -> (M, K)
    # van Oosterom-Strackee solid angles, AVERAGED over both diagonal
    # splits (mirror-covariance; still deterministic and iteration-fixed,
    # and the closed-body row-sum identity holds exactly);
    # principal value (zero) in-plane: NO -1/2 diagonal here; frames derived
    # from the received corners; far_diag switches to the point doublet beyond
    # about five diagonals (per-iteration wake work only). Chunked over pts.
source_potential_matrix(pts, corners) -> (M, K)
    # Hess flat-panel potential, mean-plane projection; always exact (the body
    # block is assembled once; no consumer needs a far switch).
source_velocity(pts, corners, sigma, chunk=300) -> (M, 3)
    # Hess source-panel velocity; far-field point-source switch internally.
    # The in-plane terms are log-singular on panel perimeters and wake node
    # row 0 lies exactly on the TE panel edges: clamp the edge-singular
    # denominators and logarithm arguments as panel_wing.py clamps its kernel,
    # pinned by a probe test at a TE node.
```

Biot–Savart stays `panel_wing.segment_velocities`/`induced_velocity`, imported unchanged. No kernel reads `panels`, module constants or stored frames — a future free-surface image is strictly `kernel(pts, mirror_corners(corners, z_fs))` with signed factors, plus mirrored segment endpoints.

### Geometry

```python
load_mesh(path) -> points                  # design validation rules 1-9
build_topology(nwrap, nspan) -> dict       # index-only, once per state lifetime
build_metric(points, topo) -> dict         # pure; no orientation flip anywhere
build_panels(points) -> panels             # topology ∪ metric, re-callable at will
te_nodes(points) -> (nspan+1, 3)           # welded TE node row
reference_area(points) -> float            # projected planform sum(c_j * dy_j)
```

### Wake

```python
init_wake(te, onset, steps, n_bound, closure, core) -> wake
    # straight filaments along the local onset via panel_wing.initial_nodes,
    # column by column; steps is caller data (the steady driver passes
    # full(NWAKE, WAKE_LENGTH_SPANS*span_extent/NWAKE)); handles empty steps
    # (nodes-only wake) so the unsteady driver can start from zero rows.
wake_quads(wake) -> (nquads, 4, 3)         # + closure quad per strip iff closure;
                                           # derived on demand, never stored
wake_segments(wake, onset) -> (p1, p2, strengths, cores)
    # trailing segments at span node j, rows r..r+1: G = mu[r,j-1] - mu[r,j]
    # (out-of-range mu = 0); spanwise inter-row segments G = mu[r,j] - mu[r-1,j]
    # at node row r (this document gave the opposite sign until the unsteady
    # extension became the first consumer of the path; the code and its
    # docstring were right, and the ring-equivalence check in
    # test_unsteady_wing.py now pins it),
    # emitted only when nonzero (identically zero in steady); the bound row's
    # TE-side spanwise segment is ALWAYS dropped (analytic Morino cancellation);
    # closure=True appends the FAR_FIELD legs, closure=False ends at the last
    # spanwise crossing — the starting vortex, emitted normatively.
body_segments(panels, mu) -> (p1, p2, strengths, cores)
    # per-panel perimeter rings in REVERSED corner order Q1->Q4->Q3->Q2 with
    # G = mu_h, cores CORE_WING, minus the two body TE spanwise segments
    # (the other two legs of the analytic drop).
set_bound_strengths(wake, mu, k_up, k_low)
    # in place: wake["mu"][:n_bound, :] = mu[k_up] - mu[k_low]
relax_wake_step(wake, v_nodes, omega, x_near) -> float
    # one under-relaxed streamline remarch over wake["steps"]; velocities are
    # evaluated EXTERNALLY — the block owns no kernel; returns the near-field
    # displacement. The future convect(wake, v_nodes, dt) is a sibling policy
    # over the same nodes array (row 0 pinned in both).
wake_node_velocity(state) -> (n_nodes, 3)
    # onset(nodes) + source_velocity + induced_velocity over body_segments and
    # wake_segments: the ONLY composition point of the three advection terms.
```

### Assembly

```python
source_strengths(normals, u_rel) -> (N,)   # sigma = -sum(normals*u_rel, axis=1)
assemble_body(panels, chunk=300) -> blocks
    # geometry only, zero wake content, exact kernels; -1/2 hard-wired onto the
    # diagonal of D HERE (analytic interior limit; no inward offset).
wake_influence(colloc, wake) -> (P_bound (N, nspan), phi_known (N,))
    # P_bound: potential per unit strip doublet summed over rows [0, n_bound);
    # phi_known: frozen-row potentials times stored mu, rows [n_bound, nrows).
    # The general row split is MANDATORY in version one even though phi_known
    # is identically zero in steady — the unsteady extension is then a zero-line
    # change here. Taking colloc explicitly (not panels) is what later admits
    # supercavity bordering rows and an image wake without new functions.
fold_column(a_mat, rhs, b_col, targets, const)
    # general affine-constraint fold: a dependent strength = sum(coeff*unknown[k])
    # + const with influence column b_col; mutates a_mat, rhs in place. Version
    # one uses it only for the Morino Kutta fold; cavitation reuses it verbatim
    # for detachment coupling.
assemble_system(D, S, P_bound, phi_known, k_up, k_low,
                unknown_type, mu_fixed, sigma_fixed) -> (a_mat, rhs)
    # column k of a fresh copy: D[:,k] if unknown_type[k]==0 else S[:,k];
    # rhs = -D_masked @ mu_fixed - S_unmasked @ sigma_fixed - phi_known
    #       - P_bound @ (known part of the TE jumps),
    # i.e. EVERY known strength accumulates into rhs with the opposite sign,
    # doublet and source alike; when unknown_type[k_up[j]] == 1 the wake strip
    # column contributes P_bound[:,j]*mu_fixed[k_up[j]] to rhs instead of
    # folding into that column (sign - for k_low). With unknown_type all zero
    # and mu_fixed zero this reproduces A mu = -B sigma with the folded wake
    # bit for bit — the version-one pass-through, guarded by a unit test, plus
    # a mixed-mask test (one sigma-unknown panel, analytic right-hand side)
    # so the mask path cannot ship silently broken. The Kutta fold is in the
    # matrix at every iterate, never lagged. P_bound=None (sphere path) skips
    # the fold. N_sys = N always: a future supercavity borders the system
    # driver-side with np.block around a_mat — assemble_system never grows.
```

### Solver API

```python
solve_once(state) -> state
    # the shared step kernel: sigma = source_strengths(...); (P_b, phi_k) =
    # wake_influence(...); assemble_system with all-zero masks; LU solve;
    # set_bound_strengths. Asserts blocks_version == geom_version. Shared by
    # the steady driver and the future time driver; the cavity driver reuses
    # it for the wetted baseline and composes assemble_system directly in its
    # outer loop (its inner solve has nonzero masks).
init_state(points, onset, u_ref=1.0, config=None) -> state
    # build_panels, derived scalars, u_rel = onset(centroids), assemble_body,
    # and — if lifting — init_wake with the steady defaults. config overrides
    # the wake parameters too ({"n_bound": 1, "closure": False, "nwake": 0} or
    # {"wake": None} to defer wake creation to the driver), so the unsteady
    # driver never builds a steady wake only to discard it. No prompts, no I/O.
update_points(state, points, validate="fast") -> state
    # THE single geometry chokepoint: absolute coordinates, same wrap topology,
    # never a remesh. Rebuilds the metric (topology untouched), recomputes
    # u_rel/s_ref/x_near, drops blocks (bumps geom_version), re-pins wake row 0
    # to the moved TE; mu/sigma/wake persist as warm starts. validate="fast":
    # O(N) re-weld and re-pin repairs, area/volume/bowtie as hard failures,
    # warp as diagnostic; "full" reruns rules 1-9. The unsteady/coupling
    # extension adds a point_velocities=None keyword here — one signature.
interface(state) -> dict
    # nodes (M,3) reference coordinates, panel_nodes, weld_map, tip_flag:
    # everything a structural driver needs to build its transfer operators once.
solve(state, wake="relax", max_wake_iter=MAX_ITER,
      warm_start=True, verbose=False, callback=None) -> state
    # wake="relax": the steady relaxation loop below. wake="frozen": body
    # reassembly if the geometry moved, then exactly one solve_once — the
    # inner-coupling fast path and the entry point the future time driver
    # calls once per step (never bare solve_once, which would trip the version
    # assertion after update_points). wake="none": sphere path. Deterministic
    # and idempotent at fixed state; diagnostics only via verbose/callback.
    # Docstring warning: a driver that composes image blocks into state must
    # also own the relaxation loop, because wake_node_velocity composes exactly
    # three terms and would advect the wake WITHOUT image terms.
```

### Loads

```python
surface_gradient(panels, f) -> (N, 3)
    # standalone operator on ANY per-panel scalar: non-uniform central stencils
    # along wrap (continuous through the LE) and span, one-sided at the TE seam
    # and tips (the -1 sentinels), dual-basis conversion, tangent projection.
surface_velocity(panels, mu, u_rel) -> (N, 3)
    # V = (I - nn^T) u_rel + surface_gradient(panels, mu)
pressure_fields(panels, mu, u_rel, u_ref, dphi_dt=None, rho=1.0, p_ref=None)
    # -> {"v_surf", "cp", "p_gauge"};
    # p_gauge = 0.5*rho*(|u_rel|^2 - |V|^2) - rho*dphi_dt + p_ref(z_c),
    # with p_ref a callable z -> reference pressure, None meaning zero (v1) and
    # hydrostatic-minus-vapour later, so p_gauge is directly the cavitation
    # margin; cp is defined from the p_ref=None path only and is rho-free.
    # dphi_dt=None means zeros; the DRIVER owns mu_prev and the finite
    # difference, never this block.
integrate_loads(panels, p_gauge, rho, u_ref, s_ref, x_ref=None)
    # -> {"force_panels" (N,3), "force" (3,), "moment" (3,), CL, CD_pressure, CM}
get_loads(state, rho=1.0, p_ref=None) -> dict
    # one-call bundle on the flat panel index: mu, sigma, v_surf, cp, p_gauge,
    # dp, force (N,3), areas, normals, centroids, corners, warp, tip_flag,
    # resultants. Dimensional path needs rho only, never u_ref. A future
    # dphi_dt=None keyword extends it for unsteady coupling (additive).
spanwise_loading(panels, p_gauge, u_ref, rho=1.0) -> (y_sec (nspan,), cl (nspan,))
    # per-strip pressure-integrated section lift on the straight LE-to-TE
    # chord c_j (the same c_j as reference_area), normalised by q_ref*c_j;
    # feeds the loading plot and the symmetry test.
leakage(state) -> (max, rms)               # Dirichlet leakage diagnostic
trefftz_drag(wake, onset, rho, u_ref, s_ref, x_plane) -> dict
    # steady diagnostic on (wake, onset) alone — no panels, no mu; the time
    # driver simply never calls it. Returns cdi_2d, cdi_3d (closure legs
    # included), trace, kj_lift.
```

### Plotting and command-line interface

`plot_results(state, loads=None, save=None)` and `plot_sphere_cp(state)` import matplotlib lazily, return the Figure and never call show. `main()` owns every prompt and print: load, `init_state`, `solve`, `get_loads`, `trefftz_drag` at both design planes (0.8b and 1.0b downstream of the TE) printing the value and the spread, with the shear caveat in the summary line, then `plot_results` and `plt.show()`. Everything above `main()` is callable without a terminal or display.

## Drivers

### Steady (version one, the body of solve(wake="relax"))

```python
if blocks stale: blocks = assemble_body(panels); stamp version
if not warm_start or wake is None: wake = init_wake(steady defaults)
for it in range(max_wake_iter):
    solve_once(state)                       # fold -> LU -> sync strengths
    v = wake_node_velocity(state)           # onset + sources + rings
    delta = relax_wake_step(wake, v, OMEGA, x_near)
    if delta < TOL * chord_mean: break
solve_once(state)                           # final solve on the settled wake
```

This is the design document's relaxation loop verbatim; recomputing σ inside `solve_once` each iterate is numerically identical in steady flow and is what the shared step kernel requires.

### Future time loop (not built; verified on paper)

```python
state = init_state(points0, onset, u_ref,
                   config={"lifting": True, "n_bound": 1, "closure": False, "nwake": 0})
mu_prev = zeros(N); points_prev = points0
for step in range(nsteps):
    t += dt                                          # dt lives here only
    points = move(points0, t)                        # rigid or FE-supplied
    shed(wake, te_prev=te_nodes(points_prev),        # BEFORE update_points: the
         te_new=te_nodes(points), frac=0.25)         #   new row needs the old TE
    state = update_points(state, points)             # re-pins wake row 0
    state["u_rel"] = onset(centroids) - u_body(centroids, t)
    state = solve(state, wake="frozen")              # newest row in the matrix,
                                                     #   frozen rows on the rhs
    fields = pressure_fields(panels, mu, u_rel, u_ref,
                             dphi_dt=(state["mu"] - mu_prev)/dt, rho=rho)
    loads = integrate_loads(panels, fields["p_gauge"], rho, u_ref, s_ref)
    convect(wake, wake_node_velocity(state), dt); truncate(wake, nmax)
    mu_prev = state["mu"]; points_prev = points
```

The unsteady review re-derived the pressure: with V the body-relative surface velocity and dμ/dt the material-panel derivative, the inertial-frame unsteady Bernoulli collapses to p = ½ρ(|u_rel|² − |V|²) − ρ dμ/dt exactly — the u_body·∇φ cross terms cancel — so the driver-owned finite difference is correct, and it is well defined because panel h is the same material panel at every step and μ persists in the state. The loops differ only in the node-update policy, the shed/truncate calls, the dφ/dt argument and the omitted Trefftz call; every physics block appears once, unmodified. Bookkeeping recorded for the future blocks: `shed` prepends to nodes, mu and steps together; `truncate` adds a lumped far-vortex key consumed additively by `wake_influence` and `wake_segments`; `convect` leaves row 0 untouched.

### Coupling loop (verified with no corrections)

```python
import thick_panel_wing as fluid
fs = fluid.init_state(points0, onset, u_ref=UREF)
iface = fluid.interface(fs)          # driver builds both transfer operators once
for k in range(K_MAX):
    fs = fluid.update_points(fs, points0 + (H_d @ u).reshape(nwrap+1, nspan+1, 3))
    fs = fluid.solve(fs, wake="relax" if k == 0 else "frozen", warm_start=True)
    loads = fluid.get_loads(fs, rho=RHO_WATER)
    u = structural_update(H_f, loads, iface)   # Aitken/IQN wholly driver-owned
```

The weld map lets the driver write displacement rows for canonical nodes and copy to duplicates, so the TE seam and tip pinch stay exactly coincident. Frozen-wake determinism makes the residual a fixed function of the displacement, as quasi-Newton coupling requires. Cost per coupling iteration: one O(N²) body assembly, one wake influence, one LU — seconds at N = 960, the accepted contract; no incremental matrix machinery exists or is planned.

### Cavity outer loop (future; verified on paper)

The cavity driver keeps `blocks` immutable, builds a per-panel `unknown_type` mask (σ unknown on cavity panels), imposes the prescribed cavity tangential speed through per-strip cumulative sums of `ds_wrap` (dμ/ds relations folded with `fold_column` onto the last wetted panel, including the TE-in-cavity wake jump through P_bound), solves the mixed system from `assemble_system`, scatters by the mask, syncs the wake, stores the per-strip closure residual in `state["residuals"]`, and updates the detachment and closure indices between solves — all against public names, with the structured (i, j) layout carrying the cavity intervals. Ventilation adds the image-block composition described in the kernels section, with the driver owning wake relaxation.

## Extension points: property now → behaviour later

| # | Present in version one | Added later | Status |
| --- | --- | --- | --- |
| 1 | u_rel array in state; no surface block sees the onset closure | subtract u_body (time, coupling) | done: one line in update_points |
| 2 | wake n_bound field; wake_influence returns the general (P_bound, phi_known) split | n_bound = 1, frozen rows on the right-hand side (time) | done: no change to the block |
| 3 | wake closure flag honoured by both views, starting vortex normative | open sheet ending at the starting vortex (time) | done: and reused to close a truncated tail |
| 4 | steps as caller data; shed/convect/truncate as declared siblings | per-step shedding and convection (time) | done: shed subsumes convect |
| 5 | axis-0 sliceable wake arrays, no per-row caches | tail truncation with a lumped far vortex (time) | done: the FAR_FIELD quad IS the lump |
| 6 | pressure_fields dphi_dt argument, zeros path | driver-computed dμ/dt; get_loads gains the same keyword (time, coupling) | done: verified to 0.16% on sphere added mass |
| 7 | solve_once shared step kernel | the time loop as second consumer; the cavity loop reuses it for its wetted baseline | done: through solve(wake="frozen") |
| 8 | update_points chokepoint, topology–metric split, warm starts | point_velocities keyword; tight coupling (coupling) | partly: point_velocities and rigid added |
| 9 | frozen mode, determinism, verbose/callback gating | quasi-Newton coupling in the structural driver (coupling) | future |
| 10 | interface(), weld_map, per-panel loads with rho parameter | transfer operators, dimensional tractions (coupling) | future |
| 11 | unknown_type/mu_fixed/sigma_fixed in assemble_system; fold_column; D and S both retained | per-panel μ/σ swap, detachment coupling (cavitation) | future |
| 12 | flat N_sys = N; wake_influence and the kernels take explicit points | supercavity bordering driver-side via np.block; assemble_system never grows (cavitation) | future |
| 13 | corner-pure kernels, −1/2 added in assembly | mirror-image free-surface wrapper (ventilation) | future |
| 14 | p_ref callable, ds_wrap array, surface_gradient standalone | hydrostatic reference, cavity-height integrals (cavitation) | future |
| 15 | trefftz_drag isolated on (wake, onset) | omitted by the time driver | future |

Explicitly absent from version one, as in the design document: pressure-Kutta iteration, side-edge tip shedding, the wake-sweep contingencies (designed, disabled), blunt TEs, intermediate-Froude free surfaces.

## What the time extension actually cost

Hooks 1 to 7 were exercised and they held: `wake_influence` needed no change at all, `pressure_fields(dphi_dt=...)` needed none, `assemble_system` needed none, and `solve(wake="frozen")` is the per-step entry point as written. The array wake sliced and grew as designed. Against that, the paper design missed five things, all found by running the code, and they are the amendments this repository carries.

1. **The zero-row wake crashed.** `wake_quads` returned an array of shape (0,) rather than (0, 4, 3) and `wake_influence` then raised an `IndexError`. That is the state `init_state(config={"nwake": 0})` produces and the state the time driver starts in, so the very first hook to be used was broken. Both now guard it, and `set_bound_strengths` clamps `n_bound` to the rows present instead of missing its slice silently.
2. **The spanwise inter-row segment sign in this document was wrong** (section Wake, `wake_segments`). The code was right. The path is identically zero in steady flow, so nothing had ever read it.
3. **`shed` cannot be given the trailing-edge velocity by `wake_node_velocity`.** Row 0 lies on the perimeter rings and the source-panel edges, where the kernels are singular and return a core-dependent near-cancellation: 0.08 U where the answer is U. `te_convection_velocity` takes it from the mean of the upper and lower surface velocities instead, which the Kutta condition makes well defined. The paper design had not noticed that the steady loop only ever uses the DIRECTION of that velocity.
4. **`shed` and `convect` are one operation, not two.** The design listed them as siblings to be called in sequence; in fact the particle that leaves the trailing edge is the same particle the convection moves, so shedding is prepending a row and convecting everything. The separate `convect` remains, for a driver that wants to settle a wake without shedding.
5. **`update_points` needed a second keyword, not one.** `point_velocities` was foreseen; `rigid` was not. The doublet and source blocks are invariant under a rotation and a translation, which makes the reassembly - 0.63 s at 960 panels, the largest single cost of a step - avoidable for every prescribed rigid manoeuvre. The hook table's row 8 is therefore two keywords wide.

One design decision was vindicated more strongly than expected: because no surface block sees the onset callable, body motion entered through the single line `u_rel = onset(centroids) - u_body` and nothing downstream knew. The wake needed no frame change at all, since the source and ring velocities that advect it are already absolute.

## Tests

`test_thick_panel_wing.py` uses the standard-library unittest — pytest is not installed in the pinned environment and the repository declares numpy and matplotlib only — and runs as `python3 test_thick_panel_wing.py`, with the slow regressions gated behind an environment variable (THICK_SLOW=1) rather than a marker. Meshes are generated in-process: a local sphere-wrap helper (poles as pinch, seam welded exactly) and the wing meshes by importing `make_thick_sample_inputs`, whose mesh-building functions are pure with file writes only under `main()`, following the existing generator's pattern.

Fast set: the five sign-pinning unit checks of the design document (closed constant-μ surface with row sums −1; source-panel jump and far field; doublet–ring equivalence on the reversed perimeter; the TE three-segment cancellation; the sphere Dirichlet solve against Cp = 1 − (9/4)sin²θ with the design tolerances and the leakage diagnostic reported), plus the architecture invariants: the all-zero mask pass-through bit for bit; a mixed-mask solve with one σ-unknown panel against an analytic right-hand side; a source-velocity probe exactly at a TE node (finite, no NaN — the edge-clamp guard); mesh validation (open-TE NACA rejected with the gap message and the −0.1036 hint, bowtie caught, reversed wrap caught, sub-tolerance gap welded with a warning); solve determinism and warm-start persistence across update_points; TE-seam and tip stencils unpolluted by the μ jump, with a linear field reproduced exactly on the swept sample wing through the dual basis.

Slow set (THICK_SLOW=1): the frozen-wake thin limit at f/c = 6% (C_L = 0.4346 ± 0.009), loading symmetry ≤ 10⁻¹⁰, and the shear ratio 0.955 ± 0.010. The AR sweep and refinement studies remain the manual verification programme of the design document.

## Build order

Each step gates on its checks before the next starts: (1) geometry, validation and topology–metric split, with the signed-volume and bowtie tests; (2) kernels with the sign-pinning checks including the source edge clamp; (3) state API and the sphere path (init_state, assemble_body, assemble_system pass-through, solve_once, leakage); (4) wake views and the Kutta fold, TE-cancellation check, frozen-wake wing; (5) relaxation and the wake-sharing cases; (6) loads, spanwise loading and Trefftz drag; (7) plotting and CLI; (8) the verification table of the design document, updating both documents with actuals.

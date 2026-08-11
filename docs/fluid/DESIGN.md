# Design: thick_panel_wing.py

**Status.** This document specifies the STEADY formulation, which the time-dependent code of this repository carries over unchanged: every result tabulated below was reproduced here, and the automated checks in `test_thick_panel_wing.py` all pass. The time dimension is specified in [UNSTEADY.md](UNSTEADY.md), which owns the shedding scheme, the unsteady pressure and the unsteady verification. Where this document says "steady" or "future work on time dependence", read UNSTEADY.md alongside it; the three approximations that shear introduces, the Kutta condition, the kernels and every sign convention below are unchanged by the extension.

This document specifies `thick_panel_wing.py`, which extends the thin lifting-surface code `panel_wing.py` of the companion repository https://github.com/ignaziomviola/free-wake-lifting-surface to wings with thickness. It records the formulation, the conventions that pin every sign, the reuse of the free-wake machinery, the verification results and the risks found in design review and implementation. The code is implemented and every case in the verification tables below was run against it; the automated checks live in `test_thick_panel_wing.py` (14 tests, all passing). Read README.md for usage; the code structure is in ARCHITECTURE.md, and the handover notes of the companion repository record the thin-surface results quoted here as references.

Throughout, μ denotes the panel doublet strength and σ the panel source strength; no viscosity appears in an inviscid code, so μ is unambiguous. The maximum section thickness is f and the thickness-to-chord ratio f/c, following the lab notation.

Last updated 9 August 2026.

## Why a source–doublet method with an internal Dirichlet condition

The existing codes impose flow tangency on the mean camber surface, so thickness effects are absent by construction. The extension panels the actual wing surface with constant-strength source and doublet panels and sets the perturbation potential to zero just inside the surface — the Morino formulation. Three reasons fix this choice.

1. The source strengths are fixed pointwise by the onset flow, σ = −n·U_onset, before any solve, so the sheared onset (U(z), 0, 0) carries over exactly as in `panel_wing.py`, which already samples the onset pointwise in its tangency condition. A formulation requiring an onset potential would exclude shear, because a rotational onset has none.
2. A constant-doublet wake strip is exactly a vortex ring around its perimeter, so the free-wake relaxation of `panel_wing.py` — `segment_velocities`, the van Garrel regularisation, the under-relaxed advection, the near-field convergence criterion — is reused with wake strengths given by differences of trailing-edge doublets.
3. The discrete system is a Fredholm equation of the second kind: eigenvalues cluster near −1/2 and the condition number is O(10)–O(10²), essentially mesh-independent. The rejected alternative, a Neumann condition on surface doublets alone, is near-singular on a closed body.

One property is lost relative to the thin code and must be stated: the Neumann collocation of `panel_wing.py` enforces zero normal velocity exactly at its collocation points, whereas the internal Dirichlet condition makes the interior potential vanish only at the N interior points, so external tangency holds approximately everywhere — the known leakage of low-order Dirichlet panel methods — and converges with refinement. A post-solve leakage diagnostic (the normal velocity at panel-edge midpoints, maximum and root mean square) is part of the design.

## Conventions

These bind every formula in this document and in the eventual code. Printed sources differ between editions in logarithm ordering and doublet signs — Katz and Plotkin define μ with the opposite sign — so the unit checks at the end of this document, not the textbooks, are the authority.

- Frame: onset flow nominally along +x, z up, y spanwise, starboard tip at +y; ρ = 1 as in `panel_wing.py`.
- Mesh: NumPy archive, key `points`, shape (nwrap+1, nspan+1, 3). Index i wraps around the section from the lower-surface trailing edge (TE) at i = 0, forward under the lower surface, around the leading edge (LE), back along the upper surface to the TE at i = nwrap; index j runs from the port tip to the starboard tip. A sharp closed TE is required: points[0, j] equals points[nwrap, j].
- Panel (i, j), flat index h = i·nspan + j, N = nwrap·nspan panels; corners in the order Q1 = points[i, j], Q2 = points[i+1, j], Q3 = points[i+1, j+1], Q4 = points[i, j+1].
- Outward unit normal n̂ from the diagonal cross product (Q3 − Q1) × (Q4 − Q2), normalised. The identity (Q3 − Q1) × (Q4 − Q2) = 2 e_i × e_j, with e_i, e_j the bimedian vectors of any quadrilateral, planar or not, guarantees outwardness from the wrap ordering alone — on the upper surface it gives +z, on the lower −z, and camber, twist, sweep, dihedral and the tip pinch deform these cases continuously without inverting a panel. No orientation test or flip is needed; the coordinate flip of `panel_wing.py` (`nvec[2] >= 0`) must not be carried over, as it cannot work on a closed body. The corner order Q1–Q4 is anticlockwise about +n̂.
- Point source of strength q: φ = −q/(4πr), so the velocity is radially outward for positive q.
- Solid angle Ω(P) = ∬ n̂·(P − Q)/|P − Q|³ dS, positive on the +n̂ side. Constant-doublet panel with axis +n̂: φ_μ(P) = (μ/4π) Ω(P).
- Jump relations across a sheet, outer meaning the +n̂ side: μ = φ_out − φ_in and σ = ∂φ/∂n|_out − ∂φ/∂n|_in.
- Ring equivalence: a constant-doublet panel of strength μ is exactly a vortex ring of circulation Γ = μ along its perimeter traversed clockwise when viewed from the +n̂ side, i.e. in reversed corner order Q1 → Q4 → Q3 → Q2. This is the hook into `segment_velocities` and was verified in review three independent ways (axis limit of a disc, the sheet vorticity n̂ × ∇μ, and the trailing-vortex senses below).

## Formulation

### Decomposition and the sheared onset

The velocity is u = U_onset(x) + ∇φ with φ harmonic and decaying at infinity. Continuity is exact, the onset alone is an exact Euler solution (parallel shear), and the wake advects in the pointwise total velocity, exactly as in `panel_wing.py`. In uniform flow the method reduces to the classical Morino formulation with no approximation. Under shear, four approximations enter, all of the same family as in the thin code:

1. the onset vorticity (0, U′(z), 0) makes the true perturbation rotational; the neglected residual is the Lamb term −∇φ × ω_onset, of relative size O((U′c/U)·(|∇φ|/U));
2. the surface pressure uses the Bernoulli constant of the onset streamline at the collocation height, accurate to O(U′·δz) with δz the streamline displacement;
3. σ is constant per panel but samples U(z) at the centroid only — a discretisation error absent in uniform flow;
4. a strip-constant wake doublet follows from pressure continuity only when both sides of the sheet share a Bernoulli constant; under shear the two sides originate at different heights, an O(U′·δz) approximation.

Because the onset is not a gradient, no total potential exists; the formulation is written in the perturbation potential alone, and the onset enters only through pointwise values. This is what admits shear.

### Internal Dirichlet condition

The perturbation potential is represented by source and doublet layers on the surface S plus a doublet layer μ_w on the wake W,

φ(P) = −(1/4π) ∮_S σ/r dS + (1/4π) ∮_S μ n̂·(P−Q)/r³ dS + (1/4π) ∮_W μ_w n̂_w·(P−Q)/r³ dS.

Choosing the interior potential to vanish makes μ the exterior surface perturbation potential itself and fixes the sources pointwise from external tangency,

σ_k = −n̂_k · U_onset(c_k),

with c_k the panel centroid. The N unknowns μ are determined by requiring φ = 0 at the interior side of each centroid. Uniqueness argument: a harmonic interior potential vanishing on the whole inner boundary vanishes identically, so its normal derivative vanishes and the source jump returns exterior tangency; nothing requires the onset to be potential. A constant added to μ is not a null mode — a closed constant-μ sheet shifts the interior potential by −μ — so the matrix is well posed.

### Influence kernels

- Doublet potential: per quadrilateral as triangles with the van Oosterom–Strackee formula, Ω_t = 2 atan2(a1·(a2×a3), a1a2a3 + (a1·a2)a3 + (a2·a3)a1 + (a3·a1)a2), averaged over BOTH diagonal splits. A single fixed split is not mirror-covariant — mirroring a panel lands the split on the other diagonal, which on warped panels shifted the spanning surface and the collocation centroids and broke the spanwise loading symmetry at O(10⁻¹) during implementation; the averaged split restored it to 10⁻¹⁰. The average is deterministic, iteration-fixed, and remains a valid potential (the mean of two closed spanning surfaces, so the closed-body row-sum identity holds exactly and the perimeter-ring velocity equivalence is unchanged). Centroids average the two splits' area-weighted triangle centroids for the same reason.
- Doublet velocity: no new kernel — Biot–Savart on the perimeter in reversed corner order via `segment_velocities`, cores CORE_WING on the surface and rc_wake on the wake. Velocities are evaluated on ring perimeters (wake nodes lie on the rings they bound); safety comes from the vanishing cross product on the segment axis and the van Garrel core, not from avoidance, and near-miss passes of wake nodes over body rings can spike exactly as in the thin code. With a finite core the wake ring velocity is not the exact gradient of the solid-angle potential; the two differ by O((rc/d)²), a deliberate regularisation.
- Source potential and velocity: the Hess flat-panel formulas in the panel frame, with the corners projected onto the mean plane through the corner mean, normal n̂. The normal velocity Ω/4π per unit σ is an exact identity giving ±σ/2 across the faces. The source layer therefore lives on a surface that differs from the doublet spanning surface by O(warp) and whose union is not exactly watertight — standard Hess practice, acceptable at the warp tolerance below, and stated here so nobody assumes full consistency.
- Far field: point-source and point-doublet forms beyond about five panel diagonals, needed only for the per-iteration wake work; the body matrices are assembled once and stay exact.
- Warp diagnostic: max |(corner − mean)·n̂| / sqrt(area) per panel, warning above 0.1. Tip panels adjacent to the LE will trigger it (Section Geometry); their pressures are flagged accordingly.

### Self-influence and collocation

Collocation points sit at the panel centroids on the surface; the interior-side limit is taken analytically by hard-wiring the own-panel doublet coefficient to −1/2 (the in-plane solid-angle evaluation returns the principal value, zero, not the limit). Do not displace points inward by a finite ε: the displacement contaminates neighbour influences at O(ε/d) and can cross the opposite surface inside the TE wedge — a classic, hard-to-diagnose bug.

### Morino Kutta condition

The wake strip behind spanwise column j carries the constant doublet

μ_w(j) = μ[K_up(j)] − μ[K_low(j)], K_up(j) = (nwrap−1)·nspan + j, K_low(j) = j.

This adds no unknowns: the wake potential-influence columns W_j fold into the matrix as A[:, K_up(j)] += W_j and A[:, K_low(j)] −= W_j. The folding must be in the matrix, never lagged to the previous iterate: it is what regularises the near-duplicate TE row pairs (the (+1, −1) direction acquires an O(1) response at every collocation point); with a lagged wake the TE pair is near-singular and the solve is noise-dominated.

At the TE, three coincident spanwise segments arise in the ring picture: the upper TE panel contributes −μ_up on a +y segment, the lower +μ_low, the first wake quad +μ_w. Their sum vanishes identically when the Morino condition holds, so no concentrated spanwise vortex survives at the sharp TE — the finite-velocity content of the Kutta condition. The three segments are dropped analytically from the velocity segment list (summing them numerically would need identical cores); this exactness holds at every iterate because μ_w is defined by the folding.

Morino's condition equates potential jumps, not pressures: a finite pressure mismatch remains at swept TEs, growing with sweep. The iterative pressure-Kutta condition is recorded as future work and should not be implemented until the plain condition demonstrably limits accuracy.

### Discrete system and cost

A is dense, square, nonsymmetric, N × N with the second-kind structure −½I + K. At the default N = 960 (48 wrap × 20 span): 8 MB per matrix, LU solve below 0.1 s, one-off assembly of the body blocks a few seconds (chunked over collocation points as in `induced_velocity`), and per wake iteration the strip potentials (N × nspan·NWAKE solid angles, about 1 s) plus relaxation velocities. Thirty iterations run in about one minute, comparable with `panel_wing.py`.

## Geometry and inputs

### Mesh format and validation

The archive carries two keys: `points` as above and `mesh_format`, a zero-dimensional string array with value `"thick-wrap-1"`. The explicit marker is preferred over geometric detection because it yields exact diagnostics and a version string; the TE-closure test remains as validation, not as the discriminator. A one-line guard in `panel_wing.load_mesh` — rejecting `thick-wrap*` files with the message that they belong to `thick_panel_wing.py` — closes the reverse confusion and is the only change proposed to the existing code.

Validation rules, in order, with d_box the bounding-box diagonal:

1. `points` present with shape (nwrap+1, nspan+1, 3);
2. `mesh_format` present, else the file is diagnosed as a camber-surface mesh for `panel_wing.py`;
3. `mesh_format == "thick-wrap-1"`;
4. nwrap ≥ 6 and even, nspan ≥ 2;
5. TE closure: gaps up to 1e−6·d_box are welded by averaging rows 0 and nwrap, with a warning; larger gaps are rejected with the measured gap and the hint that standard NACA four-digit sections (last coefficient −0.1015) have an open TE of about 0.21% chord and need the closed-TE coefficient −0.1036;
6. tip pinch: points[i, jt] equals points[nwrap−i, jt] at both tips, to 1e−9·d_box;
7. all panel areas above 1e−12·d_box²;
8. signed volume positive, V = (1/3) Σ_h A_h (c_h·n̂_h), else the wrap direction is reversed and the fix `points[::-1]` is suggested;
9. both corner triangles (Q1, Q2, Q3) and (Q1, Q3, Q4) of every panel have normals with positive projection on n̂ — this catches locally crossed (bowtie) panels that rules 7 and 8 miss, the local shoelace being provably always positive and useless as a check.

Abbreviation used above and below: National Advisory Committee for Aeronautics (NACA).

### Tip closure: the pinched wedge

Both tip stations are collapsed onto the section camber line, so the surface closes onto a zero-thickness edge like the sharp TE. Three options were compared. The open tip leaves the interior region undefined and admits a near-null mode. Cap-fan panels are worse than they look: the cap slivers near the LE and TE have small areas and collocation points close to two opposing surface rows at once — three-way near-degenerate row clusters — plus unstructured bookkeeping. The pinched wedge keeps a watertight body, proper quadrilateral panels everywhere (inboard edge on the full section, outboard edge on the camber line) and an unbroken structured grid; it is standard practice in codes of this class.

The conditioning mechanism to watch is not any singular matrix entry — doublet coefficients are bounded solid angles, |C| ≤ 1/2 — but near-duplicate row pairs: a mirrored upper/lower pair whose centroids approach each other tends to the rank-one block [[−1/2, −1/2], [−1/2, −1/2]] with μ_up − μ_low the near-null direction. Severity scales with the local thickness over the local chordwise panel size, not with the spanwise spacing (the mirrored centroids stay about half the local thickness apart regardless of the strip width). Two consequences:

- at the TE the same wedge geometry appears on every strip, but there the wake folding regularises the pair (previous section); the tip pinch pairs get no such help and rely on the O(1) geometric deviations — for NACA 0012 at 24 chordwise panels per surface the separation-to-panel-size ratios stay O(1) over the whole tip strip (about 0.06c against 0.065c at mid-chord) and the closure is safe; thin sections combined with aggressive chordwise clustering near the pinch corners are the case to monitor;
- the closed NACA 0012 TE wedge has a half-angle of arctan(1.211 f/c) ≈ 8.3° and a TE centroid separation of about 0.145 times the TE panel length — a refinement-independent ratio, so cosine clustering does not progressively degrade the TE pair.

Tip-strip panels adjacent to the LE are strongly warped (the inboard edge wraps the nose, the outboard edge lies flat on the camber line); the warp warning will fire there and tip-strip pressures are flagged as noisy in the output. Fallback if conditioning proves poor for thin sections: a zipped-strip closure, which the mirror pairing of rule 6 already permits without regenerating meshes.

### Panel geometry

Areas use the vector area ½|(Q3−Q1)×(Q4−Q2)|, exact for any quadrilateral boundary. Centroids are the area-weighted mean of the two triangle centroids of the projected quad. Each panel carries a right-handed frame (t̂_c, t̂_s, n̂) built from the wrap-direction edge-midpoint vector, for the local 2D corner coordinates of the Hess integrals and for the surface gradient below. Neighbour maps along wrap and span serve the finite differences; the wrap direction is continuous through the LE — one benefit of the wrap format — but never across the TE seam, where μ jumps by μ_w, and never across the tip pinch.

### Sample-input generator

A new file `make_thick_sample_inputs.py` reuses the planform parameterisation of `make_sample_inputs.py` (span, root chord, taper, sweep, linear twist about the quarter chord, parabolic camber) and adds the NACA four-digit half-thickness in its closed-TE variant,

y_f(ξ)/c = 5 (f/c) (0.2969 ξ^½ − 0.1260 ξ − 0.3516 ξ² + 0.2843 ξ³ − 0.1036 ξ⁴),

whose coefficients sum to zero so the TE closes exactly; y_f is the half-thickness, with maximum 0.499 f at ξ = 0.3, so f/c is the thickness ratio. Thickness is added perpendicular to the camber line and scales with the local chord. The wrap coordinate uses cosine clustering, ξ_i = ½(1 + cos(πi/n_c)) with nwrap = 2 n_c, which clusters at the LE and TE, mirror-pairs the wrap stations as the tip pinch requires, and answers the slow chordwise convergence recorded in HANDOVER.md. Two exactness rules: assign the closing TE row and the pinched tip stations by copying arrays, never by trusting cos(2π).

| file | planform | section | mesh | purpose |
| --- | --- | --- | --- | --- |
| `naca0012_rect_mesh.npz` | rectangular, span 8, chord 1 | NACA 0012 | n_c = 24, nspan = 20, N = 960 | thin-limit and AR-8 regressions |
| `thick_wing_mesh.npz` | taper 0.5, sweep 15°, twist −3°, camber 2% | f/c = 0.12 | n_c = 24, nspan = 20 | thick counterpart of the sample wing; shear case |
| `naca0012_quasi2d_mesh.npz` | rectangular, span 40, chord 1 | NACA 0012 | n_c = 32, nspan = 12 cosine span | two-dimensional references; chordwise convergence |

The velocity profiles, `pitch_mesh` (a rigid rotation preserves closure, pinch and orientation), `load_velocity_profile` and `make_onset` carry over unchanged.

## Wake

### Strips, filaments and the trailing-edge identity

One strip per spanwise column, bounded by nspan+1 free filaments shed at the TE nodes; wake quad k of strip j has corners (node_j[k], node_j[k+1], node_{j+1}[k+1], node_{j+1}[k]), an order whose normal continues the upper-surface normal across the TE. Interior spanwise edges of each constant-μ_w chain cancel pairwise, leaving the strip's perimeter ring; the filament at span node j therefore carries, downstream-oriented,

Γ_j = μ_w(j−1) − μ_w(j), with μ_w(−1) = μ_w(nspan) = 0,

so the starboard tip filament carries +μ_w at positive lift — downwash inboard, the correct sense, verified in review edge by edge. The wake is stored as arrays rather than the per-filament dictionaries of `panel_wing.py` — nodes (nrows+1, nspan+1, 3) and per-quad doublets (nrows, nspan) — so that the future unsteady extension can shed and truncate rows by plain slicing; view functions emit the same (p1, p2, strengths, cores) arrays the shared kernels consume, and the review confirmed the views reproduce the folding above numerically exactly (ARCHITECTURE.md). The folding remains two-level: segment strengths from row-wise μ differences for advection; a strip-level map folds the potential columns into the matrix, because filaments are per span node while potentials are per strip, and an isolated open filament has no doublet-sheet potential.

### One geometry, two kernels

The Dirichlet matrix needs the wake potential: each strip is a chain of doublet quads over the same nodes the velocity path uses, evaluated with the same averaged-split solid angle as the body panels — deterministic and identical across iterations, so the discrete jump surface is well defined and matrix entries change continuously except when the sheet itself sweeps a point. Inside the sliver between the two splits the averaged potential carries two half-jumps of μ/2 instead of one full jump, a milder version of the near-sheet hazard.

The known hazard is a wake panel sweeping across a collocation point during relaxation (roll-up near the tips, a wake pinned onto a thick suction surface at high incidence, drift under shear): the affected entries jump by O(1) against typical entries of O(panel solid angle/4π), perturbing the whole solution at O(μ_w·‖A⁻¹‖), and the loop can limit-cycle. Mitigations in order: the existing under-relaxation OMEGA = 0.7; a rigid TE stub (exclude the first wake step from advection, 0.2–0.3 local chords along the TE bisector) as a documented contingency; optionally an exclusion shell around the body with a warning. Version one ships with verbatim relaxation and the stub as contingency.

The far-field closure mirrors `filament_segments`: one closing quad of length FAR_FIELD along the local onset per strip. An open constant-doublet sheet has a well-defined potential, but a truncated strip would leave a spurious starting-vortex influence of O(μ_w Δy/(4π(2b)²)) at two spans; closing at FAR_FIELD = 1e4 reduces the residual inconsistency between the potential and velocity descriptions to O(1e−8). The atan2 solid angle is unconditionally valid per planar triangle, so the elongated closure quad is numerically safe.

### Relaxation loop

```
build panels; assemble body doublet and source blocks once; rhs = −B σ
initialise straight filaments along the local onset (initial_nodes, no waypoints)
repeat up to MAX_ITER:
    strip potentials W for the current wake → fold into A → solve A μ = rhs
    advect filaments in onset + source-panel velocity + ring Biot–Savart
        (body rings Γ = μ, wake filaments Γ_j), under-relaxed by OMEGA
    stop when the near-field displacement < TOL·mean chord (x < x_TE + one span)
final assemble and solve; loads
```

The advection velocity gains the source-panel term relative to `relax_filaments`, so that function is re-implemented with the same structure rather than imported. NWAKE, WAKE_LENGTH_SPANS, OMEGA, TOL, MAX_ITER, FAR_FIELD, CORE_WING and RC_WAKE_FRACTION carry over with their values; TIP_FINE_STEP_CHORDS and TIP_FINE_LENGTH_CHORDS are dropped, because the per-station tip filaments of the thin code have no analogue here: the closed body has no open side edge, ∇_s μ turns smoothly around the tip, and separation exists only where a sheet is attached. Version one sheds at the TE only. The expected consequence is a small lift deficit from the missing chordwise tip-vortex development — the thin code gained 1.7% of lift from its tip filaments at aspect ratio six and α = 8° — and a side-edge Kutta condition along a prescribed tip separation line is future work.

## Loads

### Surface velocity

With μ the exterior perturbation potential, the exterior surface velocity is

V = (I − n̂n̂ᵀ) U_onset(c) + ∇_s μ,

the tangential onset plus the surface gradient of μ; the source term is the normal component and cancels by construction. Two documented bug traps: the sign is +∇_s μ in this convention (Katz and Plotkin's μ has the opposite sign, so their tangential-velocity formula carries a minus that this code must not), and no separate tangential source term exists — μ is the total outer potential, so adding a local σ-sheet term double-counts. The sphere anchors the sign alone: μ = (Ua/2)cos θ gives |V| = 1.5 U sin θ.

The gradient uses second-order finite differences on the structured grid: non-uniform three-point central stencils along wrap (continuous through the LE) and span; one-sided three-point stencils at the TE seam and the tips, never differencing across either. Two implementation findings harden the scheme. First, the wrap spacing is max(centroid chord, mean panel arc extent): at a thin, under-resolved LE the wrap chain folds back on itself and centroid chords collapse while μ jumps across the stagnation region — with pure chords the LE rows produced |Cp| of O(10³) at f/c = 2% — while at collapsed panels (sphere poles, pinched tips) the arc extents underestimate the true separation, so neither measure alone serves; the wrap direction is the panel's own unit tangent, which stays defined across the fold. Second, the non-uniform stencil's effective direction is the weighted vector (h⁻û⁺ + h⁺û⁻)/(h⁺ + h⁻), not the normalised chord between the outer neighbours; the dual basis is built from the effective directions (a resolution-independent O(5%) error otherwise). On swept and tapered wings the grid directions are not orthogonal and the dual-basis conversion is what makes the swept sample wing come out right — the verification pairs are chosen to catch exactly that. The final velocity is projected onto the tangent plane.

### Pressure and reference quantities

Uniform onset: exact Bernoulli, Cp = 1 − |V|²/U_ref². Sheared onset: a parallel shear stream has uniform static pressure, so each surface point is tagged with the stagnation pressure of the onset streamline at its own height,

p − p_∞ = ½ρ(U(z_c)² − |V|²),

accurate to O(ρ U U′ δz) — the streamline-displacement error, a few per cent of the dynamic pressure for the sample shear, and the same approximation level as the thin code's Kutta–Joukowski loads with local velocity. The reference area must be the projected planform, Σ_j c_j Δy_j with c_j the straight LE-to-TE section chord — half the wetted area exceeds it by about 2% at f/c = 0.12 and would silently corrupt every comparison with the thin code. The reference speed remains user-supplied.

### Forces

Lift, the moment about the root quarter chord and the pressure drag come from Δp integration over panels. Near-field pressure drag is unreliable at this order: the induced drag is O(C_L²/(πAR)) ≈ 0.009 while the cancelling fore-and-aft suctions are O(1), so the truncation errors that fail to cancel in x are comparable with the answer, and the sphere case measures this error floor directly. The trusted induced drag comes from a Trefftz-plane evaluation on the relaxed wake:

C_Di = −(ρ/(q_ref S_ref)) · ½ Σ_s μ_w(s) (v_cf,s · n̂_s) L_s,

where the trace is the polyline of filament crossings of the plane x = x_TE + b, ordered port to starboard, n̂_s = x̂ × t̂_s points to the upper side, and v_cf is the crossflow velocity from the wake filaments only — body and bound contributions are excluded, since the Green-identity derivation involves the wake cut alone and adding them double-counts momentum already carried by pressure. The formula was re-derived in review, including the sign chain (positive μ_w with downwash gives positive drag). Two evaluation modes cross-check each other: regularised two-dimensional point vortices at the crossings, and the existing three-dimensional `induced_velocity` over the wake segments including the far-field closure legs — omitting the closure biases the drag low by several per cent. Planes at 0.8b and 1.0b give the quoted value and spread; a plane at 1.2b lies beyond the convergence-controlled near field and is an error indicator only. The formula tolerates a folded trace but not a self-intersecting one; strong roll-up cases should be checked visually. A far-field Kutta–Joukowski lift over the trace is a cheap consistency diagnostic against the pressure-integrated lift (2% in uniform flow). Under shear the crossflow-energy argument does not strictly hold and the Trefftz drag is reported as indicative.

### Plots

The six panels of `plot_results` are kept (the wake views are unchanged since the machinery is shared) and two are added: chordwise −Cp at the sections nearest 2y/b = 0, 0.5 and 0.85, upper and lower branches split at the LE index, and a planform Cp map of the upper surface. The sphere test writes its own figure of computed against analytic Cp along a meridian.

## Verification results

All cases below were run and the values are current for the committed code (`python3 test_thick_panel_wing.py` holds the fast set; `THICK_SLOW=1` adds the pinned regressions). LL denotes the lifting-line estimate with the thickness-corrected slope a₀ = 2π(1 + 0.77 f/c); the two-dimensional NACA 0012 reference at α = 5° is c_l = 0.599 (thin-aerofoil 0.5483 with the corrected slope), consistent with two-dimensional panel codes (XFOIL inviscid ≈ 0.60). All wake results use RC_WAKE_FRACTION = 0.2.

Exact analytical references, run by `verify_analytic.py`. These are the strongest checks in the programme, because the reference is a closed-form solution for a body of finite thickness rather than a correlation or the thin code:

| case | result |
| --- | --- |
| circular section, α = 0, midspan of an aspect-ratio-40 wing, 80 wrap panels | Cp against the exact 1 − 4sin²φ: maximum error 0.0047, root mean square 0.0030; maximum surface speed 1.99727 against exactly 2; lift 4×10⁻¹⁴ |
| the same under refinement, 40 → 80 → 160 wrap panels | root mean square error 0.0071 → 0.0030 → 0.0020. The residual is not a discretisation failure but the genuine three-dimensionality of a finite wing: it falls to 0.00073 when the aspect ratio is doubled to 80, and is unchanged (0.0019) when the spanwise panels are doubled instead |
| Joukowski section, 10.7% thick, 2.8% cambered, cusped TE, α = 5°, 120 wrap panels | at matched circulation the pressure agrees with the conformal solution to a root mean square of 0.023, the suction peak to 1.4% (−1.9385 against −1.9125); the maximum error 0.174 sits on the two panels at the nose, where Cp swings by three across one panel |
| the same under refinement, 60 → 120 → 180 wrap panels | root mean square 0.050 → 0.023 → 0.018; suction peak error 5.2% → 1.4% → 0.9%; midspan section lift 0.9229 → 0.9349 → 0.9483, approaching the exact two-dimensional 0.9658 from below and remaining above 0.9174, that value less the mean downwash, as it must because the downwash of a rectangular wing is smallest at midspan |

This case verifies thickness, camber and the circulation set by the Kutta condition together, against an exact solution rather than the empirical thickness correction used in the rows below.

Closed body, no wake:

| case | result |
| --- | --- |
| unit sphere, 24×24 (576 panels) | Cp against 1 − (9/4)sin²θ: rms 0.0131, max 0.031 away from the pole rows, 0.060 within them; max speed 1.509 (error 0.009 U); force coefficients 10⁻¹⁶ and 4.5×10⁻⁴ — the near-field drag error floor |
| sphere refinement 12² → 24² → 48² | Cp rms 0.0465 → 0.0131 → 0.0044, observed order ≈ 1.7; drag floor 1.4×10⁻² → 4.5×10⁻⁴ → 1.4×10⁻⁵ |
| Dirichlet leakage (rms normal velocity just outside, per onset speed) | 0.131 → 0.068 → 0.036 under the same refinement, first order; the wing meshes give 0.079 → 0.043 from 32×14 to 80×36 |

Lifting cases (free wake, rectangular NACA 0012 at 48×20 and α = 5° unless stated):

| case | result |
| --- | --- |
| AR sweep 4, 8, 20 | C_L = 0.3316, 0.4267, 0.5093 against LL 0.3874, 0.4705, 0.5400: deficits 14.4%, 9.3%, 5.7%, shrinking monotonically as in the thin code (whose deficits against plain LL were 9.1%, 4.7%, 1.8%); the larger deficit tracks the Dirichlet leakage, which the Neumann thin code does not have |
| chordwise/span refinement, AR 8, frozen wake | C_L = 0.4361, 0.4327, 0.4310, 0.4302 for 32×14, 48×20, 64×28, 80×36 — monotone from above, converged ≈ 0.430 |
| thin limit, f/c = 6% | C_L = 0.4218 at 48×20 (pinned regression ± 0.006), converging to 0.4233; against the thin code's converged 0.4114 the thickness ratio is 1.029, within 0.8% of the LL-damped factor 1.037. The original target 0.4346 compared against the thin code's coarse-mesh 0.4192 and conflated two resolutions |
| thin limit, f/c = 2% | C_L = 0.4267, inflated: the nose radius 1.1019 (f/c)² c = 4×10⁻⁴ c is ten times smaller than the LE panel, and the resolved suction peak reaches Cp = −13.7; the thickness ordering (1–3% physical effect) is below the discretisation spread at this mesh. cond(A) = 3.6×10³, 1.3×10³, 7.1×10² at f/c = 2, 6, 12% — the solve itself stays healthy; μ_w falls smoothly with thickness (0.233, 0.238, 0.246 at midspan) |
| quasi-two-dimensional, AR 20, 64×16 | C_L = 0.5109 (LL 0.540, the AR-sweep deficit); midspan Cp_min = −1.86 against the target −1.9 ± 0.2; C_M about the quarter chord −0.002 |
| induced drag, Trefftz plane | C_Di = 0.00647 (plane at 1.0 b) against 0.00638 (0.8 b); the two-dimensional mode gives 0.00665, within 3%. C_Di/(C_L²/(πAR)) = 0.94, 0.89, 0.91 at AR 4, 8, 20 — BELOW the elliptic unity the design anticipated: the van Garrel core (rc = 0.2 spanwise spacings) excludes near-axis crossflow energy, so the Trefftz drag is biased low by O(5–10%) and its core sensitivity must be stated whenever it is quoted |
| α linearity, 2–8°, AR 8 | slope spread 0.34% |
| loading symmetry | 8×10⁻¹¹ (thin code 4×10⁻¹⁶; the difference is cond(A) amplification). This gate is what exposed the split-covariance bug: a single fixed diagonal split broke symmetry at O(10⁻¹) |
| free against frozen wake, AR 8 | C_L = 0.4267 against 0.4327: the free wake LOWERS lift by 0.0060, unlike the thin code (+0.0009), whose per-station tip filaments the thick code deliberately lacks |
| thickened sample wing (taper, sweep, twist, camber, f/c = 12%), uniform | C_L = 0.5312 against thin 0.4848: ratio 1.096 (LL-damped thickness factor 1.073); Trefftz C_Di = 0.01039 against the thin 0.00877 scaled by the lift ratio squared, 0.01053 — within 1.3%; loading symmetric to 8×10⁻¹¹ (the swept-grid dual-basis trap passes) |
| same wing, linear shear | C_L,shear/C_L,uniform = 0.9705 against the thin code's 0.9550; the gap is the streamline-height Bernoulli tagging across the thickness (upper and lower panels sit at different z, an O(U′f) convention difference from the single-surface thin code) |
| fresh clone from GitHub | reproduces the full test suite and the documented quick start, so the pushed copy is self-contained despite carrying the shared kernels as a copy |

Diagnosis guidance, updated with what implementation actually found: sign errors produce sign-flipped or grossly wrong lift and are caught by the sphere anchor — every sign in the code was pinned by the kernel unit checks against brute-force quadrature before the first solve. A loading asymmetry at O(10⁻¹) means a mirror-covariance defect (the fixed-diagonal split was one); |Cp| of O(10³) at the LE rows of thin sections means centroid-chord collapse across the nose fold (cured by the arc floor in the gradient spacing); a resolution-independent O(5%) gradient error on swept wings means the dual basis is using chord directions instead of the stencils' effective directions.

## Unit checks that pin every sign

1. Closed constant-μ surface: potential −μ inside, zero outside; row sums of the body doublet block equal −1. Catches the normal orientation, the solid-angle sign and the −1/2 diagonal.
2. Unit source panel: normal velocity ±σ/2 across the faces via Ω/4π; in-plane velocity outward at the edges; far field matching a point source to O((d/r)²).
3. Doublet panel against its ring: the central-difference gradient of the panel potential matches `segment_velocities` on the reversed-order perimeter at off-surface points.
4. TE cancellation: with Kutta-consistent μ, the velocity near the TE computed with and without the three dropped segments (equal cores) agrees to round-off.
5. Sphere (verification table): validates the whole Dirichlet solve without a wake.

## Implementation plan

The implementation is specified block by block in ARCHITECTURE.md, which owns the state contract, the public function signatures, the test layout and the build order. The architecture was designed so that three planned extensions — time dependence, partitioned coupling with a finite element structural code, and ventilation and cavitation models — fit later without structural rewrite, and it was verified by writing all three future drivers on paper against the public signatures alone.

Summary of the decisions recorded there: a single module `thick_panel_wing.py` with disciplined sections and an explicit state dict (importable with no prompts, prints or figures on the library path), plus `make_thick_sample_inputs.py`, `test_thick_panel_wing.py` (standard-library unittest; pytest is absent from the pinned environment) and the one-line guard in `panel_wing.load_mesh`. The shared kernels and input helpers are imported from `panel_wing` (`segment_velocities`, `induced_velocity`, `load_velocity_profile`, `make_onset`, `pitch_mesh`, `initial_nodes`, `prompt`); `filament_segments` is mirrored rather than imported because the wake becomes arrays, and the relaxation step is re-implemented because advection gains the source-panel velocity. Assembly accepts per-panel unknown-type masks (a pass-through in version one, the μ/σ swap for cavitation later), the boundary condition consumes a per-panel relative-velocity array (body motion later), and the wake influence returns a bound/frozen row split (time stepping later).

Build order, each step gated by its check: (1) geometry, validation and the topology–metric split; (2) kernels with unit checks 1–3 and the source edge clamp; (3) state API, frozen-wake solve and the sphere; (4) Kutta folding and unit check 4, frozen-wake wing; (5) free-wake relaxation and the wake-sharing cases; (6) loads and the Trefftz drag; (7) plotting and the command-line interface; (8) the verification table, updating this document with actuals, HANDOVER-style.

## Design decisions and their reasons

- Internal Dirichlet over Neumann: pointwise sources admit shear; second-kind conditioning; the wake needs only potential columns folded into two TE columns.
- Wrap mesh with an explicit format marker: exact diagnostics and versioning; geometric detection misdiagnoses defects.
- Pinched-wedge tips over cap fans: watertight, structured, no sliver clusters; the conditioning risk is quantified and O(1)-safe at the default resolutions.
- Collocation at the surface centroid with an analytic −1/2 diagonal: a finite inward offset contaminates neighbours and can cross the TE wedge.
- Ring velocities and solid-angle potentials on one node set with a deterministic, iteration-fixed triangulation: the discrete wake seen by both kernels is a single geometry.
- Trefftz drag beside pressure drag: near-field drag at this order measures truncation error, not physics.
- Trailing-edge shedding only in version one: the closed surface removes the thin code's singular side edge; tip separation returns as future work with the same strip machinery.
- Blocks with an explicit state API rather than one monolithic solve: the planned extensions (unsteady flow, structural coupling, cavitation and ventilation) each reduce to a new driver over the same block calls, and every extension hook in version one is a default argument or a dict field with zero cost (ARCHITECTURE.md).

## Open items

- The iterative pressure-Kutta condition for swept TEs, once the Morino condition demonstrably limits accuracy.
- A side-edge Kutta condition for chordwise tip separation (expected O(1–2%) lift effect at moderate aspect ratio).
- The wake-sweep contingencies (TE stub, exclusion shell) are designed but not enabled by default; enable on evidence.
- Blunt TEs are rejected, not modelled; a base-pressure model would be a separate development.
- Under shear the Trefftz theorem and strip-constant μ_w are O(U′·δz) approximations; no better convention is proposed here.
- The licence, requirements and command-line-argument items of HANDOVER.md apply unchanged to the new files.

# Ventilation

This document owns the ventilation formulation, every convention that pins a sign
or a factor, the free-surface image, the cavity boundary-value problem, the
regime state machine, and the verification results with measured numbers.
`docs/FEM.md` owns the structural formulation, `docs/COUPLING.md` the interface
operators and the partitioned schemes, `docs/ARCHITECTURE.md` the block
boundaries and the public signatures, and `docs/FLUID.md` the provenance of the
vendored panel method. Symbols: `z'` depth below the free surface, `h` immersion,
`c` chord, `AR_h = h/c`, `Fn_h = u/sqrt(g h)`, `L = Lc/c`, `Psi = sigma_c/(2
alpha_2D)`, `Phi` the cavity closure angle from the horizontal.

Reference: C. M. Harwood, Y. L. Young & S. L. Ceccio, *Ventilated cavities on a
surface-piercing hydrofoil at moderate Froude numbers: cavity formation,
elimination and stability*, J. Fluid Mech. **800** (2016) 5-56. Equation numbers
in parentheses are that paper's.

`docs/paper/ventilation.tex` is a manuscript derived from this document and
from `verify_vent.py`; it holds no number of its own.

Last updated 17 August 2026.

## What is modelled, and what is not

Atmospheric ventilation of a surface-piercing strut: air entrained from the free
surface into separated, sub-atmospheric flow on the suction side, forming a gas
cavity that removes most of the suction and with it most of the lift, and the
hysteresis between the wetted and ventilated states. Four things are delivered -
the inception of a cavity, its effect on the loads, its elimination by washout,
and the consequences for the coupled fluid-structure response.

Modelled: the free surface as an exact linearised high-Froude image; the cavity's
extent, its thickness and its displacement effect on the circulation; the
depth-dependent cavity pressure; the re-entrant-jet stability criterion; the
three flow regimes and the four transitions between them, with hysteresis; a
finite cavity growth rate; and the load path to the structure.

Not modelled, and each is a real limitation rather than an approximation of a
small term:

- **The free surface does not deform.** The image is the infinite-Froude
  linearisation on the undisturbed plane, which is also what Harwood's own
  lifting-line model uses. The steep free-surface depression his photographs show
  at moderate `Fn_h` is absent. `history['drift_y']` measures how far outside
  that assumption a run is; see "The wake cannot be convected" below.
- **There is no boundary layer,** so separation is an assumption. The stall angle
  and the pressure-recovery fraction are INPUTS.
- **There is no surface tension.** The Weber number is computed and warned on,
  never acted on.
- **There is no vaporous cavitation** unless `dsigma` is set, and no gas equation
  of state: the cavity pressure is prescribed, not solved from a mass of air.
- **Hydrostatic pressure is excluded from the load by construction.** See the
  reference-pressure convention below. Buoyancy and weight are body forces for a
  caller to apply, and a static deflection compared with an experiment is missing
  both.
- **Base ventilation off a blunt trailing edge, and the aerated-tip-vortex
  inception route,** are out of scope. The immersed tip of a wrap mesh is pinched,
  so it is a rounded tip and not the paper's blunt one.

## Conventions that pin a sign or a factor

**The reference pressure is the LOCAL still-water pressure**, `P_atm + rho g z'`.
Against that reference a wetted panel carries the dynamic pressure alone - which
is exactly what the vendored `pressure_fields` computes with `p_ref=None` - and a
ventilated panel carries `Cp = -sigma_c(z')`. Two consequences:

- `p_ref` is **rejected**, not merely left `None`, when ventilating: it would
  count the hydrostatic field twice. It also only ever receives the `z` column of
  the centroids, which for a vertical span is the lift axis and not depth.
- the still-water part integrates to the buoyancy of the immersed volume and is
  absent from the fluid load. This is the choice that leaves every pre-existing
  number in the repository bit-for-bit unchanged.

**The depth axis is `y`, which is also the span axis.** The strut's span is
vertical, so a spanwise strip IS a depth station. Every vendored spanwise routine
therefore remains valid and correctly interpreted: `spanwise_loading` returns the
depth loading, and `integrate_loads`' `CM` about `y` is already the **yawing
moment** of the paper's (3.3c). `x` is chordwise and `z` is the lift direction.

**`Phi` is the angle of the cavity closure line from the HORIZONTAL**, that is
from the flow direction: `Phi = atan|dz'/dx_closure|`. Equation (3.2) fixes it -
the jet leaves at `2 Phi`, so `Phi > 45 deg` makes `cos 2 Phi` negative and the
jet turns upstream. The limiting case settles it: a cavity of **uniform** length
has `dx_closure/dz' = 0`, hence `Phi = 90 deg` and a jet directed straight
upstream, which is the classical two-dimensional re-entrant jet and the canonical
unstable one. Spanwise taper is what makes a ventilated cavity stable, by sweeping
the jet sideways, and the paper's fully ventilated photographs show exactly that
taper. The opposite reading, `Phi` from the vertical, inverts the criterion, and
`test_vent.TestClosureAngle` excludes it with the uniform-cavity case rather than
with a comment.

**The cavity pressure is a FLOOR, not a replacement**: `cp = cp_solved + w *
max(cp_cav - cp_solved, 0)` with `w` the fractional coverage. Inside a cavity the
pressure cannot fall below the cavity's own, which is what a cavity is; writing it
as a floor makes the ventilated pressure never lower than the wetted one anywhere,
by construction rather than by tolerance. A cavity **raises** the suction-side
pressure - that is precisely why ventilation destroys lift - and the opposite
reading would still reduce `|CL|` on some meshes, so the direction is asserted
with no tolerance.

**The effective aspect ratio of a surface-piercing strut is the IMMERSED one,**
`AR_h = h/c`, not twice it. The negative image makes the waterline behave as a
tip, so the immersed span is the whole span of the equivalent wing. Passing
`2 AR_h` is the rigid-wall image and moves the lift slope by tens of per cent
while looking entirely plausible; V3 prints both so the wrong one is visibly far
away.

## The sectional model

`sigma_c(z') = dsigma + (z'/h)(2/Fn_h^2)` from (1.10) and (1.12), with
`dsigma = (P_atm - P_c)/(rho u^2/2)`, zero for natural atmospheric ventilation.
A positive `dsigma` makes the cavity vaporous or pressurised and every relation
still holds; only the air path is specific to atmospheric ventilation.

Cavity length from `Psi`: Acosta (1955) exactly for `L <= 0.5`, Tulin (1953)
exactly for `L >= 1.25`, joined by a monotone C1 cubic Hermite in
`(log Psi, log L)` across the gap where both classical solutions run into the
hodograph singularity at `L = 1` and are unphysical. The paper's own fits (1.7)
and (1.8) are available; (1.8) is required to reproduce the washout derivation.

Lift slope (1.9), centre of pressure (3.4), jet speed and direction (3.1)-(3.2),
Helmbold (1.20), the elliptic shape (4.7) and the lift-weighted slope (4.8) are
all implemented in `vent_section`, which imports nothing from this repository so
that it is the fixed reference the panel-level model is checked against.

The constant `1/(2 pi)` in (1.9)'s denominator is not legible in the paper's text
layer. It was recovered two independent ways: from the two analytic limits
`a0(0) = 2 pi` and `a0(inf) = pi/2`, and by checking that it makes the paper's own
(4.3) algebraically identical to `Cl_2D/a0`. The plausible misreading `2 pi` gives
`a0(0) = 1` with an otherwise smooth curve, so the limits are what `test_vent`
pins.

## The free surface

`tpw.validate_mesh` and `tpw.update_points` both force the two span-end stations
to be pinched - `update_points` re-pins unconditionally on every call - so an open
section at the waterline is impossible. The free surface is therefore an exact
**negative image realised by mesh doubling**: the immersed strut plus its mirror
as one legal closed `thick-wrap-1` body, with the free surface at mid-span and a
FULL-CHORD section there, which is where the cavity is longest and where a single
pinched foil of span `h` would have had no chord at all.

No new geometry is written. `mts.thick_wing_mesh` with unit taper and no sweep or
twist IS the doubled strut, and the yaw rotation is about the span axis and does
not touch it. The symmetry is then **enforced** rather than inherited, because
`np.linspace(-a, a, n+1)` is bit-exactly antisymmetric only when `n` is a power of
two - it is not at `n = 12, 20, 24` - so the last bits of the spanwise coordinate
are overwritten from the immersed half. That costs nothing and makes the property
hold for every station count and spanwise spacing.

Antisymmetry is imposed by overriding the source strengths on the image half,
`sigma_image = -mirror(sigma_real)`, through `assemble_system`'s `sigma_fixed`.
Then `phi` is antisymmetric, so `phi = 0` on the plane `y = y_fs`, which is the
linearised high-Froude free-surface condition. `mu_image = -mirror(mu_real)` is
**not** imposed - it emerges from the solve, and that it does is the test that the
image is right. Flipping the sign gives the zero-Froude rigid wall.

> **On a doubled mesh the vendored solve IS the rigid-wall answer.** For an onset
> with no spanwise component, mirroring leaves `-n.u_rel` unchanged, so the
> unflipped source strengths are the positive image, and `tpw.solve` on a
> ventilating state returns the zero-Froude solution silently and plausibly, with
> about twice the lift and its peak loading at the waterline instead of zero
> there. `test_vent` asserts the bit-for-bit agreement so the hazard is on the
> record rather than latent.

### The wake cannot be convected, and the drift it loses is the wave elevation

With `phi` antisymmetric the perturbation velocity obeys `v'(Rx) = -R v'(x)` while
the onset obeys `v(Rx) = +R v(x)`, so a mirror-consistent convection would require
`v' = 0`. Left to `convect`, the image wake drifts off the mirror position and the
free-surface condition is lost silently and progressively over a march. The image
wake is a mathematical device and must be **placed**, not convected:
`vent_mesh.symmetrise_wake` projects it onto the mirror-symmetric set after every
`shed`/`truncate`, with the immersed half authoritative, and sets the waterline
column's depth to the plane exactly.

Both operations have a physical reading, and the second earns a diagnostic. On the
plane the streamwise and lift components of the perturbation velocity are
antisymmetric and therefore vanish, while the depthwise component does not, so the
perturbation velocity on the free surface is purely normal to it: **the discarded
drift is exactly the linearised wave elevation.** It is reported as
`vent['drift_y']` beside `transfer['offset_max']`, and it is the quantitative
measure of how far below the model's valid Froude range a run is being pushed.

## The cavity

**Extent.** Parameterised by a continuous per-station length with a **fractional
weight on the closure panel**, never by a boolean panel mask. A boolean mask makes
the load piecewise-constant in the displacement: the cavity chatters between two
adjacent panels, the coupling residual is discontinuous at the panel scale,
IQN-ILS fits its least-squares system to a staircase, and the coupling cannot
converge below one panel width at any tolerance.

Detachment is **found**, not fixed at `xi = 0`: the first suction panel of a yawed
section straddles the stagnation point and carries a positive pressure
coefficient, so a scan anchored at the leading edge aborts at once and reports no
cavity at any incidence.

**Pressure.** On a cavity panel the dynamic condition fixes the pressure, hence
the total surface speed `q_c = u sqrt(1 + sigma_c)`, hence `mu` by integration
from detachment along the panel's own tangent with the arc-floored spacing the
vendored wrap stencil uses. Integrating along the chord instead puts the
leading-edge panels badly wrong, because there the surface tangent is nearly
normal to the chord.

**Thickness.** The unknown on a cavity panel is then the source strength, which is
the cavity's transpiration, and the thickness follows by chordwise integration.
This is what gives the cavity a displacement effect on the circulation rather than
merely a modified pressure. `closure="clip"` is retained as an auditable fallback
and is **not** self-consistent: clipping at fixed `mu` leaves the trailing-edge
jump, and hence the wake circulation and the induced field, at their wetted
values.

**Why the extent comes from the pressure and not from the thickness closing.** The
closure condition of a partial cavity is kinematic - the thickness returns to zero
at the closure point - and `extent_rule="thickness"` implements it with a secant.
But with the Dirichlet closure the dynamic condition makes the *computed* pressure
equal the cavity pressure everywhere on the cavity, so a pressure margin read off
the solved field is identically zero there; and read off the baseline instead, the
pressure condition is a contraction that converges in a few iterations while the
thickness secant limit-cycles at one panel width on this discretisation. The
default is therefore `extent_rule="pressure"` on the BASELINE wetted pressure -
the cavity occupies the run over which the outer flow would otherwise be below the
cavity pressure - with the thickness residual reported. Reading the extent off the
cavity's own imposed pressure collapses the cavity every iteration, and did.

**Growth rate.** The committed cavity front is rate-limited to `GROWTH_CHORDS`
chords of cavity per chord of travel, and the bound applies to the cavity the LOAD
sees, not merely to the next step's starting point. The paper's time histories
show formation taking a finite time - of order a tenth of a second on a 0.1 m
chord at a few metres per second - because the cavity grows by propagating
separation ahead of itself. Without the limit the extent jumps from nothing to
full coverage in one step, and the structural response then measures the time step
rather than the flow: peak tip displacement fell from 3.0e-2 chords to 3.0e-3 when
the limit was applied to the same case.

## Inception, and the regime machine

Three things must coincide, as the paper requires: a sub-cavity pressure, a
connected path to the air, and a broken free-surface seal.

- **Ventilation-ready** is `Cp < -sigma_c`, panel by panel.
- **The air path** is a breadth-first fill over the vendored face adjacency from
  the waterline strip. A sub-cavity-pressure pocket with no connected path is a
  cavitation site, not a ventilated one, and is refused - that is the whole
  difference between the two phenomena.
- **The seal** breaks at the stall angle, or on an injection. At sub-stall
  incidence the paper's oil-film visualisations show the separation bubble
  stopping short of the free surface, leaving a thin attached layer that seals the
  ventilation-prone flow from the air; a panel method cannot resolve that layer,
  so the stall angle is an input and it is exactly the vertical stall boundary of
  the paper's regime map.

Separation gates **inception**, not each panel of an existing cavity. The paper is
explicit that entrained air modifies the local pressure gradients and so
propagates separation ahead of itself; gating every panel on the wetted separation
indicator forbids the cavity from ever reaching the leading edge, where the
paper's own photographs show it detaching.

Inception is decided on the criterion and not on a trial cavity, because a fully
wetted state does not compute one: while the regime is FW the cavity length is
held at zero, so a rule that waited for a cavity to appear could never fire.

**The regime advances only on `commit_vent`** - the same rule as the doublet
history and for the same reason, plus a stronger one: the regimes are bi-stable,
so a regime that flipped inside a coupling subiteration would not merely make the
load path-dependent, it would stop the load being a function of the displacement
at all. The ventilation regime is the **third** thing that becomes a time level in
`time_march_fsi`, beside the structural level and the doublet strengths.

**The hysteresis is structural, not a tolerance band.** Inception needs an air path
and a broken seal; persistence needs only that ventilation-ready flow remains. A
cavity therefore survives to incidences well below the one that formed it, which
is the paper's own mechanism and the reason the regions of its figure 16 overlap.
A single threshold with a dead band would reproduce a sweep and fail
`test_vent.TestInception`.

## Verification

Every number below is measured by `python3 verify_vent.py`. V1, V2 and V5 compare
against closed forms, exact symmetries and an equation re-derived here, so their
agreement is verification. V3, V4 and V6 compare against a towing-tank experiment
through a low-order model, so their trends are the verified statement and their
magnitudes are reported, not verified.

### V1 the sectional model against the closed forms

| L | Psi (Acosta) | L from (1.7) | error in L | error in Psi |
| --- | --- | --- | --- | --- |
| 0.05 | 17.89 | 0.033 | -33.5% | +22.6% |
| 0.10 | 12.66 | 0.086 | -14.0% | +7.8% |
| 0.20 | 8.97 | 0.203 | +1.3% | +0.6% |
| 0.30 | 7.36 | 0.307 | +2.3% | +1.1% |
| 0.40 | 6.43 | 0.391 | -2.2% | +1.0% |
| 0.50 | 5.83 | 0.456 | -8.7% | +4.0% |

The comparison that means anything is in `Psi`: `dPsi/dL` is large near the branch
join, so the 8.7% error in `L` at `L = 0.5` is about 1% in `Psi`.

`a0` from (1.9) against Acosta: within 1.5% over the whole partial branch, 0.11%
at `L = 0.3`. `a0(0) = 6.283185307180` against `2 pi = 6.283185307180`, and
`a0(inf) = 1.570796326795` against `pi/2 = 1.570796326795` - both exact. The fit
peaks at `a0 = 7.2022` at `L = 0.511`, above `2 pi`, which is the local maximum of
the paper's figure 2(b); it then **undershoots its own limit by 0.93% near
`L = 10`** and approaches `pi/2` from below, so it is not monotone in the tail.
That belongs to the rational polynomial, not to the physics.

On the supercavity branch (1.7) holds to about 5% for `1.25 <= L <= 2.6` and
degrades to -63% at `L = 23`, where (1.8) is the better fit. (1.8) is **not** an
approximation to Tulin: Tulin grows as `sigma^-2` and (1.8) as `sigma^-1`, so
their ratio diverges. `e(0) = 0.248876`, 0.45% below the quarter chord because
(3.4) is a tanh blend; `e(0.5) = 7/32` and `e(inf) = 3/16` exactly.

### V2 exactness

At `alpha = 14 deg`, `Fn_h = 2.5`, 160 immersed panels:

| quantity | error |
| --- | --- |
| mirror symmetry of the doubled strut | 0.00e+00 |
| image doublet strength + mirror of the real one | 2.4e-14 |
| image source strength + mirror of the real one | 6.6e-15 |
| image wake strength + mirror of the real one | 2.7e-14 |
| `phi` on the free-surface plane, relative | 1.7e-16 |
| unflipped image minus the vendored solve | 0.00e+00 |
| total force through the lumping | 1.5e-16 |
| total force through the transfer | 1.1e-16 |
| total moment through the lumping | 9.3e-17 |
| virtual work across the interface | 0.00e+00 |
| largest pressure the cavity lowered | 0.00e+00 |

`phi = 0` on the plane holds to round-off rather than to the `1e-10` of the
vendored mirror covariance, because the mesh is bit-exactly symmetric and the
antisymmetry is exact, so the cancellation is pairwise.

Immersed-half `CL = +0.17428` on `s_ref = h c = 1.000`; the whole doubled-mesh
`CL = +0.00407`, a near-cancellation and **not** the answer.

### V3 the wetted surface-piercing baseline

At `alpha = 10 deg`, `Fn_h = 2.5`, `AR_h = 1`:

| strips per half | waterline cl / peak | CL | CL/sin(alpha) |
| --- | --- | --- | --- |
| 4 | 0.6030 | +0.2552 | 1.4698 |
| 8 | 0.4648 | +0.2382 | 1.3716 |
| 16 | 0.4232 | +0.2336 | 1.3453 |

The loading at the waterline falls as the mesh refines but is `O(dy)`, not zero:
`phi = 0` on the plane is exact while the panel loading vanishes only in the
limit, because with an even span count no panel straddles the plane and the
shallowest strip's circulation is `O(dy)`.

Measured lift slope 1.3453 against Helmbold at `AR_h = 1` of 1.4947, a ratio of
0.900; against `2 AR_h` it is 0.506, printed only to show how far away the
rigid-wall reading is.

### V4 the load effect, against figures 12 to 14

At `Fn_h = 2.5`, `AR_h = 1`:

| alpha | CL wet | CL vent | ratio | e wet | e vent | Lc max | D/h | Phi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | +0.1391 | +0.0757 | 0.544 | 0.342 | 0.084 | 0.904 | 0.94 | 51.6 |
| 10 | +0.2300 | +0.1322 | 0.575 | 0.339 | 0.144 | 0.976 | 0.94 | 50.4 |
| 14 | +0.3183 | +0.1779 | 0.559 | 0.335 | 0.181 | 1.000 | 0.94 | 49.9 |
| 18 | +0.4029 | +0.2191 | 0.544 | 0.329 | 0.186 | 1.000 | 0.94 | 49.9 |

The lift ratio is about 0.55 throughout, a loss of some 45%. Breslin & Skalak
report losses of up to 70% and the paper's figure 12 shows the same ordering, so
the sign, the ordering and the magnitude to within a factor are what this
comparison supports; the mechanism this model lacks is the cavity closure region,
where the experiment's re-entrant jet removes suction over an area a
prescribed-pressure cavity cannot place. **Reported, not verified.**

The centre of pressure moves from about 0.33 to about 0.19 of the chord forward of
mid-chord. The paper's figure 14 quotes 1/4 wetted and 3/16 = 0.1875
supercavitating; **the ventilated value lands on 3/16**, and the wetted value sits
further forward than a thin foil's quarter chord because the loading of a
low-aspect-ratio surface-piercing strut is concentrated forward.

The cavity is longest at the waterline and shortest at the tip - at `alpha = 14`,
`Fn_h = 2.5` the per-station lengths run 0.17, 0.35, 0.43, 0.48, 0.52, 0.55, 0.65,
1.00 from tip to surface - which is the planform of the paper's figure 3, and it
follows from `sigma_c` growing with depth: hydrostatic pressure arrests the cavity
at depth and does not at the surface. `Phi_bar` lands at 50 deg against the
paper's measured 40.75 deg at washout.

### V5 the washout boundary

(4.5) against the chain (4.1)-(4.4) solved directly, over `AR_h` in 0.5 to 3 and
`CL` in 0.2 to 0.8: largest relative difference **2.1e-16**. This verifies the
transcription and the algebra of (4.5), and with it (1.9), (4.1), (4.2), (4.3) and
(4.4), because the chain uses all of them. It does **not** verify the physics; the
wording to use is that (4.5) is verified against the relations it was derived
from. Sample value `AR_h = 1`, `CL = 0.5`: `Fn_h = 1.1064`, against the paper's
1.1062.

Breslin & Skalak's boundary bounds it from above everywhere, by a factor 2.86 at
`AR_h = 1, CL = 0.5`, which is the paper's own finding. Their boundary is
`max(sqrt(5/CL), 3)`; the two conditions cross at `CL = 5/9` and a code applying
only the first is wrong above that.

### V6 hysteresis and the coupled transition

At `Fn_h = 2.5`, `AR_h = 1`, stall angle 14.5 deg:

| alpha | from wetted | from ventilated | bi-stable |
| --- | --- | --- | --- |
| 6 | FW, CL = +0.14403 | PV, CL = +0.07794 | yes |
| 10 | FW, CL = +0.23801 | PV, CL = +0.13413 | yes |
| 13 | FW, CL = +0.30657 | PV, CL = +0.16505 | yes |
| 16 | PV, CL = +0.37296 | PV, CL = +0.19743 | no |
| 20 | PV, CL = +0.45732 | PV, CL = +0.23459 | no |

The conditions are identical along each row and only the history differs. Above
the stall angle the flow incepts from either start, so the bi-stable band runs
from the bifurcation angle up to stall, which is the paper's figure 16.

The coupled march at `alpha = 20 deg`, `dt = 0.01`, `rho_inf = 0.8`: dry
frequencies 6.62, 16.24, 37.36 Hz, frozen-cavity added-mass ratio 4.05, 15.1 steps
per period. Inception is recorded at the first commit, the cavity front then grows
at 0.01 chords per step under the rate limit, strong coupling converges in 9 to 14
subiterations per step, and the peak tip displacement is 3.0e-3 chords.

The energy balance residual during the transition is order one, and is meant to
be: C6's balance closes because the only work done on the structure is by the
fluid, and here the cavity is growing. Entrained air displaces water and does work
this model does not account for - there is no gas equation of state and no
entrainment energy - so the balance cannot close during a transition. It is
reported as a measure of how far the step is from a conservative one, not as a
verification.

## Expected results that are not errors

- **The doubled mesh's `CL` is a near-cancellation, not the answer.** The image
  half carries the negated loading. Resultants come from the immersed half with
  `s_ref = h c`, and `tpw.get_loads` must never be called on a ventilating state.
  Scaling its small number by two is a real bug wearing a plausible face.
- **The depth loading at the waterline is `O(dy)`, not zero.** `phi = 0` on the
  plane is the exact statement; the vanishing of the panel loading is its
  discretisation.
- **Equation (1.7) is 8.7% low in `L` at `L = 0.5` while 1% right in `Psi`.** The
  comparison is made in `Psi`.
- **Equation (1.8) is not an approximation to Tulin,** and (1.7) saturates at
  `L = 100.54` as `Psi -> 0` because its denominator has no positive root.
- **The lift slope (1.9) undershoots `pi/2` by 0.93% near `L = 10`** and
  approaches its limit from below, so it is not monotone in the tail.
- **`e(0) = 0.2489`, not exactly a quarter chord.** (3.4) is a tanh blend.
- **A ventilated strip's lift does not vanish.** The cavity and the wetted
  pressure side together present a cambered surface, exactly as the paper notes.
- **The drag ratio is near one, and that is a coincidence of two omissions.** This
  model has neither the increased profile drag of the cavity nor its spray drag,
  and no friction at all, so the near-continuity the paper observes is not the
  same statement here.
- **The regime can differ between two runs at identical conditions.** That is
  hysteresis working. It also means a ventilated case is reproducible only together
  with its history, which is why every verification case builds its own `vent`
  state and never reuses one.
- **The energy balance does not close while the cavity is growing.** See V6.
- **A ventilation transition is a load step, so the second resolution constraint
  binds hardest there.** Expect the `steps_per_period` warning and real overshoot,
  and start from the static equilibrium: a march begun from a non-equilibrium
  state at these added-mass ratios can distort the pinched immersed tip enough for
  the vendored bowtie check to fire on the first predictor.
- **The stall angle, the pressure-recovery fraction, the inception area fraction
  and the Weber threshold are INPUTS.** Every case that depends on one prints it
  as an input.
- All the expected results of `docs/COUPLING.md` and the vendored documents still
  apply.

## Open items

- The dynamic condition is satisfied in the cavity interior to about 3e-2 in `Cp`
  on the meshes used here - a measured consistency, not a tolerance. The residual
  is the non-orthogonality of the vendored dual basis: the prescribed doublet
  strength is integrated along the wrap while the reconstructed velocity mixes
  wrap and span components. The two edge panels of the cavity are out by `O(0.2)`
  because their gradient stencil reaches into wetted flow, which is a property of
  a surface-gradient pressure evaluation and does not reach the load.
- `extent_rule="thickness"` limit-cycles at about one panel width on this
  discretisation and is not the default. A fractional closure panel in the
  *system* mask, rather than only in the pressure, would remove the remaining
  discreteness.
- **A cavity growing during a marched computation produces an oscillation in `CL`
  whose amplitude is set by the growth rate, and on a rigid structure the
  mechanism is not identified.** Two parts, and only the second is open.
  *Physical:* on a compliant strut the load step of a transition excites the WET
  fundamental, and the ringing does not decay because the model carries no
  structural damping while the generalised-alpha dissipation acts on the high
  modes. Measured at `alpha = 10 deg`, `Fn_h = 2.5`, `dt = 0.02`, `nwake = 24`,
  `E = 2e8`: the standard deviation of `CL` over a 70-step march falls from 0.329
  at `growth_chords = 0.2` to 0.030 at 0.05, the period is 0.7 in `s`, and
  `steps_per_period` reports seventeen steps per wet period, so the response is
  resolved. Resolve the front and the ringing shrinks with it.
  *Open:* the same oscillation survives making the structure rigid. At
  `E = 2e12` the tip deflection is 6.5e-7 chords, so there is no structural
  response, yet at `growth_chords = 1.0` `CL` ranges over -1.50 to +0.55. Four
  causes are excluded by measurement. Not the added-mass term: a quasi-steady
  evaluation, which has no `dphi/dt`, oscillates identically (standard deviation
  0.249 against 0.232). Not structural ringing, by the rigid case above. Not the
  steady load's dependence on the cavity length: at frozen extents `CL(L)` falls
  smoothly from 0.380 to 0.211 over `0 <= L <= 0.6`. Not the marched wetted path:
  the same case below stall is smooth to 5.7e-3. The remaining suspect is the
  unsteady pressure of the cavity's own prescribed doublet strength on a strip
  whose cavity has reached the trailing edge, which would make it a facet of the
  wake-borne cavity item below. Until it is resolved, marched transients are
  reported for the growth of the cavity, the depth it reaches, and structural
  responses at a resolved growth rate - never for the detailed time history of the
  load at a fast one.
- The cavity that leaves the trailing edge closes in the wake, and the wake sheet
  is then the cavity's continuation. That case is treated by prescribing the
  cavity condition over the whole suction side of the strip and taking the closure
  line from the sectional model; a wake-borne cavity surface is future work.
- No free-surface deformation, no surface tension, no gas equation of state, no
  vaporous-to-ventilated transition, and no base or tip-vortex inception route.
- A Woodbury update would remove the inner dense solve as the cavity mask changes,
  and the doubled system could be folded analytically to half its size with the
  doubled solve as its verification reference. Neither is built.

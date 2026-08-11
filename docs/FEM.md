# The structural solver: formulation, conventions and verification

This document owns the structural formulation, every convention that pins a
sign or a factor, and the verification results with their measured numbers. The
coupling is in [COUPLING.md](COUPLING.md), the block boundaries and public
signatures in [ARCHITECTURE.md](ARCHITECTURE.md), and the fluid formulation in
the upstream repository recorded in [FLUID.md](FLUID.md).

The tables below hold what `verify_fem.py` measured on the machine this was
written on. They are actuals, not targets: where the code disagrees with a
classical reference, the disagreement is tabulated and explained rather than
tuned away.

Last updated 11 August 2026.

## Why a three-dimensional continuum, and not a beam

The requirement that fixed the whole design was "isotropic elastic now,
anisotropic and inelastic later". That is a statement about **where the
constitutive law is evaluated**, and it excludes the model an aeroelastic code
would otherwise reach for first.

An equivalent beam carries sectional `EI`, `GJ` and mass per unit length. Those
are not a constitutive law: they are integrals of one over a cross-section,
taken under an assumed kinematics. A composite lay-up does not enter them as a
material, it enters as a re-derivation of the section; a viscoelastic skin does
not enter at all. Whatever the interface looks like, the extension is a rewrite.

A three-dimensional continuum element evaluates stress and tangent modulus at
each Gauss point through a material object. Anisotropy is then a different
`(6, 6)` tangent and a rotation of the material axes; inelasticity is a stress
update with internal variables. Neither touches the element, the assembly, the
solvers or the coupling. That is the whole argument, and
`fem_materials.OrthotropicElastic` and `fem_materials.KelvinVoigt` exist to
keep it from being merely an argument: they exercise the anisotropy path and
the rate-and-history path respectively, and both are gated below.

The cost of the choice is real and is stated: a continuum model of a thin
structure needs many more degrees of freedom than a beam, and low-order
hexahedra lock in bending unless something is done about it. Section
"The element" is what is done about it, and case F9 measures how much it
matters.

## Conventions

These bind every formula in this module, in the way the conventions section of
the upstream `DESIGN.md` binds the fluid.

- **Voigt order** `[e11, e22, e33, 2e12, 2e23, 2e31]` for strain and
  `[s11, s22, s33, s12, s23, s31]` for stress, with **engineering shear**, so
  that `sigma . epsilon` is twice the strain energy density with no stray
  factor and the shear rows of the tangent carry `G` rather than `2G`.
- **Element node order**: the `zeta = -1` face anticlockwise about `+zeta`
  (local nodes 0 to 3), then the `zeta = +1` face in the same in-plane order
  (4 to 7). A right-handed `(i, j, k)` grid then has a positive Jacobian, and a
  non-positive Jacobian at any Gauss point raises, naming the element - the
  counterpart of the fluid's signed-volume and bowtie rules.
- **Degrees of freedom** are indexed `3*node + component`; displacement,
  velocity and acceleration are `(Ns, 3)` arrays and are flattened only inside
  the linear-algebra blocks.
- **Material axes** are the COLUMNS of the `orientation` matrix, expressed in
  global coordinates, so a tensor transforms as `e_mat = R^T e_glob R`.
- Density is per unit volume, forces are consistent nodal forces, and nothing
  is normalised: the coupling is dimensional and the fluid delivers dimensional
  tractions.

The Voigt rotation operator is built by pushing the six Voigt basis strains
through the tensor transformation rather than by transcribing a Bond matrix.
That is deliberate: the engineering-shear factors then take care of themselves,
and factor-of-two errors in the shear rows are the classic way anisotropic
elasticity goes wrong silently. Case F6 gates it.

## The element

Eight-node trilinear hexahedron, `2 x 2 x 2` Gauss, isoparametric.

### Incompatible modes, and the correction that makes them legal

A trilinear hexahedron in bending has no way to represent the linear variation
of axial strain through the thickness except by shearing, so it locks: at the
resolutions a coupled run can afford it is an order of magnitude too stiff.
Case F9 measures exactly that.

Three Wilson bubbles `M_i = 1 - xi_i^2`, each with three components, add nine
internal degrees of freedom that carry the missing strain. They do no external
work, so they condense out exactly at element level:

    Ke = Kuu - Kua Kaa^-1 Kau

The bubbles as written fail the patch test on a distorted mesh, which is the
one thing an element may not do. The Taylor, Beresford and Wilson correction
fixes it: the incompatible gradient is evaluated with the Jacobian at the
element **centre** and scaled by `detJ0/detJ`, which makes the integral of the
bubble gradient over the element vanish, so a linear displacement field leaves
the amplitudes identically zero. Case F1 gates both statements at round-off.

The condensed amplitudes are resolved in `element_alpha`, a separate
element-level solve. That is where a nonlinear material's local Newton
iteration goes, and nowhere else.

### The internal force is always an integral

`internal_force` returns the integral of `B^T sigma` with the material in the
loop, never `K u`, even though the two agree exactly for the linear elastic
material of version one and a unit test asserts that they do to 1e-11. Writing
it the other way would put the constitutive law back inside the element.

## The materials

| class | parameters | what it is for |
| --- | --- | --- |
| `IsotropicElastic` | `E`, `nu`, `rho` | the verified default |
| `OrthotropicElastic` | nine constants, `rho` | the anisotropy path made real |
| `KelvinVoigt` | an elastic base and `tau` | the rate and history path made real |

`OrthotropicElastic` builds the compliance in the material frame and inverts
it, so the reciprocal relations hold by construction and only three Poisson
ratios are asked for; a set that gives an indefinite compliance raises rather
than producing negative strain energy in silence.

`KelvinVoigt` returns `tau*D` as its **rate tangent**, kept separate from the
tangent, so the assembly puts it in the damping matrix where it belongs. The
modal damping is then exactly stiffness-proportional and the analytical decay
gate is exact rather than first order in the step. Its viscous stress reaches
equilibrium through `internal_force(v=...)` like any other stress; the matrix
assembled from the rate tangent is used **only** to build the effective dynamic
operator, where it is a tangent. Adding it to the damping matrix as well would
count the viscous stress twice.

A plastic material is a fourth class: trial stress, yield check, return
mapping, consistent algorithmic tangent, with the plastic strain and hardening
variable written into `history`. Nothing above it changes.

## Solvers

**Statics** is a Newton loop on `r = f_ext - f_int(u)`, which exits after one
correction for a linear material. **Dynamics** is generalised-alpha in the same
residual form, so a nonlinear material simply takes more iterations.

`step_dynamic` is a **pure function** of the committed level and the new load;
`commit` is the only thing that advances the state. This is not tidiness: a
strongly coupled step re-solves the same time level several times with
different loads, and an integrator that has already advanced its own state
cannot be asked twice.

Both loops measure their residual against the **first** residual of the call
and stop on stagnation. A free-vibration step has no external force, and a
scale taken from it collapses to zero, which runs every step silently to the
iteration cap - a defect the free-vibration gate found. The attainable floor is
round-off in the integral of `B^T sigma`, near 1e-9 relative on a slender stiff
beam, so `TOL_NEWTON` is 1e-8 and a tighter target only spins.

Operators are **dense and factored once**, against a version stamp, the same
contract the fluid states for its own assembly and solve. scipy is deliberately
absent upstream and here, so the triangular solves are written out: one
vectorised dot product per row, about ten milliseconds at three thousand
degrees of freedom against ten seconds for the factorisation being reused.

## Verification

Every number below is from `python3 verify_fem.py`. Aluminium is
`E = 70 GPa`, `nu = 0.3`, `rho = 2700`; steel is `210 GPa`, `0.3`, `7800`. The
cantilever is `1.0 x 0.1 x 0.02 m`.

### F1 Patch test: constant strain on a distorted mesh

A 3x3x3 cube with its interior nodes randomly displaced, every boundary node
prescribed to an exact linear displacement field with all six strain components
nonzero.

| incompatible modes | displacement error | stress error |
| --- | --- | --- |
| on | 5.4e-16 | 2.0e-15 |
| off | 1.4e-16 | 1.1e-15 |

Both at round-off. Without the Taylor-Beresford-Wilson correction the bubbles
survive a linear field and this test is what detects it.

### F2 Rigid-body modes and the symmetry of K

On the same distorted mesh: the six rigid-body displacement fields give
`|K u| / |K| |u|` between 1.5e-16 and 4.8e-16; `|K - K^T| / |K|` is 6.0e-17;
and exactly six eigenvalues lie below 1e-9 of the largest.

### F3 Mass properties of a 2 x 3 x 4 box

| quantity | error |
| --- | --- |
| total mass | 0 |
| centre of mass | 2.2e-16 |
| principal moments of inertia | 3.0e-16 |
| off-diagonal terms | 2.8e-17 |

Exact, because the consistent mass of a parallelepiped mesh integrates a
quadratic exactly on the `2 x 2 x 2` rule.

### F4 Cantilever tip deflection

100 N distributed over the tip face. Euler-Bernoulli `PL^3/3EI` gives
-7.142857e-3 m; with the Timoshenko shear term, -7.145000e-3 m.

| elements along the length | deflection (m) | ratio to Timoshenko |
| --- | --- | --- |
| 4 | -6.522924e-03 | 0.9129 |
| 8 | -6.823105e-03 | 0.9549 |
| 16 | -6.981163e-03 | 0.9771 |
| 32 | -7.060798e-03 | 0.9882 |

Monotone from below and converging. The residual 1.2% at 32 elements is the
clamped end face, which restrains the Poisson contraction and the warping a
beam solution allows; it is a difference between the two models, not an error
in either.

### F5 Flapwise natural frequencies

24 elements. The flapwise modes are selected by their shape - a cantilever of
this section has an edgewise mode between the first and second flapwise ones,
and taking the first three eigenvalues would compare the wrong things.

| mode | finite element | Euler-Bernoulli | ratio |
| --- | --- | --- | --- |
| 1 | 16.6133 Hz | 16.4504 Hz | 1.0099 |
| 2 | 104.3813 Hz | 103.0931 Hz | 1.0125 |
| 3 | 294.2705 Hz | 288.6637 Hz | 1.0194 |

Above the beam reference, and by more in the higher modes, for the same reason
as F4 plus the rotary inertia and shear the beam reference omits.

### F6 Anisotropy

| check | error |
| --- | --- |
| orthotropic with equal constants reduces to isotropic | 2.4e-16 |
| isotropic modulus invariant under a rotation | 3.2e-16 |
| carbon-like modulus rotates as `D -> T^T D T` | 2.5e-16 |
| strain energy invariant under the rotation | 4.4e-16 |

### F7 Kelvin-Voigt decay

Three cycles at 128 steps per period, against `exp(-zeta omega t)` with
`zeta = tau omega / 2`.

| tau | damping ratio | measured | exact | ratio |
| --- | --- | --- | --- | --- |
| 1.0e-4 | 0.00527 | 0.905485 | 0.905457 | 1.00003 |
| 3.0e-4 | 0.01581 | 0.742389 | 0.742342 | 1.00006 |

### F8 Generalised-alpha

Period elongation on a single resolved mode, against the leading term
`(omega dt)^2 / 12`:

| steps per period | measured `T/T_exact - 1` | `(omega dt)^2/12` | amplitude after 5 cycles |
| --- | --- | --- | --- |
| 16 | +1.269562e-02 | 1.285105e-02 | 0.923136 |
| 32 | +3.203777e-03 | 3.212762e-03 | 0.994969 |
| 64 | +8.031328e-04 | 8.031905e-04 | 0.999683 |
| 128 | +2.008179e-04 | 2.007976e-04 | 0.999980 |

Four figures at every resolution, falling at second order.

Numerical dissipation, with the step sized for mode 1 at 32 steps per period.
Mode 9 is 68 times faster and therefore far beyond its own Nyquist limit:

| rho_inf | resolved mode 1 | unresolved mode 9 |
| --- | --- | --- |
| 1.0 | 0.994969 | 0.760343 |
| 0.9 | 0.994826 | 0.020421 |
| 0.8 | 0.994238 | 0.000001 |
| 0.5 | 0.984612 | 0.000000 |

The right-hand column is the point of the parameter: it leaves a resolved mode
alone and annihilates an unresolved one. A coupled march whose structural
spectrum reaches far above the step needs exactly that, for reasons
[COUPLING.md](COUPLING.md) sets out.

### F9 Shear locking

Tip deflection as a fraction of the Timoshenko value, with the incompatible
modes on and off:

| elements | with bubbles | without | ratio |
| --- | --- | --- | --- |
| 4 | 0.9129 | 0.0163 | 55.9 |
| 8 | 0.9549 | 0.0618 | 15.5 |
| 16 | 0.9771 | 0.2035 | 4.8 |
| 32 | 0.9882 | 0.4783 | 2.1 |

Without the bubbles the element is sixty times too stiff at the resolution a
coupled run can afford. This is the entire reason they are there.

## Expected results that are not errors

- **The cantilever is 1 to 2% stiffer than the beam solution and converges from
  below.** The clamped end face restrains Poisson contraction and warping; the
  beam solution allows both. The gap grows with mode number in F5 for the same
  reason.
- **A resolved mode is barely damped by `rho_inf`.** That is what `rho_inf` is
  designed to do. Its effect is on the unresolved modes, and F8's second table
  is where it shows.
- **`solve_static` reports a residual near 1e-9, not 1e-14.** The internal
  force is an integral of `B^T sigma` with large cancellations on a stiff
  slender structure. The displacement it returns agrees with a direct
  factorisation to twelve figures, which is the statement worth making.
- **The lofted solid foil has wedge-degenerate elements at its leading and
  trailing edges.** The vertical extent of an aerofoil section is zero there,
  so this is the geometry and not an artefact. Their Gauss-point Jacobians are
  checked like any other element's.

## Open items

- Kinematics are linear: small displacement and small strain. The residual and
  tangent form is in place throughout, so a total Lagrangian path is a change
  of strain measure plus a geometric stiffness, but it is not written.
- Only the hexahedron exists. Thin skins, membranes and sails want a shell
  element, which would share the material interface through its thickness
  integration points.
- The operators are dense. That bounds the model at a few thousand degrees of
  freedom before the factorisation dominates the fluid, which is a real limit
  on how finely a structure can be resolved.
- The materials are exercised but only the isotropic one is used in anger. A
  layered or fibre-reinforced case would be the natural next verification.

"""Sectional cavity physics for ventilated and cavitating hydrofoil sections.

The two-dimensional half of the ventilation model: the classical linearised
cavity solutions, the smooth fits Harwood, Young & Ceccio (2016) built from
them, the re-entrant-jet stability criterion, and the semi-theoretical washout
boundary. Everything here is a function of scalars or arrays of scalars.

Six dependency-ordered sections, the layout of fem_solid.py, and the same three
discipline rules: nothing imports downward, every block is a pure function over
explicit arguments with all mutable data in an explicit state dictionary, and
prompts, prints and figures appear only under main().

    1 constants     gravity, surface tension, the branch joins, 45 degrees
    2 imports       numpy, and nothing else from anywhere
    3 similarity    sigma_c(z'), Froude, Weber, the cavity parameter Psi
    4 length        Acosta, Tulin, the two Harwood fits, and the blend
    5 sectional     lift slope, centre of pressure
    6 stability     the re-entrant jet, the closure angle, the flow regime
    7 washout       Helmbold, the elliptic shape, the washout Froude number

This module imports nothing from this repository, deliberately. It is the only
part of the ventilation model with closed-form verification, so it is the fixed
reference the panel-level model is checked against, and a module with no
dependencies cannot drift with the thing it is checking. It never sees a panels
dictionary. `vent_liftline` imports this and nothing else for the same reason:
a reference implementation that shares code with the thing it verifies verifies
nothing.

Four decisions are worth stating before the code, because each looks like an
inefficiency or an oddity and none is.

**The reference pressure is the LOCAL still-water pressure, P_atm + rho g z'.**
Against that reference a wetted panel carries the dynamic pressure alone - which
is exactly what the panel method already computes with p_ref=None - and a
ventilated panel carries Cp = -sigma_c(z'). So the ventilation model is stated
entirely through sigma_c, the fully wetted baseline needs no change at all, and
no hydrostatic term is ever added to a pressure. The still-water part integrates
to the buoyancy of the immersed volume and is deliberately absent from the fluid
load, as are weight and gravity on the structure; both are body forces for a
caller to apply.

**Phi is the angle of the cavity closure line from the HORIZONTAL**, that is
from the flow direction, so Phi = atan|dz'/dx_closure|. Equation (3.2) fixes it:
the jet leaves at 2*Phi, so Phi > 45 degrees makes cos(2 Phi) negative and the
jet turns upstream, which is the paper's stability criterion. The limiting case
settles it - a cavity of uniform length has dx_closure/dz' = 0, hence Phi = 90
degrees and a jet directed straight upstream, which is the classical
two-dimensional re-entrant jet and the canonical unstable one. Spanwise taper is
what makes a ventilated cavity stable, by sweeping the jet sideways. The
opposite reading, Phi from the vertical, inverts the criterion, and
`test_vent.TestRegime` excludes it with the uniform-cavity case rather than with
a comment.

**Comparisons with Acosta are made in Psi, not in cavity length.** dPsi/dL is
large near the branch join, so at Acosta's L = 0.5 the fit (1.7) returns
L = 0.456, 8.7% low in L while about 1% right in Psi. A test written on L reads
as a failure of a fit that is correct.

**Equation (1.8) is not an approximation to Tulin.** Tulin's length grows as
sigma^-2 and (1.8) as sigma^-1, so their ratio diverges. (1.8) is a simplified
long-cavity relation whose only purpose is to make the washout boundary (4.5)
integrable in closed form, and it agrees with (1.7) to 5% only near L = 2. That
interval is inherited by everything (4.5) says.

Setting dsigma to something other than zero makes the cavity vaporous rather
than ventilated, and every relation here still holds; only the air path of
`vent_cavity` is specific to atmospheric ventilation.

Reference: C. M. Harwood, Y. L. Young & S. L. Ceccio, Ventilated cavities on a
surface-piercing hydrofoil at moderate Froude numbers: cavity formation,
elimination and stability, J. Fluid Mech. 800 (2016) 5-56. Equation numbers in
the docstrings below are that paper's.
"""

# ------------------------------------------------------------- 1 constants

import numpy as np

G = 9.81                       # m/s^2, gravity; the only place it is defaulted
GAMMA_ST = 0.072               # N/m, air-water surface tension at 20 C
RHO_WATER = 1000.0             # kg/m^3, for the Weber number's default

WE_MIN = 250.0                 # Wetzel (1957): below this the spray sheet closes
                               # and surface tension inhibits inception. There is
                               # no surface tension in this model, so the number
                               # is a gate that warns, never a term that acts.
PHI_CRIT = np.pi / 4.0         # 45 degrees, the closure-angle stability limit.
                               # The paper measures 40.75 degrees at washout over
                               # every incidence and immersed aspect ratio.
ALPHA_STALL = np.radians(14.5)  # 14 to 15 degrees for the paper's section, and
                               # an INPUT of this model: a panel method has no
                               # boundary layer and cannot predict separation.

L_PARTIAL = 0.5                # (1.6) holds to here; Acosta's own limit
L_SUPER = 1.25                 # (1.5) is fitted from here; the paper's choice.
                               # Between the two both classical solutions run
                               # into the hodograph singularity at L = 1 and are
                               # unphysical, so section 4 blends across the gap.
L_FIT_MAX = 100.54             # the saturation of (1.7) as Psi -> 0, quoted so a
                               # caller can tell a saturated fit from a divergence
TOL_INVERT = 1e-13             # bisection tolerance on the Acosta inverse
MAX_INVERT = 80                # bisection iterations; 80 halvings of [0, 0.5]


# ------------------------------------------------------------ 3 similarity

def sigma_cavity(depth, h, fn_h, dsigma=0.0):
    """Cavitation number at depth below the free surface, (1.10) and (1.12).

    sigma_c = (P_atm + rho g z' - P_c)/(rho u^2/2) = dsigma + (z'/h)(2/Fn_h^2),
    with z' = depth measured DOWNWARD from the undisturbed free surface. The
    natural-ventilation case has the cavity open to the atmosphere, P_c = P_atm,
    hence dsigma = 0 and sigma_c = 0 at the waterline; a positive dsigma is a
    vaporous or a pressurised cavity. -> array of depth's shape
    """
    depth = np.asarray(depth, dtype=float)
    if h <= 0.0:
        raise ValueError(f"immersion depth h must be positive; got {h}")
    if fn_h <= 0.0:
        raise ValueError(f"depth Froude number must be positive; got {fn_h}")
    return dsigma + (depth / h) * (2.0 / fn_h ** 2)


def froude_h(u, h, g=G):
    """Depth-based Froude number Fn_h = u/sqrt(g h), (1.1). -> float"""
    if h <= 0.0:
        raise ValueError(f"immersion depth h must be positive; got {h}")
    return float(abs(u) / np.sqrt(g * h))


def weber(u, chord, rho=RHO_WATER, gamma=GAMMA_ST):
    """Weber number We = rho u^2 c/gamma on the CHORD, not the immersion. -> float

    Wetzel's threshold WE_MIN applies to the chord of a strut section (or the
    diameter of a rod), and computing it on the immersion instead makes the gate
    fire always or never.
    """
    if chord <= 0.0:
        raise ValueError(f"chord must be positive; got {chord}")
    return float(rho * u ** 2 * chord / gamma)


def psi(sigma_c, alpha_2d):
    """The cavity parameter Psi = sigma_c/(2 alpha_2D), (1.8). -> array

    Both classical cavity solutions collapse onto this single ratio once the
    alpha_2D term of (1.5a) is dropped, which is the small-angle limit. Positive
    for a lifting section at positive incidence; the sign is the caller's
    business and a non-positive alpha_2D returns +inf, meaning no cavity.
    """
    sigma_c = np.asarray(sigma_c, dtype=float)
    alpha_2d = np.asarray(alpha_2d, dtype=float)
    small = alpha_2d <= 0.0
    safe = np.where(small, 1.0, alpha_2d)
    out = sigma_c / (2.0 * safe)
    return np.where(small, np.inf, out)


# ---------------------------------------------------------------- 4 length

def acosta_psi(length):
    """Acosta (1955) partial cavity, exact: Psi(L) for L <= 0.5, (1.6a). -> array

    Psi = (2 - L + 2 sqrt(1-L))/sqrt(L(1-L)). Monotone decreasing, diverging as
    L -> 0 and finite (5.8284) at L = 0.5. Evaluated outside (0, 1) it is not
    meaningful and returns nan rather than a complex number.
    """
    length = np.asarray(length, dtype=float)
    live = (length > 0.0) & (length < 1.0)
    safe = np.where(live, length, 0.5)
    root = np.sqrt(1.0 - safe)
    out = (2.0 - safe + 2.0 * root) / np.sqrt(safe * (1.0 - safe))
    return np.where(live, out, np.nan)


def acosta_lift_slope(length):
    """Acosta (1955) partial cavity lift slope pi(1 + 1/sqrt(1-L)), (1.6b). -> array"""
    length = np.asarray(length, dtype=float)
    live = (length >= 0.0) & (length < 1.0)
    safe = np.where(live, length, 0.5)
    return np.where(live, np.pi * (1.0 + 1.0 / np.sqrt(1.0 - safe)), np.nan)


def acosta_length(psi_value):
    """Invert (1.6a) by bisection on L in (0, 0.5]. -> array of psi_value's shape

    Monotone, so bisection is unconditionally safe and needs no derivative;
    scipy is deliberately absent from this repository. Psi below Acosta's own
    limit of 5.8284 is outside the partial-cavity branch and returns nan, which
    is what makes `cavity_length` choose the branch rather than extrapolate.
    """
    p = np.asarray(psi_value, dtype=float)
    lo = np.full(p.shape, TOL_INVERT)
    hi = np.full(p.shape, L_PARTIAL)
    live = p >= acosta_psi(L_PARTIAL)
    for _ in range(MAX_INVERT):
        mid = 0.5 * (lo + hi)
        # acosta_psi decreases with L, so a Psi above the target means L is small
        too_long = acosta_psi(mid) < p
        hi = np.where(too_long, mid, hi)
        lo = np.where(too_long, lo, mid)
    return np.where(live, 0.5 * (lo + hi), np.nan)


def tulin_length(psi_value, alpha_2d=0.0):
    """Tulin (1953) supercavity, exact: L = (2 alpha/sigma + alpha)^2 + 1, (1.5a).

    In terms of Psi, 2 alpha/sigma = 1/Psi, so L = (1/Psi + alpha)^2 + 1. The
    alpha term is the one the paper drops for small angles; it is kept here and
    defaulted away, so the caller chooses. -> array
    """
    p = np.asarray(psi_value, dtype=float)
    live = p > 0.0
    safe = np.where(live, p, 1.0)
    return np.where(live, (1.0 / safe + alpha_2d) ** 2 + 1.0, np.inf)


def tulin_psi(length, alpha_2d=0.0):
    """Invert (1.5a) analytically: Psi(L) for L > 1. -> array"""
    length = np.asarray(length, dtype=float)
    live = length > 1.0
    safe = np.where(live, length, 2.0)
    denom = np.sqrt(safe - 1.0) - alpha_2d
    return np.where(live & (denom > 0.0), 1.0 / np.where(denom > 0.0, denom, 1.0),
                    np.nan)


def harwood_length(psi_value):
    """The paper's smooth blend of both classical branches, (1.7). -> array

    L = (2.67 Psi + 96.62)/(Psi^3 - 7.1 Psi^2 + 49.42 Psi + 0.961). The
    denominator has no positive real root - Psi^2 - 7.1 Psi + 49.42 has negative
    discriminant - so the fit is finite everywhere and saturates at L_FIT_MAX as
    Psi -> 0. That saturation belongs to the fit, not to the physics.
    """
    p = np.asarray(psi_value, dtype=float)
    finite = np.isfinite(p)
    safe = np.where(finite, p, 1.0)
    out = ((2.67 * safe + 96.62)
           / (safe ** 3 - 7.1 * safe ** 2 + 49.42 * safe + 0.961))
    return np.where(finite, out, 0.0)


def harwood_length_low_order(psi_value):
    """The single-term long-cavity relation L = 2.31/Psi, (1.8). -> array

    Used by the washout derivation (4.1) because it is the form that makes (4.5)
    close analytically. It agrees with (1.7) to about 5% only near L = 2, and it
    is NOT an approximation to Tulin: Tulin grows as sigma^-2 and this as
    sigma^-1.
    """
    p = np.asarray(psi_value, dtype=float)
    live = p > 0.0
    return np.where(live, 2.31 / np.where(live, p, 1.0), np.inf)


def _monotone_slope(slope, secant):
    """Fritsch-Carlson limiter: clamp a Hermite end slope to keep it monotone."""
    if secant == 0.0:
        return 0.0
    if slope * secant <= 0.0:
        return 0.0
    return np.sign(secant) * min(abs(slope), 3.0 * abs(secant))


def _blend_anchors(alpha_2d):
    """The two branch joins in (log Psi, log L), with limited end slopes.

    The end slopes are differenced in L and evaluated through each branch's own
    Psi(L), which stays inside that branch's domain; differencing in Psi instead
    steps off the end of the branch and returns nan.
    """
    psi_a = float(acosta_psi(L_PARTIAL))               # partial end, L = 0.5
    psi_b = float(tulin_psi(L_SUPER, alpha_2d))        # super end,   L = 1.25
    d = 1e-5
    m_a = (np.log(1.0 + d) - np.log(1.0 - d)) / np.log(
        float(acosta_psi(L_PARTIAL * (1.0 + d)))
        / float(acosta_psi(L_PARTIAL * (1.0 - d))))
    m_b = (np.log(1.0 + d) - np.log(1.0 - d)) / np.log(
        float(tulin_psi(L_SUPER * (1.0 + d), alpha_2d))
        / float(tulin_psi(L_SUPER * (1.0 - d), alpha_2d)))
    span = np.log(psi_a) - np.log(psi_b)
    secant = (np.log(L_PARTIAL) - np.log(L_SUPER)) / span
    return (psi_b, psi_a, np.log(L_SUPER), np.log(L_PARTIAL),
            _monotone_slope(m_b, secant) * span, _monotone_slope(m_a, secant) * span)


def cavity_length(psi_value, model="exact", alpha_2d=0.0):
    """Cavity length over chord from the cavity parameter. -> array

    model='exact'  the classical solutions where each is valid - Acosta below
                   L = 0.5, Tulin above L = 1.25 - joined by a monotone C1 cubic
                   Hermite in (log Psi, log L) across the gap where both run into
                   the hodograph singularity at L = 1 and are unphysical. This is
                   the default because it is the most accurate available, and
                   because the blend is monotone and continuously differentiable,
                   which the cavity fixed point of vent_solve needs.
    model='fit'    equation (1.7), the paper's rational-polynomial blend.
    model='low'    equation (1.8), L = 2.31/Psi; required to reproduce the
                   washout derivation and valid only near L = 2.
    """
    if model == "fit":
        return harwood_length(psi_value)
    if model == "low":
        return harwood_length_low_order(psi_value)
    if model != "exact":
        raise ValueError(f"unknown cavity length model '{model}'; choose from "
                         f"'exact', 'fit', 'low'")
    p = np.asarray(psi_value, dtype=float)
    psi_b, psi_a, log_lb, log_la, slope_b, slope_a = _blend_anchors(alpha_2d)
    partial = p >= psi_a
    super_c = p <= psi_b
    out = np.zeros(p.shape, dtype=float)
    # the two branches, each evaluated only where it holds
    out = np.where(partial, acosta_length(np.where(partial, p, psi_a)), out)
    out = np.where(super_c, tulin_length(np.where(super_c, p, psi_b), alpha_2d), out)
    # the blend, a cubic Hermite in t from the supercavity end to the partial end
    mid = ~(partial | super_c)
    if np.any(mid):
        t = ((np.log(np.where(mid, p, psi_a)) - np.log(psi_b))
             / (np.log(psi_a) - np.log(psi_b)))
        h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
        h10 = t ** 3 - 2.0 * t ** 2 + t
        h01 = -2.0 * t ** 3 + 3.0 * t ** 2
        h11 = t ** 3 - t ** 2
        blend = np.exp(h00 * log_lb + h10 * slope_b + h01 * log_la + h11 * slope_a)
        out = np.where(mid, blend, out)
    return np.where(np.isfinite(p), out, 0.0)


# ------------------------------------------------------------- 5 sectional

def lift_slope(length):
    """Sectional lift-curve slope from the cavity length, (1.9). -> array

    a0 = ((pi/2)L^3 - 2L^2 + 4.5L + 1)/(L^3 - L^2 + 0.75L + 1/(2 pi)), which
    reaches 2 pi at L = 0 and pi/2 as L -> infinity, both exactly, and sits
    within 1.5% of Acosta over the whole partial-cavity branch.

    The constant 1/(2 pi) is not legible in the paper's text layer. It was
    recovered two independent ways: from the two analytic limits, and by
    checking that it makes the paper's own (4.3) algebraically identical to
    Cl_2D/a0. The plausible misreading 2 pi gives a0(0) = 1 with an otherwise
    smooth curve, so `test_vent` pins the limits rather than the coefficients.
    """
    length = np.asarray(length, dtype=float)
    big = ~np.isfinite(length)
    safe = np.where(big, 1.0, np.maximum(length, 0.0))
    num = (np.pi / 2.0) * safe ** 3 - 2.0 * safe ** 2 + 4.5 * safe + 1.0
    den = safe ** 3 - safe ** 2 + 0.75 * safe + 1.0 / (2.0 * np.pi)
    return np.where(big, np.pi / 2.0, num / den)


def centre_of_pressure(length):
    """Centre of pressure forward of mid-chord, x_cp/c, from (3.4). -> array

    A tanh blend from the wetted quarter chord to the supercavitating 3/16, so
    e(0) = 0.24888 rather than exactly 1/4 - 0.45% low, which is the blend and
    not an error - while e(0.5) = 7/32 and e(inf) = 3/16 exactly.
    """
    length = np.asarray(length, dtype=float)
    e_wet, e_super = 0.25, 3.0 / 16.0
    big = ~np.isfinite(length)
    safe = np.where(big, 1.0, length)
    s = np.tanh((safe - 0.5) / 0.25)
    out = 0.5 * (e_wet * (1.0 - s) + e_super * (1.0 + s))
    return np.where(big, e_super, out)


# ------------------------------------------------------------- 6 stability

def jet_speed(u, sigma_c):
    """Re-entrant jet speed |Uj| = u sqrt(1 + sigma_c), (3.1). -> array

    Bernoulli between the onset flow and the cavity boundary, so it equals u
    exactly at sigma_c = 0 and never falls below it. Viscosity would slow a thin
    jet (Callenaere et al. 2001); the paper argues that above about 5 degrees the
    jet is thick enough for that to be neglected, and it is neglected here.
    """
    return np.abs(u) * np.sqrt(1.0 + np.asarray(sigma_c, dtype=float))


def jet_components(u, sigma_c, phi):
    """Streamwise and spanwise jet components, (3.2a) and (3.2b). -> (Uj, Wj)

    The jet is the incoming flow reflected about the cavity closure line, so it
    leaves at 2*phi from the horizontal: Uj = |Uj| cos(2 phi), Wj = |Uj| sin(2
    phi). phi > 45 degrees makes Uj negative, an upstream jet, which is the
    destabilising case.
    """
    mag = jet_speed(u, sigma_c)
    phi = np.asarray(phi, dtype=float)
    return mag * np.cos(2.0 * phi), mag * np.sin(2.0 * phi)


def closure_angle(depth, x_closure):
    """Mean and local angle of the cavity closure line from the HORIZONTAL.

    phi = atan|dz'/dx_closure|, so a cavity of uniform length gives phi = 90
    degrees - the closure line normal to the flow, the classical two-dimensional
    re-entrant jet, and the unstable case - while a strongly tapered cavity gives
    a small phi and a jet swept towards the trailing edge. phi_bar comes from a
    least-squares affine fit through the closure line over the depth extent
    given, which is how the paper measures it (figure 8a, three stations).

    Needs at least two stations. -> (phi_bar, phi_local of depth's shape)
    """
    depth = np.asarray(depth, dtype=float).ravel()
    x_closure = np.asarray(x_closure, dtype=float).ravel()
    if depth.size != x_closure.size:
        raise ValueError(f"depth and x_closure must match; got {depth.size} "
                         f"and {x_closure.size}")
    if depth.size < 2:
        raise ValueError("the closure angle needs at least two depth stations")
    # local angle from one-sided differences of the closure line
    dz = np.gradient(depth)
    dx = np.gradient(x_closure)
    phi_local = np.arctan2(np.abs(dz), np.abs(dx))
    # phi_bar from the affine fit dz'/dx of the whole line
    slope = np.polyfit(x_closure, depth, 1)[0] if np.ptp(x_closure) > 0.0 \
        else np.inf
    phi_bar = np.arctan(abs(slope)) if np.isfinite(slope) else np.pi / 2.0
    return float(phi_bar), phi_local


def unstable_closure(phi_bar, phi_crit=PHI_CRIT):
    """True when the re-entrant jet has an upstream component, (3.2a). -> bool"""
    return bool(phi_bar > phi_crit)


def regime(depth_cavity, phi_bar, h, tol=1e-9):
    """Flow regime from the cavity depth and the mean closure angle, section 3.2.

    D = 0                              fully wetted        'FW'
    D = h and phi_bar < 45 degrees      fully ventilated    'FV'
    0 < D < h or phi_bar > 45 degrees   partially ventilated 'PV'

    The three are exhaustive and mutually exclusive, which `test_vent` checks on
    a grid; the regions of the paper's map OVERLAP in the parameters, not in this
    classification, and the overlap lives in the hysteresis of vent_cavity.
    -> 'FW' | 'PV' | 'FV'
    """
    if h <= 0.0:
        raise ValueError(f"immersion depth h must be positive; got {h}")
    if depth_cavity <= tol * h:
        return "FW"
    if depth_cavity >= (1.0 - tol) * h and not unstable_closure(phi_bar):
        return "FV"
    return "PV"


# --------------------------------------------------------------- 7 washout

def helmbold(a0, ar):
    """Helmbold (1942) small-aspect-ratio lift slope, (1.20). -> float or array

    dCL/dalpha = a0/(a0/(pi AR) + sqrt(1 + (a0/(pi AR))^2)), which tends to a0
    as AR -> infinity and to (pi/2) AR as AR -> 0, the slender-body limit,
    independently of a0.

    For a surface-piercing strut the aspect ratio to use is the IMMERSED one,
    AR_h = h/c, not twice it. The negative free-surface image makes the waterline
    behave as a tip - the loading vanishes there - so the immersed span is the
    whole span of the equivalent wing. Passing 2 AR_h is the rigid-wall image and
    moves the lift slope by tens of per cent while looking entirely plausible.
    """
    ar = np.asarray(ar, dtype=float)
    if np.any(ar <= 0.0):
        raise ValueError("the aspect ratio must be positive")
    r = a0 / (np.pi * ar)
    return a0 / (r + np.sqrt(1.0 + r ** 2))


def elliptic_shape(kappa):
    """Elliptic loading shape with unit integral over kappa in [0,1], (4.7).

    E = (4/pi) sqrt(1 - (2 kappa - 1)^2), kappa = z'/h. Symmetric about
    mid-immersion, vanishing at the waterline and at the tip, which is the shape
    the negative image and the free tip together impose. -> array
    """
    kappa = np.asarray(kappa, dtype=float)
    live = (kappa >= 0.0) & (kappa <= 1.0)
    safe = np.where(live, kappa, 0.5)
    return np.where(live, (4.0 / np.pi) * np.sqrt(1.0 - (2.0 * safe - 1.0) ** 2),
                    0.0)


def lift_weighted_slope(ar_h, n=400, model_length=None):
    """Lift-weighted mean sectional slope a0*, (4.6) and (4.8). -> float

    The cavity length along the immersion is constrained by the phi_bar = 45
    degree condition to Lc(kappa)/c = AR_h (1 - kappa), equation (4.6); a0 of
    (1.9) is evaluated on that and weighted with the elliptic shape (4.7).
    Trapezoidal quadrature, as the paper's "evaluated using numerical
    integration", but on the substitution kappa = (1 - cos theta)/2, which turns
    E(kappa) dkappa into (2/pi) sin^2(theta) dtheta. The elliptic shape has an
    infinite derivative at both ends, so the trapezoidal rule in kappa converges
    as n^-1.5 and reaches only 4e-7 on the unit integral at n = 20000; after the
    substitution the integrand is smooth and 400 intervals are ample.
    """
    theta = np.linspace(0.0, np.pi, int(n) + 1)
    kappa = 0.5 * (1.0 - np.cos(theta))
    length = ar_h * (1.0 - kappa) if model_length is None else model_length(kappa)
    weight = (2.0 / np.pi) * np.sin(theta) ** 2
    return float(np.trapezoid(lift_slope(length) * weight, theta))


def washout_froude(cl, ar_h, kappa=0.5):
    """Semi-theoretical washout Froude number, (4.5). -> float or array

    Fn_h = (pi/4) sqrt( ((pi/2)AR^4 - 4AR^3 + 18AR^2 + 8AR)
                        / (2.31 CL (pi AR^3 - 2 pi AR^2 + 3 pi AR + 4)) )

    at the representative section kappa = 0.5, on the reasoning that a jet
    threatening the cavity is reflected about its deeply submerged part. Below
    this Froude number a fully ventilated cavity washes out.

    kappa is accepted for the general case and solved from the chain (4.1)-(4.4)
    rather than from this closed form, which is only derived at kappa = 0.5. The
    two agree to 4e-16 at kappa = 0.5, and `test_vent` asserts it: the closed
    form is verified against the relations it was derived from, which tests the
    transcription and the algebra, not the physics.
    """
    cl = np.asarray(cl, dtype=float)
    ar = np.asarray(ar_h, dtype=float)
    if np.any(cl <= 0.0):
        raise ValueError("the lift coefficient must be positive at washout")
    if np.any(ar <= 0.0):
        raise ValueError("the immersed aspect ratio must be positive")
    if kappa != 0.5:
        return washout_froude_chain(cl, ar, kappa)
    num = (np.pi / 2.0) * ar ** 4 - 4.0 * ar ** 3 + 18.0 * ar ** 2 + 8.0 * ar
    den = 2.31 * cl * (np.pi * ar ** 3 - 2.0 * np.pi * ar ** 2
                       + 3.0 * np.pi * ar + 4.0)
    return (np.pi / 4.0) * np.sqrt(num / den)


def washout_froude_chain(cl, ar_h, kappa=0.5):
    """The same boundary from (4.1)-(4.4) directly, without the closed form.

    Lc/c = (1-kappa) AR_h from the unity-slope condition; Cl_2D from the elliptic
    shape (4.2); alpha_2D = Cl_2D/a0 with a0 of (1.9) at that length; sigma_c =
    2 kappa/Fn_h^2 from (4.4); and (4.1), (1-kappa) AR_h = 4.62 alpha_2D/sigma_c,
    solved for Fn_h. Independent of (4.5) except for sharing (1.9), which is
    what makes the comparison a check on the transcription. -> float or array
    """
    cl = np.asarray(cl, dtype=float)
    ar = np.asarray(ar_h, dtype=float)
    length = (1.0 - kappa) * ar
    cl_2d = (4.0 / np.pi) * cl * np.sqrt(1.0 - (2.0 * kappa - 1.0) ** 2)
    alpha_2d = cl_2d / lift_slope(length)
    # (4.1) with sigma_c = 2 kappa / Fn^2:  length = 4.62 alpha_2d Fn^2/(2 kappa)
    return np.sqrt(length * 2.0 * kappa / (4.62 * alpha_2d))


def breslin_skalak(cl):
    """Breslin & Skalak (1959) stability boundary, (1.2). -> float or array

    Fully ventilated flow is stable only if CL > 5 Fn_h^-2 AND Fn_h > 3, so the
    boundary is max(sqrt(5/CL), 3); the two conditions cross at CL = 5/9 and a
    code applying only the first is wrong above that. The paper shows this
    over-predicts the washout Froude number - by a factor 2.86 at AR_h = 1,
    CL = 0.5 - and is properly read as a bound above which full ventilation is
    likely stable, not one below which it is necessarily unstable.
    """
    cl = np.asarray(cl, dtype=float)
    if np.any(cl <= 0.0):
        raise ValueError("the lift coefficient must be positive")
    return np.maximum(np.sqrt(5.0 / cl), 3.0)


def washout_margin(fn_h, cl, ar_h):
    """Signed distance from the washout boundary, (Fn_h - Fn_washout)/Fn_washout.

    Negative means a fully ventilated cavity cannot be sustained. Small in
    magnitude means the regime is metastable and a marched run will flip, which
    is what the driver warns on. -> float
    """
    fn_w = float(washout_froude(cl, ar_h))
    return float((fn_h - fn_w) / fn_w)


# ------------------------------------------------------ 8 plotting and CLI

def _report(title):
    """Underlined title, the house table header."""
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    """Check the sectional relations against the closed forms they came from."""
    print("Sectional cavity physics: vent_section.py")
    print("Every number below is measured by this run.")

    _report("cavity length: the fits against Acosta's exact relation")
    print("      L    Psi (Acosta)   L from (1.7)   error in L   L from (1.8)")
    for length in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        p = float(acosta_psi(length))
        fit = float(harwood_length(p))
        low = float(harwood_length_low_order(p))
        print(f"  {length:5.2f}   {p:12.4f}   {fit:12.4f}   "
              f"{100 * (fit - length) / length:+9.1f}%   {low:12.4f}")
    print("  The comparison that matters is in Psi, not in L: dPsi/dL is large")
    print("  near the branch join, so the 8.7% error in L at L = 0.5 is about")
    print("  1% in Psi. Equation (1.8) is a long-cavity relation and is not")
    print("  expected to hold on this branch at all.")

    _report("lift slope (1.9) against Acosta (1.6b), and its two limits")
    print("      L    a0 from (1.9)   Acosta a0    ratio")
    for length in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        got, exact = float(lift_slope(length)), float(acosta_lift_slope(length))
        print(f"  {length:5.2f}   {got:13.4f}   {exact:9.4f}   {got / exact:6.4f}")
    print(f"  a0(0) = {float(lift_slope(0.0)):.10f} against 2 pi = "
          f"{2 * np.pi:.10f}")
    print(f"  a0(inf) = {float(lift_slope(np.inf)):.10f} against pi/2 = "
          f"{np.pi / 2:.10f}")

    _report("the washout boundary (4.5) against the chain (4.1) to (4.4)")
    print("   AR_h     CL   Fn from (4.5)   Fn from the chain   difference"
          "   Breslin & Skalak")
    for ar in (0.5, 1.0, 1.5, 2.0):
        for cl in (0.2, 0.5, 1.0):
            closed = float(washout_froude(cl, ar))
            chain = float(washout_froude_chain(cl, ar))
            print(f"  {ar:5.1f}  {cl:5.2f}   {closed:13.4f}   {chain:17.4f}   "
                  f"{abs(closed - chain):10.2e}   {float(breslin_skalak(cl)):16.3f}")
    print("  The closed form is verified against the relations it was derived")
    print("  from, which tests the transcription and the algebra, not the")
    print("  physics. Breslin & Skalak bounds it from above throughout.")

    _report("the closure angle convention, from the horizontal")
    depth = np.linspace(0.0, 1.0, 5)
    for name, x_c in (("uniform cavity", np.full(5, 0.6)),
                      ("tapered, dx = dz'", 0.9 - depth),
                      ("strongly tapered", 1.4 - 2.0 * depth)):
        phi_bar, _ = closure_angle(depth, x_c)
        u_j, w_j = jet_components(1.0, 0.2, phi_bar)
        print(f"  {name:20s} phi_bar = {np.degrees(phi_bar):5.1f} deg   "
              f"Uj = {u_j:+.3f}   Wj = {w_j:+.3f}   "
              f"{'unstable' if unstable_closure(phi_bar) else 'stable'}")
    print("  A cavity of uniform length is the classical two-dimensional case:")
    print("  the jet points straight upstream and pinches the cavity off.")
    print("\nDone. docs/VENTILATION.md tabulates these numbers.")


if __name__ == "__main__":
    main()

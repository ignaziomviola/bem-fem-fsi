"""Constitutive models for fem_solid.py: the seam the material extension runs through.

The whole structural design turns on one decision recorded here. The stress and
the tangent modulus are evaluated at the Gauss point by a material object, not
built into the element, because that is where anisotropy and inelasticity live.
An equivalent-beam model with sectional EI and GJ cannot be extended to a
composite lay-up or to a viscoelastic skin, whatever its interface looks like:
sectional stiffnesses are not a constitutive law. A continuum element calling
`response` at each Gauss point can, and the extension is then a new class in
this module with nothing else in the code touched.

The interface is

    response(strain, strain_rate=None, history=None, dt=None, orientation=None)
        -> {"stress", "tangent", "rate_tangent", "history"}

vectorised over the G Gauss points of an element: strain (G, 6) in, stress
(G, 6) and tangent (G, 6, 6) out. `history` carries n_history internal
variables per Gauss point, `dt` the time step, `orientation` a (3, 3) rotation
of the material axes into the global frame. Version one leaves history empty
for the two elastic materials and uses it for the viscoelastic one, so neither
argument is a speculative code path: every one of them is exercised and gated.

An elastic-plastic material enters as a fourth class whose `response` performs
a return mapping, writes the plastic strain and the hardening variable into
`history`, and returns the consistent algorithmic tangent. Nothing above it
changes. That is the claim this module exists to make true.

Voigt convention, binding on this module and on fem_solid.py:

    strain = [e11, e22, e33, 2*e12, 2*e23, 2*e31]     engineering shear
    stress = [s11, s22, s33,   s12,   s23,   s31]

so that stress . strain is twice the strain energy density with no stray
factors, and the shear rows of the tangent carry G rather than 2G.
"""

import numpy as np

# Voigt index pairs, in the order above. Used to build the rotation operator
# generically rather than by transcribing a Bond matrix, which is where sign
# and factor-of-two errors in anisotropic elasticity come from.
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def voigt_to_tensor(strain):
    """Strain Voigt vector(s) (..., 6) -> symmetric tensor(s) (..., 3, 3)."""
    e = np.asarray(strain, dtype=float)
    out = np.zeros(e.shape[:-1] + (3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = e[..., 0], e[..., 1], e[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = 0.5 * e[..., 3]
    out[..., 1, 2] = out[..., 2, 1] = 0.5 * e[..., 4]
    out[..., 2, 0] = out[..., 0, 2] = 0.5 * e[..., 5]
    return out


def tensor_to_voigt(tensor):
    """Symmetric strain tensor(s) (..., 3, 3) -> Voigt vector(s) (..., 6)."""
    t = np.asarray(tensor, dtype=float)
    return np.stack([t[..., 0, 0], t[..., 1, 1], t[..., 2, 2],
                     t[..., 0, 1] + t[..., 1, 0],
                     t[..., 1, 2] + t[..., 2, 1],
                     t[..., 2, 0] + t[..., 0, 2]], axis=-1)


def strain_rotation(orientation):
    """Voigt strain transformation T from the global to the material frame.

    `orientation` R has the material axes as its COLUMNS, expressed in global
    coordinates, so a tensor transforms as e_mat = R^T e_glob R. T is built by
    pushing the six Voigt basis strains through that operation rather than by
    transcribing the Bond matrix: the engineering-shear factors then take care
    of themselves, which is the usual place anisotropic elasticity goes wrong.

    With e_mat = T e_glob, energy invariance gives s_glob = T^T s_mat and hence

        D_glob = T^T D_mat T,

    which is the only formula anything outside this module needs.
    """
    rot = np.asarray(orientation, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError(f"orientation must be (3, 3); got {rot.shape}")
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-10):
        raise ValueError("orientation must be orthonormal (a rotation of the "
                         "material axes into the global frame)")
    basis = np.eye(6)
    cols = [tensor_to_voigt(rot.T @ voigt_to_tensor(basis[k]) @ rot)
            for k in range(6)]
    return np.stack(cols, axis=1)


def rotate_tangent(d_mat, orientation):
    """D in material axes -> D in global axes. Identity when orientation is None."""
    if orientation is None:
        return d_mat
    t_mat = strain_rotation(orientation)
    return t_mat.T @ d_mat @ t_mat


class Material:
    """Interface every constitutive model implements.

    Attributes
    ----------
    name : str
    rho : float                mass density
    n_history : int            internal variables stored per Gauss point
    constant_tangent : bool    True if `tangent` is independent of the state,
                               which lets the assembly evaluate it once per
                               element instead of once per Gauss point. It is
                               an optimisation only: the answer must not depend
                               on it, and a unit test asserts that it does not.
    """

    name = "material"
    rho = 0.0
    n_history = 0
    constant_tangent = False

    def tangent(self, orientation=None):
        """The (6, 6) tangent modulus, for materials that have a constant one."""
        raise NotImplementedError

    def response(self, strain, strain_rate=None, history=None, dt=None,
                 orientation=None):
        """Stress, tangents and updated history at G Gauss points."""
        raise NotImplementedError

    def initial_history(self, n_points):
        """(n_points, n_history) starting internal variables."""
        return np.zeros((int(n_points), self.n_history))


class IsotropicElastic(Material):
    """Hooke's law for an isotropic material. The verified default.

    Parameters are the engineering ones, Young's modulus and Poisson's ratio,
    with the Lame constants derived; nu -> 0.5 is rejected rather than clipped,
    because a low-order displacement element locks volumetrically long before
    it gets there and a silent clip would hide that.
    """

    constant_tangent = True

    def __init__(self, e_mod, nu, rho=1.0, name="isotropic"):
        if e_mod <= 0.0:
            raise ValueError(f"Young's modulus must be positive; got {e_mod}")
        if not -1.0 < nu < 0.5:
            raise ValueError(
                f"Poisson's ratio must satisfy -1 < nu < 0.5; got {nu}. At nu "
                f"= 0.5 the material is incompressible and a displacement "
                f"element locks: use a mixed formulation, not this one")
        self.e_mod, self.nu, self.rho, self.name = float(e_mod), float(nu), \
            float(rho), name
        self.shear = 0.5 * e_mod / (1.0 + nu)
        self.lame = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.bulk = e_mod / (3.0 * (1.0 - 2.0 * nu))

    def tangent(self, orientation=None):
        d_mat = np.zeros((6, 6))
        d_mat[:3, :3] = self.lame
        d_mat[0, 0] = d_mat[1, 1] = d_mat[2, 2] = self.lame + 2.0 * self.shear
        d_mat[3, 3] = d_mat[4, 4] = d_mat[5, 5] = self.shear
        return rotate_tangent(d_mat, orientation)      # isotropic: a no-op

    def response(self, strain, strain_rate=None, history=None, dt=None,
                 orientation=None):
        eps = np.atleast_2d(np.asarray(strain, dtype=float))
        d_mat = self.tangent(orientation)
        return {"stress": eps @ d_mat.T,
                "tangent": np.broadcast_to(d_mat, (len(eps), 6, 6)),
                "rate_tangent": None,
                "history": np.zeros((len(eps), 0))}


class OrthotropicElastic(Material):
    """Nine-constant orthotropic elasticity in the material axes.

    Present in version one for one reason: it turns the claim that the code
    extends to non-isotropic materials into a property that is exercised and
    gated rather than a promise. It costs forty lines because the element never
    assumed isotropy in the first place.

    The compliance is built in the material frame and inverted, so the reciprocal
    relations nu_ij/E_i = nu_ji/E_j hold by construction and only the three
    independent Poisson ratios are asked for. Positive definiteness is checked:
    an orthotropic set that violates it is a data error, not a stiff material,
    and it produces negative strain energy in silence otherwise.
    """

    constant_tangent = True

    def __init__(self, e1, e2, e3, g12, g23, g31, nu12, nu13, nu23, rho=1.0,
                 name="orthotropic"):
        self.moduli = (float(e1), float(e2), float(e3))
        self.shears = (float(g12), float(g23), float(g31))
        self.poisson = (float(nu12), float(nu13), float(nu23))
        self.rho, self.name = float(rho), name
        if min(self.moduli) <= 0.0 or min(self.shears) <= 0.0:
            raise ValueError("all moduli and shear moduli must be positive")
        compliance = np.zeros((6, 6))
        compliance[0, 0], compliance[1, 1], compliance[2, 2] = \
            1.0 / e1, 1.0 / e2, 1.0 / e3
        compliance[0, 1] = compliance[1, 0] = -nu12 / e1
        compliance[0, 2] = compliance[2, 0] = -nu13 / e1
        compliance[1, 2] = compliance[2, 1] = -nu23 / e2
        compliance[3, 3], compliance[4, 4], compliance[5, 5] = \
            1.0 / g12, 1.0 / g23, 1.0 / g31
        if np.min(np.linalg.eigvalsh(compliance)) <= 0.0:
            raise ValueError(
                "the orthotropic constants give a compliance that is not "
                "positive definite; check the Poisson ratios against the "
                "moduli (nu12^2 < E1/E2 and its two companions)")
        self._d_mat = np.linalg.inv(compliance)

    def tangent(self, orientation=None):
        return rotate_tangent(self._d_mat, orientation)

    def response(self, strain, strain_rate=None, history=None, dt=None,
                 orientation=None):
        eps = np.atleast_2d(np.asarray(strain, dtype=float))
        d_mat = self.tangent(orientation)
        return {"stress": eps @ d_mat.T,
                "tangent": np.broadcast_to(d_mat, (len(eps), 6, 6)),
                "rate_tangent": None,
                "history": np.zeros((len(eps), 0))}


class KelvinVoigt(Material):
    """Viscoelastic solid: sigma = D e + tau D e_dot, over any elastic base.

    The smallest material that exercises the rate and history arguments, so
    that neither is dead code in version one. It wraps an elastic material
    rather than duplicating it, which is what the interface is for, and it
    stores the previous strain in `history` even though the rate is supplied
    directly - a genuinely history-dependent law (plasticity, damage) needs
    exactly that slot, and leaving it empty here would leave the path untested.

    The stress rate term is returned as `rate_tangent` rather than folded into
    `tangent`, so the assembly puts it into the damping matrix where it belongs
    and the time integrator sees an exactly stiffness-proportional damping
    C = tau K. That is what makes the analytical decay gate exact instead of
    first order in the step.
    """

    constant_tangent = True

    def __init__(self, base, tau, name=None):
        self.base = base
        self.tau = float(tau)
        if self.tau < 0.0:
            raise ValueError(f"retardation time must be non-negative; got {tau}")
        self.rho = base.rho
        self.name = name or f"kelvin-voigt({base.name})"
        self.n_history = 6                       # the previous strain

    def tangent(self, orientation=None):
        return self.base.tangent(orientation)

    def response(self, strain, strain_rate=None, history=None, dt=None,
                 orientation=None):
        eps = np.atleast_2d(np.asarray(strain, dtype=float))
        d_mat = self.tangent(orientation)
        if strain_rate is not None:
            rate = np.atleast_2d(np.asarray(strain_rate, dtype=float))
        elif history is not None and dt:
            rate = (eps - np.atleast_2d(history)[:, :6]) / float(dt)
        else:
            rate = np.zeros_like(eps)
        stress = eps @ d_mat.T + self.tau * (rate @ d_mat.T)
        return {"stress": stress,
                "tangent": np.broadcast_to(d_mat, (len(eps), 6, 6)),
                "rate_tangent": np.broadcast_to(self.tau * d_mat,
                                                (len(eps), 6, 6)),
                "history": eps.copy()}


def isotropic_from_lame(lame, shear, rho=1.0):
    """IsotropicElastic from the Lame constants, for textbook comparisons."""
    e_mod = shear * (3.0 * lame + 2.0 * shear) / (lame + shear)
    nu = 0.5 * lame / (lame + shear)
    return IsotropicElastic(e_mod, nu, rho)

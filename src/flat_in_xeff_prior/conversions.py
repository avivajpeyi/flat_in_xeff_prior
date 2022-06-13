import numpy as np

from typing import List

from scipy.optimize import minimize
from bilby.core.prior import ConditionalPowerLaw, Prior


def get_m1_m2(q, mc):
    mt = mc * (1 + q) ** 1.2 / q ** 0.6
    m1 = mt / (1 + q)
    m2 = m1 * q
    return m1, m2


def get_s1z(xeff, q, s2z):
    return (1 + q) * xeff - q * s2z


def get_s2z(xeff, q, s1z):
    return ((1 + q) * xeff - s1z) / q


def get_s1z_from_limits(s1z_min, s1z_max, cxd):
    return s1z_min + cxd * (s1z_max - s1z_min)


def s1z_lim(xeff, q):
    s1z_min = np.maximum(get_s1z(xeff, q, s2z=1), -1)
    s1z_max = np.minimum(get_s1z(xeff, q, s2z=-1), 1)
    return s1z_min, s1z_max


def condition_func_r1(reference_params, xeff, q, cxd):
    s1z_min, s1z_max = s1z_lim(xeff, q)
    s1z = get_s1z_from_limits(s1z_min, s1z_max, cxd)
    s2z = get_s2z(xeff, q, s1z)
    return dict(alpha=1, minimum=0, maximum=1 - s1z ** 2)


def condition_func_r2(reference_params, xeff, q, cxd):
    s1z_min, s1z_max = s1z_lim(xeff, q)
    s1z = get_s1z_from_limits(s1z_min, s1z_max, cxd)
    s2z = get_s2z(xeff, q, s1z)
    return dict(alpha=1, minimum=0, maximum=1 - s2z ** 2)


@np.vectorize
def calc_chix(phi, xy_mag):
    chix = xy_mag * np.cos(phi)
    if chix ** 2 <= 1:
        return chix
    else:
        return np.nan


@np.vectorize
def calc_chiy(phi, xy_mag):
    chiy = xy_mag * np.sin(phi)
    if chiy ** 2 <= 1:
        return chiy
    else:
        return np.nan


@np.vectorize
def sqr_sum(x, y, z):
    return x ** 2 + y ** 2 + z ** 2


def flat_in_xeff_conversion_fn(parameters):
    """
    To handle the conditional prior stuff
    """
    xeff, q, cxd = parameters["chi_eff"], parameters["mass_ratio"], parameters["cxd"]
    p = parameters.copy()
    s1z_min, s1z_max = s1z_lim(xeff, q)
    p["s1z"] = get_s1z_from_limits(s1z_min, s1z_max, cxd)
    p["s2z"] = get_s2z(xeff, q, p["s1z"])

    p["R1"] = 1 - p["s1z"] ** 2
    p["R2"] = 1 - p["s2z"] ** 2

    p["s1x"] = calc_chix(phi=p["phi_1"], xy_mag=p["r1"])
    p["s1y"] = calc_chiy(phi=p["phi_1"], xy_mag=p["r1"])

    p["s2x"] = calc_chix(phi=p["phi_2"], xy_mag=p["r2"])
    p["s2y"] = calc_chiy(phi=p["phi_2"], xy_mag=p["r2"])

    return p

def flatxeff_params_to_lal(parameters):
    """
    From:
        mass_ratio, chirp_mass, chi_eff, cxd, phi_1, phi_2, r1, r2,
        luminosity_distance, dec, ra, theta_jn, psi,phase, s1, s2

    To:
        luminosity_distance, theta_jn, phase, a_1, a_2, tilt_1,
        tilt_2, phi_12, phi_jl,

    """
    p = parameters.copy()
    p = flat_in_xeff_conversion_fn(p)
    x = {}
    x["a_1"] = np.sqrt(sqr_sum(p["s1x"], p["s1y"], p["s1z"]))  # r1 <= 1
    x["a_2"] = np.sqrt(sqr_sum(p["s2x"], p["s2y"], p["s2z"]))  # r1 <= 1
    cos_tilt_1 = p["s1z"] / x["a_1"]
    cos_tilt_2 = p["s2z"] / x["a_2"]
    x['tilt_1'] = np.arccos(cos_tilt_1)
    x['tilt_2'] = np.arccos(cos_tilt_2)
    x['mass_1'], x['mass_2'] = get_m1_m2(p['mass_ratio'], p['chirp_mass'])
    phi_12 = p['phi_2'] - p['phi_1']
    if phi_12 < 0:
        phi_12 += np.pi * 2
    x['phi_12'] = phi_12
    for i in ["luminosity_distance", "theta_jn", "phase", "phi_jl", "psi", "ra", "dec"]:
        x[i] = p[i]
    return x





class ConditionalR1(ConditionalPowerLaw):
    def __init__(self, name=None, latex_label=None, minimum=0, maximum=1):
        super(ConditionalPowerLaw, self).__init__(
            alpha=1, minimum=0, maximum=1, name=None, latex_label=None,
            unit=None, boundary=None, condition_func=self._condition_function
        )
        self._required_variables = ["chi_eff", "mass_ratio", "cxd"]

    def _condition_function(self, reference_params, **kwargs):
        xeff = kwargs[self._required_variables[0]]
        q = kwargs[self._required_variables[1]]
        cxd = kwargs[self._required_variables[2]]
        s1z_min, s1z_max = s1z_lim(xeff, q)
        s1z = get_s1z_from_limits(s1z_min, s1z_max, cxd)
        return dict(alpha=1, minimum=0, maximum=1 - s1z ** 2)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict






class ConditionalR2(ConditionalPowerLaw):
    def __init__(self, name=None, latex_label=None, minimum=0, maximum=1):
        super(ConditionalPowerLaw, self).__init__(
            alpha=1, minimum=0, maximum=1, name=name, latex_label=latex_label,
            unit=None, boundary=None, condition_func=self._condition_function
        )
        self._required_variables = ["chi_eff", "mass_ratio", "cxd"]

    def _condition_function(self, reference_params, **kwargs):
        xeff = kwargs[self._required_variables[0]]
        q = kwargs[self._required_variables[1]]
        cxd = kwargs[self._required_variables[2]]
        s1z_min, s1z_max = s1z_lim(xeff, q)
        s1z = get_s1z_from_limits(s1z_min, s1z_max, cxd)
        s2z = get_s2z(xeff, q, s1z)
        return dict(alpha=1, minimum=0, maximum=1 - s2z ** 2)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict

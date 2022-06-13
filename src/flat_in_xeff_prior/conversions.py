import numpy as np

from typing import List

from scipy.optimize import minimize
from bilby.core.prior import ConditionalPowerLaw, Prior



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
    xeff, q, cxd = parameters["xeff"], parameters["q"], parameters["cxd"]
    p = parameters.copy()
    s1z_min, s1z_max = s1z_lim(xeff, q)
    p["s1z"] = get_s1z_from_limits(s1z_min, s1z_max, cxd)
    p["s2z"] = get_s2z(xeff, q, p["s1z"])

    p["R1"] = 1 - p["s1z"] ** 2
    p["R2"] = 1 - p["s2z"] ** 2

    p["s1x"] = calc_chix(phi=p["phi1"], xy_mag=p["r1"])
    p["s1y"] = calc_chiy(phi=p["phi1"], xy_mag=p["r1"])

    p["s2x"] = calc_chix(phi=p["phi2"], xy_mag=p["r2"])
    p["s2y"] = calc_chiy(phi=p["phi2"], xy_mag=p["r2"])

    p["a1"] = np.sqrt(sqr_sum(p["s1x"], p["s1y"], p["s1z"]))  # r1 <= 1
    p["a2"] = np.sqrt(sqr_sum(p["s2x"], p["s2y"], p["s2z"]))  # r1 <= 1

    p["cos1"] = p["s1z"] / p["a1"]
    p["cos2"] = p["s2z"] / p["a2"]
    return p


class ConditionalR1(ConditionalPowerLaw):
    def __init__(self, name=None, latex_label=None, minimum=0, maximum=1):
        super(ConditionalPowerLaw, self).__init__(
            alpha=1, minimum=0, maximum=1, name=None, latex_label=None,
            unit=None, boundary=None, condition_func=self._condition_function
        )
        self._required_variables = ["xeff", "q", "cxd"]

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
        self._required_variables = ["xeff", "q", "cxd"]

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

import numpy as np
from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from bilby.gw.source import _base_lal_cbc_fd_waveform
from bilby.gw.utils import (lalsim_GetApproximantFromString,
                            lalsim_SimInspiralChooseFDWaveform,
                            lalsim_SimInspiralChooseFDWaveformSequence,
                            lalsim_SimInspiralFD,
                            lalsim_SimInspiralWaveformParamsInsertTidalLambda1,
                            lalsim_SimInspiralWaveformParamsInsertTidalLambda2)



from .conversions import flatxeff_params_to_lal

def lal_binary_black_hole_flat_in_xeff(
frequency_array,
mass_ratio, chirp_mass, chi_eff, cxd, phi_1, phi_2, r1, r2, luminosity_distance, dec, ra, theta_jn, psi,phase,
**kwargs
):
    kwargs = add_standard_params(kwargs)
    kwargs.update(dict(catch_waveform_errors=True))
    params = dict(mass_ratio=mass_ratio, chirp_mass=chirp_mass, chi_eff=chi_eff, cxd= cxd, phi_1=phi_1, phi_2=phi_2, r1=r1, r2=r2, luminosity_distance=luminosity_distance, dec=dec, ra=ra, theta_jn= theta_jn, psi=psi,phase=phase)
    new_params = flatxeff_params_to_lal(params)
    return _base_lal_cbc_fd_waveform(
     frequency_array=frequency_array, **new_params, **waveform_kwargs)

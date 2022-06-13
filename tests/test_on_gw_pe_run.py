import numpy as np
import bilby
import unittest
import os, shutil
from bilby.gw.prior import CBCPriorDict

from flat_in_xeff_prior.conversions import flat_in_xeff_conversion_fn, flatxeff_params_to_lal


class PERunnerTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "out_prior"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_run(self):
        bilby.core.utils.setup_logger(outdir=self.outdir, label="test")
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=4.0,
            sampling_frequency=2048.0,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                waveform_approximant="IMRPhenomPv2",
                reference_frequency=50.0,
                minimum_frequency=20.0,
            )
        )


        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=2048.0,
            duration=4.0,
            start_time=0,
        )


        priors = CBCPriorDict(
            filename="tests/flat_xeff.prior",
            conversion_function=flat_in_xeff_conversion_fn,
        )
        priors["geocent_time"] = bilby.core.prior.Uniform(
            minimum=- 0.1,
            maximum= + 0.1,
            name="geocent_time",
            latex_label="$t_c$",
            unit="$s$",
        )
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator
        )
        print(waveform_generator.source_parameter_keys)

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            npoints=1,
            outdir=self.outdir,
            label="test",
            dlogz=100,

        )

        # Make a corner plot.
        result.plot_corner()


if __name__ == "__main__":
    unittest.main()

from bilby.gw.prior import CBCPriorDict
from corner import corner
import pandas as pd
from flat_in_xeff_prior.conversions import flat_in_xeff_conversion_fn
import unittest
import os, shutil


class PriorTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "out_prior"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_prior(self):
        prior = CBCPriorDict(
            filename="tests/flat_xeff.prior",
            conversion_function=flat_in_xeff_conversion_fn,
        )
        samples = pd.DataFrame(prior.sample(10000))
        samples = flat_in_xeff_conversion_fn(samples)
        fig = corner(samples[["chi_eff", "mass_ratio"]])
        fig.savefig(f"{self.outdir}/flat_in_xeff_samples.png")



if __name__ == "__main__":
    unittest.main()

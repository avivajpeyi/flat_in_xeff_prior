from bilby.gw.prior import CBCPriorDict
from corner import corner
import pandas as pd
from flat_in_xeff_prior.conversions import flat_in_xeff_conversion_fn



def test_prior():
    prior = CBCPriorDict(
        filename='tests/flat_xeff.prior',
        conversion_function=flat_in_xeff_conversion_fn
    )
    samples = pd.DataFrame(prior.sample(10000))
    fig = corner(
        samples[['xeff', 'q', 'chirp_mass']]
    )
    fig.savefig("flat_in_xeff_samples.png")

if __name__ == '__main__':
    unittest.main()

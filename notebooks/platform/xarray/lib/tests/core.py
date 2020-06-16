import numpy as np
import numpy.testing as npt

from lib import core

def test_convert_probabilities_to_dosages():
    # Simulate allele probability array with dims (variant, sample, ploidy, allele)
    gp = np.random.rand(7, 4, 3)
    # Convert random values to probabilities so that the sum of values across genotypes is 1
    gp /= gp.sum(axis=-1, keepdims=True)
    
    # Convert to xarray probability dataset
    gp = core.create_genotype_probability_alt_dataset(gp, attrs={'description': 'Genotype probability example'})

    # Convert to xarray dosage dataset
    gpd = gp.to.genotype_dosage_dataset()

    # Check dosage is as expected
    assert gpd.data.shape == (7, 4)
    npt.assert_almost_equal(gpd.data.values[0, 0], gp.data.values[0, 0, 1] + 2 * gp.data.values[0, 0, 2])


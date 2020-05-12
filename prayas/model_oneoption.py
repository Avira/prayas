"""
Module model_oneoption
----------------------
"""

import copy
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import beta

from string import ascii_uppercase

from .model_conversion import *


class OneOptionModel(ConversionModel):
    """
    Class for one option models

    This model is applicable for scenarios where a trial consists of
    only one option to choose from. The prior distribution of the
    conversion rate follows a Beta(a, b) distribution.

    :param variants: Integer defining the number of variants
        or a list variant names
    :param baseline: Baseline variant
    :param prior_alpha: Hyperparameter a of the prior distribution
    :param prior_beta: Hyperparameter b of the prior distribution

    References
    ----------

    "Bayesian A/B Testing for Business Decisions" by Shafi Kamalbasha
    and Manuel J. A. Eugster, 2020. https://arxiv.org/abs/2003.02769
    """

    def __init__(self, variants, baseline=None, prior_alpha=1, prior_beta=1):
        assert (isinstance(variants, int)) or \
               (isinstance(variants, list) and all(isinstance(x, str) for x in variants))

        super(OneOptionModel, self).__init__()

        if isinstance(variants, int):
            n = variants
            variants = ["Variant " + s for s in list(ascii_uppercase)[0:n]]
        else:
            n = len(variants)

        assert n == len(variants)
        assert baseline in variants

        self._model_type = "One option model"
        self.size = n
        self.variants = variants
        self.baseline = baseline
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.add_measure("conversion", np.resize(1, n))

    def add_measure(self, name, success_value, nonsuccess_value=None):
        if nonsuccess_value is None:
            nonsuccess_value = np.resize(0, self.size)

        assert len(success_value) == self.size
        assert len(success_value) == len(nonsuccess_value)

        self.weight_conversions.update({name: success_value})
        self.weight_nonconversions.update({name: nonsuccess_value})
        self.measures.append(name)

    def set_result(self, successes, trials):
        assert len(successes) == len(trials)
        assert len(successes) == len(self.variants)

        self.posteriors = [None] * len(successes)
        for i, (conv, vis) in enumerate(zip(successes, trials)):
            self.posteriors[i] = beta(self.prior_alpha + conv, self.prior_beta + vis - conv)

        self.conversions = successes
        self.visitors = trials

    def new(self, successes, trials):
        assert len(successes) == len(trials)
        assert len(successes) == len(self.variants)

        m = copy.deepcopy(self)
        m.set_result(successes, trials)

        return m

    def sample(self, measure=None, n=20000):
        if measure is None:
            measure = 'conversion'

        samples = [None] * self.size
        for i, post in enumerate(self.posteriors):
            s = post.rvs(n)
            samples[i] = s * self.weight_conversions[measure][i] + \
                (1-s) * self.weight_nonconversions[measure][i]

        return samples

    def measure(self, measure=None):
        if measure is None:
            measure = 'conversion'

        cr = (np.array(self.conversions) / np.array(self.visitors))

        df = pd.DataFrame(cr * np.array(self.weight_conversions[measure]) +
                          cr * np.array(self.weight_nonconversions[measure]))
        df.columns = [measure]
        df.index = self.variants

        return df

    def plot(self, n=20000):
        xlim = max(np.divide(self.conversions, self.visitors)) * 2
        x = np.linspace(0, xlim, 5000)

        for measure in self.measures:
            samples = self.sample(measure, n)
            plt.figure()
            for i, samp in enumerate(samples):
                sns.distplot(samp, hist=False, label=self.variants[i])
            plt.title(measure)
            plt.legend()

        return plt


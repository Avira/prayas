"""
Module model_multioptions
-------------------------
"""

import copy
import seaborn as sns

from matplotlib import pyplot as plt
from string import ascii_uppercase
from scipy.stats import dirichlet

from .model_conversion import *


class MultiOptionsModel(ConversionModel):
    """
    Class for multi-options models

    This model is applicable for scenarios where a trial consists of
    one or more options to choose from. The prior distribution of the
    conversion rate follows a Dirichlet(a) distribution.

    :param variants: Integer defining the number of variants
        or a list variant names
    :param baseline: Baseline variant
    :param options: List of integers or list of names defining
        the individual options per variant
    :param priors: Hyperparameters of the prior distribution

    References
    ----------

    "Bayesian A/B Testing for Business Decisions" by Shafi Kamalbasha
    and Manuel J. A. Eugster, 2020. https://arxiv.org/abs/2003.02769
    """

    def __init__(self, variants, options, baseline=None, priors=1):
        """
        :param variants: Either an integer defining the number of variants or a list of strings with the variant names
        :param options: Either a list of integers defining the number of options per variant or a list of list of
            strings defining the individual option names
        :param priors:
        """
        assert isinstance(variants, int) or \
                (isinstance(variants, list) and all(isinstance(x, str) for x in variants))

        assert isinstance(options, list) and \
                (all(isinstance(x, int) for x in options) or
                 all(isinstance(flatten_list(x), str) for x in options))

        super(MultiOptionsModel, self).__init__()

        if isinstance(variants, int):
            n = variants
            variants = ["Variant " + s for s in list(ascii_uppercase)[0:n]]
        else:
            n = len(variants)

        assert n == len(variants)
        assert baseline in variants

        if isinstance(list(flatten_list(options))[0], str):
            raise NotImplementedError

        self._model_type = "Multi-options model"
        self.size = n
        self.variants = variants
        self.baseline = baseline
        self.priors = [np.resize(priors, options[i]+1) for i in range(0, n)]
        self.add_measure("conversion", [np.resize(1, options[i]) for i in range(0, n)])

    def add_measure(self, name, success_value, nonsuccess_value=None):

        if nonsuccess_value is None:
            nonsuccess_value = [(np.resize(0, len(n))) for i, n in enumerate(success_value)]

        assert len(success_value) == self.size
        assert len(success_value) == len(nonsuccess_value)

        self.weight_conversions.update({name: success_value})
        self.weight_nonconversions.update({name: nonsuccess_value})
        self.measures.append(name)

    def set_result(self, successes, trials):
        assert len(successes) == len(trials)
        assert len(successes) == len(self.variants)

        self.posteriors = [None] * len(successes)

        for i, (conv, vis, pri) in enumerate(zip(successes, trials, self.priors)):
            obs = conv + [vis - sum(conv)]
            self.posteriors[i] = dirichlet(pri + obs)

        self.conversions = successes
        self.visitors = trials

    def new(self, successes, trials):
        assert len(successes) == len(trials)
        assert len(successes) == len(self.variants)

        priors = self.priors
        m = copy.deepcopy(self)
        m.set_result(successes, trials)

        return m

    def sample(self, measure=None, n=20000):
        if measure is None:
            measure = 'conversion'

        samples = [None] * self.size
        for i, post in enumerate(self.posteriors):
            s = post.rvs(n)
            post_value = np.zeros(n)
            for j in range(0, s.shape[1]-1):
                post_value = post_value + (s[:, j] * self.weight_conversions[measure][i][j]) + \
                             ((1-s[:, j]) * self.weight_nonconversions[measure][i][j])

            samples[i] = post_value

        return samples

    def measure(self, measure=None):
        if measure is None:
            measure = 'conversion'

        #cr = (np.array([np.sum(c) for c in self.conversions]) / np.array(self.visitors))
        cr = np.divide([np.sum([c*w_c + c*w_nc for c, w_c, w_nc in zip(conv, weight_conv, weight_nonconv)])
                                            for conv, weight_conv, weight_nonconv in
                                            zip(self.conversions,
                                                self.weight_conversions[measure],
                                                self.weight_nonconversions[measure])], np.array(self.visitors))

        df = pd.DataFrame(cr)
        df.columns = [measure]
        df.index = self.variants
        return df

    def plot(self, n=20000):
        xlim = max(np.divide([np.sum(c) for c in self.conversions], self.visitors)) * 2
        x = np.linspace(0, xlim, 5000)

        for measure in self.measures:
            samples = self.sample(measure, n)
            plt.figure()
            for i, samp in enumerate(samples):
                sns.distplot(samp, hist=False, label=self.variants[i])
            plt.title(measure)
            plt.legend()

        return plt

    def plot_options(self, measure=None, n=20000):
        """
        Plot the curves of the individual options.

        :param measure: Name of the measure; default is `conversion`
        :param n: Number of samples to draw
        :return: Matplotlib pyplot object
        """

        if measure is None:
            measure = 'conversion'

        for i, post in enumerate(self.posteriors):
            s = post.rvs(n)
            k = s.shape[1]-1

            plt.figure()
            for j in range(0, k):
                d = (s[:, j] * self.weight_conversions[measure][i][j])
                sns.distplot(d, label="Option " + str(j))
            plt.title(self.variants[i])
            plt.legend()

        return plt

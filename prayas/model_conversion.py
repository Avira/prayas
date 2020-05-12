"""
Module model_conversion
-----------------------
"""

import pandas as pd

from textwrap import dedent
from abc import ABC, abstractmethod

from .utils import *


class ConversionModel(ABC):
    """
    Base class for conversion-based models

    This class defines the basic properties each conversion-based model
    needs to define as well as methods that either are common to all
    conversion-based models or need to be implemented specifically.
    """

    def __init__(self):
        self.posteriors = []
        self.size = len(self.posteriors)
        self.conversions = None
        self.visitors = None
        self.weight_conversions = {}
        self.weight_nonconversions = {}
        self.variants = None
        self.measures = []

        self._model_type = None
        self._baseline = None
        self._primary = "conversion"
        self._loss_threshold = 0.05

    def __str__(self):
        s = f"""\
            {self._model_type}
            Variants              : {", ".join(self.variants)}
            Baseline              : {self._baseline}
            Measures              : {", ".join(self.measures)}
            Primary measure       : {self._primary}
            Maximum loss threshold: {self._loss_threshold}\
            """
        return dedent(s)

    @property
    def baseline(self):
        """Returns or sets the baseline variant"""
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        assert value in self.variants
        self._baseline = value

    @property
    def primary_measure(self):
        """Returns or sets the primary measure; default is `conversion`"""
        return self._primary

    @primary_measure.setter
    def primary_measure(self, value):
        assert value in list(self.weight_conversions.keys())
        self._primary = value

    @property
    def loss_threshold(self):
        """Returns or sets the loss threshold; default is 0.05"""
        return self._loss_threshold

    @loss_threshold.setter
    def loss_threshold(self, value):
        self._loss_threshold = value

    # Functions that are extended in subclass: ######################

    @abstractmethod
    def add_measure(self, name, success_value, nonsuccess_value=None):
        """
        Method to add an additional measure

        :param name: Name of the measure, e.g., revenue
        :param success_value: Value in case of success for each variant and option
        :param nonsuccess_value: Value in case of nonsuccess for each variant and option
        :return: None, used for its side-effect
        """
        pass

    @abstractmethod
    def sample(self, measure=None, n=20000):
        """
        Sample from the posterior distribution of each variant

        :param measure: Name of the measure; default is `conversion`
        :param n: Number of samples to draw
        :return: Array with samples for each variant
        """
        pass

    @abstractmethod
    def measure(self, measure=None):
        """
        Return the value of a specific measure

        :param measure: Name of the measure; default is `conversion`
        :return: Data frame with measure per variant
        """
        pass

    @abstractmethod
    def plot(self, n=20000):
        """
        Plot the posterior distribution of each variant

        :param n: Number of samples to draw
        :return: Matplotlib pyplot object
        """
        pass

    @abstractmethod
    def set_result(self, successes, trials):
        """
        Set the result of the experiment

        :param successes: Successes per variant and option
        :param trials: Trials per variant and option
        :return: None, used for its side-effect
        """
        pass

    @abstractmethod
    def new(self, successes, trials):
        """
        Create a new ConversionModel object, used for experiments

        :param successes: Successes per variant and option
        :param trials: Trials per variant and option
        :return: ConversionModel object
        """
        pass

    # Pairwise: #####################################################

    def simulate_pairs(self, fn, measure=None, n=20000):
        samples = self.sample(measure, n)
        grid = comb2(list(range(self.size)))

        mat = np.array([fn(x, samples) for x in grid])

        df = pd.DataFrame(mat.reshape([self.size, self.size]))
        df.columns = self.variants
        df.index = self.variants

        return df

    def prob_pairwise(self, measure=None, n=20000):
        fn = lambda x, samples: (samples[x[1]] > samples[x[0]]).mean()
        return self.simulate_pairs(fn, measure=measure, n=n)

    def lift_pairwise(self, measure=None, n=20000):
        fn = lambda x, samples: (samples[x[1]].mean() - samples[x[0]].mean()) / samples[x[0]].mean()
        return self.simulate_pairs(fn, measure=measure, n=n)

    def loss_pairwise(self, measure=None, n=20000):
        def loss_fn(x, samples):
            prob_error = np.zeros(n)
            prob_error[np.where(samples[x[0]] > samples[x[1]])] = 1.0
            return np.mean(prob_error * np.maximum(abs(samples[x[0]]-samples[x[1]]), 0.0))/samples[x[0]].mean()
        return self.simulate_pairs(loss_fn, measure=measure, n=n)

    def score_pairwise(self, n=20000, drop=True):
        """
        Compute pairwise comparisons between all variants

        :param n: Number of samples from posteriors as basis for comparison
        :param drop: If `True` show only entries with `Uplift > 0`
        :return: Data frame with

            * Left: First variant of the pairwise comparison
            * Right: Second variant
            * Measure: Comparison measure
            * LeftMeasure: Value of left
            * RightMeasure: Value of right
            * Uplift: Uplift of left in percentage
            * Loss: Potential loss in percentage if we go with left
            * ProbabilityUplift: Probability of the uplift
            * ProbabilityLoss: Probability of the loss if we go with left
            * LeftMeasureMaxLoss: Maximum loss if we go with left (subtracted from LeftMeasure)
            * Score: Score based on uplift * probability of the uplift
        """

        ret = []

        for measure in self.measures:
            prob = self.prob_pairwise(measure=measure, n=n)
            lift = self.lift_pairwise(measure=measure, n=n)
            loss = self.loss_pairwise(measure=measure, n=n)
            kpi = self.measure(measure=measure)

            # Gain:  In {measure}, {left} is to {prob}% better than {right} with an uplift of {uplift}%.
            # Measure: On average we can expect {left_measure} {measure} for {right} and {left_measure} {measure}
            #   for {left} per visitor.
            # Risk: The risk of going with {left} is a maximum drop of {loss}% with {1-prob}% probability, resulting
            #   in an average of {left_measure_max_loss} {measure}.

            for i in prob.columns:
                for j in prob.index:
                    ret.append({'Left': i,
                                'Right': j,
                                'Measure': measure,
                                'LeftMeasure': kpi.loc[i, measure],
                                'RightMeasure': kpi.loc[j, measure],
                                'Uplift': lift.loc[i, j]*100,
                                'ProbabilityUplift': prob.loc[i, j],
                                'ProbabilityLoss': (1 - prob.loc[i, j]),
                                'Loss': loss.loc[i, j]*100,
                                'LeftMeasureMaxLoss': kpi.loc[i, measure] - (kpi.loc[i, measure] * loss.loc[i, j]),
                                'Score': (lift.loc[i, j] * 100) * (prob.loc[i, j] * 100)
                                })

        df = pd.DataFrame.from_records(ret).\
            query("@drop == False | Uplift > 0").\
            sort_values(["Score"], ascending=False).\
            reset_index(drop=True)

        return df

    # Baseline: #####################################################

    def simulate_matrix(self, fn, measure=None, n=20000):
        samples = self.sample(measure, n)
        grid = comb_mat(list(range(self.size)))

        mat = np.array([fn(x, samples) for x in grid])

        return pd.Series(mat, index=self.variants)

    def bayes_factor(self, measure=None, n=20000):
        def fn(x, samples):
            res = np.asarray([True] * n)
            for i in range(1, len(x)):
                res = (res & (samples[x[0]] > samples[x[i]]))
            return np.sum(res)/(n-np.sum(res))

        return self.simulate_matrix(fn, measure=measure, n=n)

    def prob_best(self, measure=None, n=20000):
        def fn(x, samples):
            res = np.asarray([True] * n)
            for i in range(1, len(x)):
                res = (res & (samples[x[0]] > samples[x[i]]))
            return res.mean()

        return self.simulate_matrix(fn, measure=measure, n=n)

    def loss_overall(self, measure=None, n=20000):
        def fn(x, samples):
            loss_error = []
            for i in range(1, len(x)):
                prob_error = np.zeros(n)
                prob_error[np.where(samples[x[i]] > samples[x[0]])] = 1.0
                loss_error.append(abs(np.mean(prob_error * np.maximum(abs(samples[x[i]] - samples[x[0]]), 0.0)) /
                                      samples[x[i]].mean()))
            return max(loss_error)

        return self.simulate_matrix(fn, measure=measure, n=n)

    def lift_overall(self, measure=None, n=20000):
        def fn(x, samples):
            lift_from_all = []
            for i in range(1, len(x)):
                lift_from_all.append((samples[x[0]].mean() - samples[x[i]].mean()) / samples[x[i]].mean())

            return max(lift_from_all)

        return self.simulate_matrix(fn, measure=measure, n=n)

    def prob_baseline(self, measure=None, n=20000):
        def fn(x, samples):
            baseline_idx = list(x).index(self.variants.index(self.baseline))
            return (samples[x[0]] > samples[x[baseline_idx]]).mean()

        return self.simulate_matrix(fn, measure=measure, n=n)

    def lift_baseline(self, measure=None, n=20000):
        def fn(x, samples):
            baseline_idx = list(x).index(self.variants.index(self.baseline))
            return (samples[x[0]].mean() - samples[x[baseline_idx]].mean()) / samples[x[baseline_idx]].mean()

        return self.simulate_matrix(fn, measure=measure, n=n)

    def loss_baseline(self, measure=None, n=20000):
        def fn(x, samples):
            baseline_idx = list(x).index(self.variants.index(self.baseline))
            prob_error = np.zeros(n)
            prob_error[np.where(samples[x[baseline_idx]] > samples[x[0]])] = 1.0
            return abs(np.mean(prob_error * np.maximum(abs(samples[x[baseline_idx]] - samples[x[0]]), 0.0)) / samples[
                x[baseline_idx]].mean())

        return self.simulate_matrix(fn, measure=measure, n=n)

    def score_baseline(self, n=20000):
        """
        Compute baseline comparisons for all variants

        :param n: Number of samples from posteriors as basis for comparison
        :return: Data frame with

            * Variant: Name of the variant
            * Measure: Comparison measure
            * ProbabilityToBeBest: Probability to be the best among all variants
            * ProbabilityToBeatBaseline: Probability to be better than the baseline
            * UpliftFromBaseline: Uplift from baseline in percentage
            * PotentialLossFromBaseline: Potential loss if we go with the variant in percentage
            * MaxUplift: Maximum uplift in percentage
            * MaxPotentialLoss: Maximum loss in percentage
        """
        
        assert(self.baseline is not None)

        ret = []

        for measure in self.measures:
            prob_best = self.prob_best(measure=measure, n=n)
            prob_bl = self.prob_baseline(measure=measure, n=n)
            lift_bl = self.lift_baseline(measure=measure, n=n)
            loss_bl = self.loss_baseline(measure=measure, n=n)
            lift_all = self.lift_overall(measure=measure, n=n)
            loss_all = self.loss_overall(measure=measure, n=n)

            for i in prob_best.index:
                ret.append({'Variant': i,
                            'Measure': measure,
                            'ProbabilityToBeBest': prob_best[i],
                            'ProbabilityToBeatBaseline': prob_bl[i],
                            'UpliftFromBaseline': lift_bl[i]*100,
                            'PotentialLossFromBaseline': loss_bl[i]*100,
                            'MaxUplift': lift_all[i]*100,
                            'MaxPotentialLoss': loss_all[i]*100
                            })
        df = pd.DataFrame.from_records(ret). \
            sort_values(["ProbabilityToBeBest"], ascending=False). \
            reset_index(drop=True)

        return df

    def decision(self, n=20000, ignore_primary=False, ignore_loss_threshold=None):
        """
        Filter the result from the `score_baseline` method based on
        the defined experiment setup, i.e., the primary measure and
        the loss threshold

        :param n: Number of samples from posteriors as basis for comparison
        :param ignore_primary: If `True` ignores the primary measure filter
        :param ignore_loss_threshold: If `True` ignores the loss threshold
        :return: Data frame as returned by `score_baseline` with filtered rows
        """

        score = self.score_baseline(n=n)
        score = score.iloc[score.groupby("Measure")["ProbabilityToBeBest"].idxmax()]

        if not ignore_primary and self.primary_measure is not None:
            score = score.query("Measure == @self.primary_measure")
        if not ignore_loss_threshold and self.loss_threshold is not None:
            score = score.query("MaxPotentialLoss < @self.loss_threshold")

        return score

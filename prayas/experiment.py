"""
Module experiment
-----------------
"""

import seaborn as sns

from matplotlib import pyplot as plt
from collections import OrderedDict
from textwrap import dedent

from .model_conversion import *


class Experiment:
    """
    Class for experiments

    Experiments provide a simple way to continuously monitor the
    experiment performance and to decide when to stop based on the
    maximum potential loss of the variants.

    :param name: Name of the experiment
    """

    def __init__(self, name):
        self.name = name
        self.setup = None
        self.models = OrderedDict()

    def __str__(self):
        s = f"""\
            Experiment with a {self.setup._model_type}
            Variants              : {", ".join(self.setup.variants)}
            Baseline              : {self.setup._baseline}
            Measures              : {", ".join(self.setup.measures)}
            Primary measure       : {self.setup._primary}
            Maximum loss threshold: {self.setup._loss_threshold}\
            """
        return dedent(s)

    @property
    def result(self):
        """Get the latest result of the experiment"""
        k = next(reversed(self.models))
        return self.models[k]

    def add_data(self, ts, successes, trials):
        """
        Add data to the experiment

        :param ts: Timestamp of the data, can be any object
        :param successes: Successes per variant and option
        :param trials: Trials per variant and option
        :return: None, used for its side-effect
        """
        if len(self.models) > 0:
            assert isinstance(ts, type(list(self.models.keys())[0]))

        self.models.update({ts: self.setup.new(successes, trials)})

    def monitor_loss(self):
        loss = pd.DataFrame()
        for n in self.setup.measures:
            l = pd.DataFrame([dict(self.models[m].loss_overall(measure=n)) for m in self.models]) * 100
            l["Measure"] = n
            l["Timestamp"] = self.models.keys()
            loss = loss.append(l)

        return loss

    def monitor_decision(self, days=5):
        """
        Compute the decision to continue or stop the experiment.

        :param days: Number of days to be lower than threshold
        :return: Data frame with the daily decision to continue or stop
            based on the estimated loss
        """
        loss = self.monitor_loss(). \
            query("Measure == @self.setup.primary_measure").\
            drop("Measure", axis=1)

        ts = loss["Timestamp"]
        loss = loss.\
            drop("Timestamp", axis=1)

        runs = loss < self.setup.loss_threshold
        runs = runs.rolling(days).sum()

        max = loss.rolling(days).max()

        stop = runs == days

        loss.columns = "Loss " + loss.columns
        runs.columns = "Runs " + runs.columns
        max.columns = "Max loss " + max.columns
        stop.columns = "Stop " + stop.columns

        df = pd.concat([ts, loss.reset_index(drop=True),
                        runs.reset_index(drop=True),
                        max.reset_index(drop=True),
                        stop.reset_index(drop=True)], axis=1)

        return df

    def monitor_plot(self):
        """
        Plot the loss of the variants.

        :return: Seaborn FacetGrid object
        """

        loss = self.monitor_loss()
        loss = pd.melt(loss, id_vars=["Timestamp", "Measure"],
                       value_vars=self.setup.variants,
                       var_name="Variant", value_name="Value")

        p = sns.relplot(x="Timestamp", y="Value", hue="Variant", row="Measure", kind="line", data=loss)
        for ax in p.axes:
            ax[0].axhline(self.setup.loss_threshold, linestyle="--", color="black")

        return p

    def monitor_score_baseline(self):
        """
        Compute the scoring against the baseline for each timestamp
        of the experiment.

        :return: Data frame with baseline scoring for
            each timestamp of the experiment
        """
        def sc(ts):
            s = self.models[ts].score_baseline()
            s.insert(loc=0, column="Date", value=ts)
            return s

        res = [sc(ts) for ts in self.models]
        res = pd.\
            concat(res).\
            reset_index(drop=True)

        return res

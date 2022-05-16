import pymc3 as pm
import pandas as pd

class BayesFeatures:
    def __init__(self, target, data, priors = None):
        self.target = target
        self.data = data
        if priors == None:
            cols = list(data.columns)
            cols.remove(target)
            priors = {}
            for i in cols:
                priors.update({i: pm.Normal.dist(mu = 0, sigma = 1)})
        self.priors = priors
        self.formula = f'{target} ~ '
        for i in data.columns:
            self.formula = f'{self.formula} + {i}'

    def sample(self):
        with pm.Model() as model:
            pm.glm.GLM.from_formula(formula=self.formula,
                                    data=self.data,
                                    family=pm.glm.families.Binomial(),
                                    priors = self.priors)
            self.trace = pm.sample(draws=4000, cores = 2, tune = 1000)
            self.summary = pm.summary(self.trace)
            self.trace_plot = pm.plot_trace(self.trace)

    def plot_posteriors(self):
        return self.trace_plot

    def get_posterior_stats(self):
        return self.summary

    def create_new_priors(self):
        prior_dict = {}
        for i in range(1, self.summary.shape[0] - 1):
            prior_dict.update({self.summary.index[i]: pm.Normal.dist(mu = self.summary.iloc[i,0], 
                                                                     sigma = self.summary.iloc[i,1])})
        self.prior_dict = prior_dict
        return prior_dict


"""
Copyright (C) 2020  Patrick Schwab, F. Hoffmann-La Roche Ltd
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Cameron Davidson-Pilon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -*- coding: utf-8 -*-


from datetime import datetime
import warnings
import time

import os
import numpy as np
import pandas as pd

from numpy import sum as array_sum_to_scalar
from autograd import elementwise_grad
from autograd import numpy as anp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from covews.apps.util import info


from lifelines.fitters import SemiParametricRegressionFittter
from lifelines.fitters.mixins import ProportionalHazardMixin
from lifelines.utils.printer import Printer
from lifelines.utils import (
    _get_index,
    _to_list,
    # check_for_overlapping_intervals,
    check_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation_low_variance,
    check_for_immediate_deaths,
    check_for_instantaneous_events_at_time_zero,
    check_for_instantaneous_events_at_death_time,
    check_for_nonnegative_intervals,
    pass_for_numeric_dtypes_or_raise_array,
    ConvergenceError,
    normalize,
    StepSizer,
    check_nans_or_infs,
    string_justify,
    coalesce,
)


__all__ = ["CoxNonLinearTimeVaryingFitter"]

matrix_axis_0_sum_to_1d_array = lambda m: np.sum(m, 0)


class CoxNonLinearTimeVaryingFitter(SemiParametricRegressionFittter, ProportionalHazardMixin):
    r"""
    This class implements fitting Cox's nonlinear time-varying proportional hazard model:
        .. math::  h(t|x(t)) = h_0(t)\exp((x(t)-\overline{x})'\beta)
    Parameters
    ----------
    learning_rate: float, optional (default=0.05)
       the level in the confidence intervals.
    l2_weight: float, optional
        the coefficient of an L2 penalizer in the regression
    Attributes
    ----------
    params_ : Series
        The estimated coefficients. Changed in version 0.22.0: use to be ``.hazards_``
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """
    _KNOWN_MODEL = True

    def __init__(self, learning_rate=0.001, l2_weight=0.0, l1_ratio=0.0, strata=None, num_units=32, num_layers=2,
                 p_dropout=0.0, batch_size=128, num_epochs=100, output_directory=""):
        super(CoxNonLinearTimeVaryingFitter, self).__init__(alpha=learning_rate)
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.strata = strata
        self.l1_ratio = l1_ratio
        self.num_units = num_units
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.type_pt = torch.float
        self.tmp_file_name = "model.tmp.pt"
        self.output_directory = output_directory

    def preprocess_df(self, df, event_col, start_col, stop_col, weights_col, id_col):
        df = df.copy()

        if not (event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the DataFrame provided.")

        if weights_col is None:
            self.weights_col = None
            assert "__weights" not in df.columns, "__weights is an internal lifelines column, please rename your column first."
            df["__weights"] = 1.0
        else:
            self.weights_col = weights_col
            if (df[weights_col] <= 0).any():
                raise ValueError("values in weights_col must be positive.")

        df = df.rename(columns={event_col: "event", start_col: "start", stop_col: "stop", weights_col: "__weights"})
        if self.strata is not None and self.id_col is not None:
            df = df.set_index(_to_list(self.strata) + [id_col])
            df = df.sort_index()
        elif self.strata is not None and self.id_col is None:
            df = df.set_index(_to_list(self.strata))
        elif self.strata is None and self.id_col is not None:
            df = df.set_index([id_col])

        events, start, stop = (
            pass_for_numeric_dtypes_or_raise_array(df.pop("event")).astype(bool),
            df.pop("start"),
            df.pop("stop"),
        )
        weights = df.pop("__weights").astype(float)

        df = df.astype(float)
        self._check_values(df, events, start, stop)
        return df, events, start, stop, weights

    def fit(
        self,
        df,
        event_col,
        start_col="start",
        stop_col="stop",
        weights_col=None,
        id_col=None,
        show_progress=False,
        robust=False,
        strata=None,
        initial_point=None,
        val_df=None
    ):  # pylint: disable=too-many-arguments
        """
        Fit the Cox Nonlinear Hazard model to a time varying dataset. Tied survival times
        are handled using Efron's tie-method.
        Parameters
        -----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
           `event_col`, plus other covariates. `duration_col` refers to
           the lifetimes of the subjects. `event_col` refers to whether
           the 'death' events was observed: 1 if observed, 0 else (censored).
        event_col: string
           the column in DataFrame that contains the subjects' death
           observation. If left as None, assume all individuals are non-censored.
        start_col: string
            the column that contains the start of a subject's time period.
        stop_col: string
            the column that contains the end of a subject's time period.
        weights_col: string, optional
            the column that contains (possibly time-varying) weight of each subject-period row.
        id_col: string, optional
            A subject could have multiple rows in the DataFrame. This column contains
           the unique identifier per subject. If not provided, it's up to the
           user to make sure that there are no violations.
        show_progress: since the fitter is iterative, show convergence
           diagnostics.
        robust: bool, optional (default: True)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
          ties, so if there are high number of ties, results may significantly differ. See
          "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078
        step_size: float, optional
            set an initial step size for the fitting algorithm.
        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        Returns
        --------
        self: CoxNonLinearTimeVaryingFitter
            self, with additional properties like ``hazards_`` and ``print_summary``
        """
        self.strata = coalesce(strata, self.strata)
        self.robust = robust
        if self.robust:
            raise NotImplementedError("Not available yet.")

        self.event_col = event_col
        self.id_col = id_col
        self.stop_col = stop_col
        self.start_col = start_col
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        df, events, start, stop, weights = self.preprocess_df(df, event_col, start_col, stop_col, weights_col, id_col)
        val_df, val_events, val_start, val_stop, val_weights = \
            self.preprocess_df(val_df, event_col, start_col, stop_col, weights_col, id_col)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)
        self._norm_std[self._norm_std == 0] = 1.0  # Avoid div by zero.

        # Network architecture
        in_features = df.values.shape[-1]
        out_features = 1
        self.type_pt = torch.float
        self.net = Net(in_features, self.num_units, out_features, self.num_layers, self.p_dropout, self.type_pt)
        self.net = self._neural_cox(
            normalize(df, self._norm_mean, self._norm_std),
            events,
            start,
            stop,
            weights,
            normalize(val_df, self._norm_mean, self._norm_std),
            val_events,
            val_start,
            val_stop,
            val_weights,
            net=self.net,
            show_progress=show_progress,
            training_epochs=self.num_epochs,
            batch_size=self.batch_size,
            step_size=self.learning_rate,
        )

        self.beta_params_ = pd.Series(list(self.net.beta.parameters())[0].detach().numpy().ravel(), name="coef")
        self.baseline_cumulative_hazard_ = self._compute_cumulative_baseline_hazard(df, events, start, stop, weights)
        self.baseline_survival_ = self._compute_baseline_survival()
        self.event_observed = events
        self.start_stop_and_events = pd.DataFrame({"event": events, "start": start, "stop": stop})
        self.weights = weights
        self._n_examples = df.shape[0]
        self._n_unique = df.index.unique().shape[0]
        return self

    def _neural_cox(self, X, events, start, stop, weights,
                    val_X, val_events, val_start, val_stop, val_weights, net,
                    show_progress=True, training_epochs=10, batch_size=16, step_size=0.01):
        events = events.values.reshape(-1, 1)
        start = start.values.reshape(-1, 1)
        stop = stop.values.reshape(-1, 1)
        weights = weights.values.reshape(-1, 1)

        val_events = val_events.values.reshape(-1, 1)
        val_start = val_start.values.reshape(-1, 1)
        val_stop = val_stop.values.reshape(-1, 1)
        val_weights = val_weights.values.reshape(-1, 1)

        n, d = X.shape
        val_n, val_d = val_X.shape

        assert d == val_d

        optimizer = optim.AdamW(net.parameters(), lr=step_size, weight_decay=self.l2_weight)

        full_table = np.concatenate([X, events, start, stop, weights], axis=1)
        val_full_table = np.concatenate([val_X, val_events, val_start, val_stop, val_weights], axis=1)

        loader = DataLoader(
            full_table,
            batch_size=len(full_table),
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=True,
            timeout=0,
            worker_init_fn=None,
        )
        val_loader = DataLoader(
            val_full_table,
            batch_size=len(val_full_table),
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=True,
            timeout=0,
            worker_init_fn=None,
        )

        checkpoint_path = os.path.join(self.output_directory, self.tmp_file_name)
        min_loss = np.finfo(float).max
        net = net.train()
        for epoch in range(training_epochs):
            self.log_likelihood_ = 0
            batch_losses = []
            for batch_ndx, batch_data in enumerate(loader):
                X, events, start, stop, weights = (
                    batch_data[:, 0:d],
                    batch_data[:, d],
                    batch_data[:, d + 1],
                    batch_data[:, d + 2],
                    batch_data[:, d + 3],
                )
                X = X.to(self.type_pt)
                weights = weights.to(self.type_pt)
                events = events.to(torch.bool)
                start = start.to(torch.int)
                stop = stop.to(torch.int)

                def closure():
                    optimizer.zero_grad()
                    batch_negloglik = self._get_log_lik(X, events, start, stop, weights, net)
                    if batch_negloglik is None:
                        return torch.zeros(1, requires_grad=False)
                    batch_negloglik = -batch_negloglik
                    batch_negloglik.backward()
                    return batch_negloglik

                batch_negloglik = optimizer.step(closure)
                if batch_negloglik is None:
                    continue
                batch_losses.append(batch_negloglik)

            net = net.eval()

            val_batch_losses = []
            for batch_ndx, batch_data in enumerate(val_loader):
                X, events, start, stop, weights = (
                    batch_data[:, 0:d],
                    batch_data[:, d],
                    batch_data[:, d + 1],
                    batch_data[:, d + 2],
                    batch_data[:, d + 3],
                )
                X = X.to(self.type_pt)
                weights = weights.to(self.type_pt)
                events = events.to(torch.bool)
                start = start.to(torch.int)
                stop = stop.to(torch.int)
                batch_negloglik = self._get_log_lik(X, events, start, stop, weights, net)
                if batch_negloglik is None:
                    continue
                batch_negloglik = -batch_negloglik
                val_batch_losses.append(batch_negloglik)

            self.log_likelihood_ = torch.mean(torch.stack(batch_losses))
            val_loss = torch.mean(torch.stack(val_batch_losses))
            if np.asscalar(val_loss.detach().numpy()) < min_loss:
                min_loss = val_loss
                torch.save({'model_state_dict': net.state_dict()}, checkpoint_path)

            info("Epoch [{:d}] loss={:.4f}, val_loss={:.4f}".format(epoch, self.log_likelihood_, val_loss))
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.eval()
        return net

    def _check_values(self, df, events, start, stop):
        # check_for_overlapping_intervals(df) # this is currently too slow for production.
        check_nans_or_infs(df)
        check_low_var(df)
        check_complete_separation_low_variance(df, events, self.event_col)
        check_for_numeric_dtypes_or_raise(df)
        check_for_nonnegative_intervals(start, stop)
        check_for_immediate_deaths(events, start, stop)
        check_for_instantaneous_events_at_time_zero(start, stop)
        check_for_instantaneous_events_at_death_time(events, start, stop)

    def _partition_by_strata(self, X, events, start, stop, weights):
        for stratum, stratified_X in X.groupby(self.strata):
            stratified_W = weights.loc[stratum]
            stratified_start = start.loc[stratum]
            stratified_events = events.loc[stratum]
            stratified_stop = stop.loc[stratum]
            yield (
                stratified_X.values,
                stratified_events.values,
                stratified_start.values,
                stratified_stop.values,
                stratified_W.values,
            ), stratum

    def _partition_by_strata_and_apply(self, X, events, start, stop, weights, function, *args):
        for ((stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W), _) in self._partition_by_strata(
            X, events, start, stop, weights
        ):
            yield function(stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W, *args)

    def _compute_z_values(self):
        raise NotImplementedError

    def _compute_p_values(self):
        raise NotImplementedError

    def _compute_confidence_intervals(self):
        raise NotImplementedError

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            df = pd.DataFrame()
            return df

    def _get_log_lik(self, X_pt, events_pt, start_pt, stop_pt, weights_pt, net):
        """
        Calculate the pytorch compatibale log-likelihood
        -------
        X: (m, d) tensor of covariates,
        events: (1, d) tensor of events
        start: (1, d) tensor of start times
        stop: (1, d) tensor of stop times
        weights: (1, d) tensor of weight times
        net: the current state of nonlinear link function h(t|x) = h_0(t|x)exp(net(x))
        """
        log_lik = None
        events_pt = events_pt.to(torch.bool)
        unique_death_times = np.unique(stop_pt[events_pt])
        for t in reversed(unique_death_times):
            ix = (start_pt < t) & (t <= stop_pt)
            X_at_t_pt = X_pt[ix]
            weights_at_t_pt = weights_pt[ix][:, None]
            stops_events_at_t_pt = stop_pt[ix]
            events_at_t_pt = events_pt[ix]

            phi_i = weights_at_t_pt * torch.exp(net(X_at_t_pt))
            risk_phi = torch.sum(phi_i, dim=0)  # Calculate sums of risk set.

            deaths_pt = events_at_t_pt & (stops_events_at_t_pt == t)
            deaths = deaths_pt.detach().numpy()
            ties_counts = array_sum_to_scalar(deaths)
            phi_death_sum = torch.sum(phi_i[deaths], dim=0)
            weight_count = torch.sum(weights_at_t_pt[deaths], axis=0)
            weighted_average = weight_count / ties_counts

            if log_lik is None:
                log_lik = torch.zeros(1, requires_grad=False)

            # No tie.
            for l in range(ties_counts):
                if ties_counts > 1:
                    increasing_proportion = l / ties_counts
                    denom = risk_phi - increasing_proportion * phi_death_sum
                else:
                    denom = risk_phi
                log_lik -= weighted_average * torch.log(denom)
            log_lik += phi_death_sum
        return log_lik

    def attribute(self, X, baseline):
        integrated_grads = []
        for i in range(len(X)):
            integrated_grad = CoxNonLinearTimeVaryingFitter.integrated_gradients(X[i], self.get_grads, baseline)
            integrated_grads.append(integrated_grad)
        return integrated_grads

    def get_grads(self, X):
        X_pt = torch.tensor(X, dtype=self.type_pt, requires_grad=True)
        for i in range(len(X_pt)):
            xi = X_pt[i]
            torch.exp(self.net(xi)).backward(retain_graph=True)
        grads = X_pt.grad
        return grads.detach().numpy()

    @staticmethod
    def integrated_gradients(inputs, gradients_fun, baseline, steps=50):
        """
        MIT License

        Copyright (c) 2018 Tianhong Dai

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """
        if baseline is None:
            baseline = 0 * inputs
            # Scale inputs and compute gradients.
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
        grads = gradients_fun(scaled_inputs)
        avg_grads = np.average(grads[:-1], axis=0)
        integrated_grad = (inputs - baseline) * avg_grads
        return integrated_grad

    def predict_log_partial_hazard(self, X):
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - \bar{x})'\beta`
        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        Returns
        -------
        DataFrame
        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        if isinstance(X, pd.DataFrame):
            check_for_numeric_dtypes_or_raise(X)

        X = X.astype(float)
        X = normalize(X, self._norm_mean.values, 1)
        X_pt = torch.tensor(X, dtype=self.type_pt)
        return pd.Series(self.net(X_pt).detach().numpy().ravel())

    def predict_partial_hazard(self, X):
        r"""
        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`\exp{(x - \bar{x})'\beta }`
        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        Returns
        -------
        DataFrame
        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.exp(self.predict_log_partial_hazard(X))

    def print_summary(self, decimals=2, style=None, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.
        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        style: string
            {html, ascii, latex}
        kwargs:
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.
        """
        justify = string_justify(18)

        headers = []

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if isinstance(self.l2_weight, np.ndarray) or self.l2_weight > 0:
            headers.append(("penalizer", self.l2_weight))
        if self.strata:
            headers.append(("strata", self.strata))

        headers.extend([
                ("number of subjects", self._n_unique),
                ("number of periods", self._n_examples),
                ("number of events", self.event_observed.sum()),
                ("partial log-likelihood", "{:.{prec}f}".format(self.log_likelihood_, prec=decimals)),
                ("time fit was run", self._time_fit_was_called),
        ])
        footers = []

        p = Printer(self, headers, footers, justify, decimals, kwargs)
        p.print(style=style)

    def log_likelihood_ratio_test(self):
        raise NotImplementedError

    def _compute_cumulative_baseline_hazard(self, tv_data, events, start, stop, weights):  # pylint: disable=too-many-locals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hazards = self.predict_partial_hazard(tv_data).values

        unique_death_times = np.unique(stop[events.values])
        baseline_hazard_ = pd.DataFrame(np.zeros_like(unique_death_times), index=unique_death_times, columns=["baseline hazard"])

        for t in unique_death_times:
            ix = (start.values < t) & (t <= stop.values)

            events_at_t = events.values[ix]
            stops_at_t = stop.values[ix]
            weights_at_t = weights.values[ix]
            hazards_at_t = hazards[ix]

            deaths = events_at_t & (stops_at_t == t)

            death_counts = (weights_at_t.squeeze() * deaths).sum()  # Should always be at least 1.
            baseline_hazard_.loc[t] = death_counts / hazards_at_t.sum()

        return baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        survival_df = np.exp(-self.baseline_cumulative_hazard_)
        survival_df.columns = ["baseline survival"]
        return survival_df

    def __repr__(self):
        classname = self._class_name
        try:
            s = """<lifelines.%s: fitted with %d periods, %d subjects, %d events>""" % (
                classname,
                self._n_examples,
                self._n_unique,
                self.event_observed.sum(),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

    def _compute_residuals(self, df, events, start, stop, weights):
        raise NotImplementedError()


class Net(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.0,
                 type_pt=torch.float, with_bn=False):
        super(Net, self).__init__()
        last_features = in_features
        self.forwards, self.dropouts, self.batch_norms = [], [], []
        for i in range(num_layers):
            this_layer = nn.Linear(last_features, hidden_features, bias=True).to(type_pt)
            torch.nn.init.xavier_uniform(this_layer.weight)
            self.add_module("l_{:d}".format(i), this_layer)
            self.forwards.append(this_layer)

            if not np.isclose(dropout, 0.0):
                this_dropout = nn.Dropout(p=dropout)
                self.add_module("d_{:d}".format(i), this_dropout)
                self.dropouts.append(this_dropout)
            else:
                self.dropouts.append(None)

            if with_bn:
                this_batchnorm = nn.BatchNorm1d(hidden_features)
                self.add_module("bn_{:d}".format(i), this_batchnorm)
                self.batch_norms.append(this_batchnorm)
            else:
                self.batch_norms.append(None)
            last_features = hidden_features
        self.beta = nn.Linear(hidden_features, out_features, bias=False).to(type_pt)
        torch.nn.init.xavier_uniform(self.beta.weight)
        self.with_bn = with_bn

    def forward(self, x):
        for layer, dropout, bn in zip(self.forwards, self.dropouts, self.batch_norms):
            x = F.leaky_relu(layer(x))
            if self.with_bn:
                x = bn(x)
            if dropout is not None:
                x = dropout(x)
        x = torch.tanh(self.beta(x))
        return x
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

sklearn.impute.IterativeImputer:

New BSD License

Copyright (c) 2007-2020 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
import scipy
import numpy as np
from scipy import stats
from sklearn.base import clone
from distutils.version import LooseVersion
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array, check_random_state, _safe_indexing
from covews.data_access.meta_data.feature_types import FeatureTypeDiscrete


class MockEstimator(object):
    def __init__(self, constant):
        self.constant = constant

    def predict(self, x):
        y_pred = np.ones((len(x),)) * self.constant
        return y_pred


class CustomImputer(IterativeImputer):
    """
    Custom multiple Imputer based on sklearn.impute.IterativeImputer.
    Differentiates between Continuous and Discrete features, whereas IterativeImputer is continuous-only.
    """
    def __init__(self,
                 feature_types,
                 regression_estimator=None,
                 classification_estimator=None,
                 missing_values=np.nan,
                 sample_posterior=False,
                 max_iter=10,
                 tol=1e-3,
                 n_nearest_features=None,
                 initial_strategy="mean",
                 imputation_order='ascending',
                 skip_complete=False,
                 min_value=None,
                 max_value=None,
                 verbose=0,
                 random_state=None,
                 add_indicator=False):
        super(CustomImputer, self).__init__(
            estimator=regression_estimator,
            missing_values=missing_values,
            sample_posterior=sample_posterior,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_features,
            initial_strategy=initial_strategy,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            add_indicator=add_indicator
        )
        self.classification_estimator = classification_estimator
        self.feature_types = feature_types

    def _impute_one_feature(self,
                            X_filled,
                            mask_missing_values,
                            feat_idx,
                            neighbor_feat_idx,
                            estimator=None,
                            fit_mode=True):
        """
        SOURCE: sklearn.impute.IterativeImputer

        Impute a single feature from the others provided.
        This function predicts the missing values of one of the features using
        the current estimates of all the other features. The ``estimator`` must
        support ``return_std=True`` in its ``predict`` method for this function
        to work.
        Parameters
        ----------
        X_filled : ndarray
            Input data with the most recent imputations.
        mask_missing_values : ndarray
            Input data's missing indicator matrix.
        feat_idx : int
            Index of the feature currently being imputed.
        neighbor_feat_idx : ndarray
            Indices of the features to be used in imputing ``feat_idx``.
        estimator : object
            The estimator to use at this step of the round-robin imputation.
            If ``sample_posterior`` is True, the estimator must support
            ``return_std`` in its ``predict`` method.
            If None, it will be cloned from self._estimator.
        fit_mode : boolean, default=True
            Whether to fit and predict with the estimator or just predict.
        Returns
        -------
        X_filled : ndarray
            Input data with ``X_filled[missing_row_mask, feat_idx]`` updated.
        estimator : estimator with sklearn API
            The fitted estimator used to impute
            ``X_filled[missing_row_mask, feat_idx]``.
        """
        if estimator is None and fit_mode is False:
            raise ValueError("If fit_mode is False, then an already-fitted "
                             "estimator should be passed in.")

        is_discrete = isinstance(self.feature_types[feat_idx], FeatureTypeDiscrete)
        if estimator is None:
            if is_discrete:
                # Case: Classification
                if self.classification_estimator is None:
                    estimator = LogisticRegression()
                else:
                    estimator = clone(self.classification_estimator)
            else:  # Case: Regression
                estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            X_train = _safe_indexing(X_filled[:, neighbor_feat_idx],
                                     ~missing_row_mask)
            y_train = _safe_indexing(X_filled[:, feat_idx],
                                     ~missing_row_mask)
            all_y_same = len(set(y_train)) == 1
            if all_y_same:
                estimator = MockEstimator(constant=y_train[0])
            else:
                estimator.fit(X_train, y_train)

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        # get posterior samples if there is at least one missing value
        X_test = _safe_indexing(X_filled[:, neighbor_feat_idx],
                                missing_row_mask)
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # two types of problems: (1) non-positive sigmas
            # (2) mus outside legal range of min_value and max_value
            # (results in inf sample)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value
            imputed_values[mus_too_low] = self._min_value
            mus_too_high = mus > self._max_value
            imputed_values[mus_too_high] = self._max_value
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value - mus) / sigmas
            b = (self._max_value - mus) / sigmas

            if scipy.__version__ < LooseVersion('0.18'):
                # bug with vector-valued `a` in old scipy
                imputed_values[inrange_mask] = [
                    stats.truncnorm(a=a_, b=b_,
                                    loc=loc_, scale=scale_).rvs(
                        random_state=self.random_state_)
                    for a_, b_, loc_, scale_
                    in zip(a, b, mus, sigmas)]
            else:
                truncated_normal = stats.truncnorm(a=a, b=b,
                                                   loc=mus, scale=sigmas)
                imputed_values[inrange_mask] = truncated_normal.rvs(
                    random_state=self.random_state_)
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(imputed_values,
                                     self._min_value,
                                     self._max_value)

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, estimator
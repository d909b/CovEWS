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
import numpy as np
from covews.apps.util import isfloat
from covews.data_access.preprocessing.base_processor import BasePreprocessor
from covews.data_access.meta_data.feature_types import FeatureTypeMissingIndicator


class ImputeMissing(BasePreprocessor):
    def __init__(self, max_num_elements_for_discretisation=6, imputation_mode="multiple", add_missing_indicators=True,
                 random_state=909):
        super(ImputeMissing, self).__init__()
        self.max_num_elements_for_discretisation = max_num_elements_for_discretisation
        self.imputation_mode = imputation_mode
        self.add_missing_indicators = add_missing_indicators
        self.random_state = random_state
        self.col_state = None
        self.imputer = None

    def fit(self, x, y=None):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = x.shape[-1]

        self.col_state, self.missing_mask = [], np.zeros_like(x)
        for j in range(num_cols):
            # Replace nan's with "nan" strings as nan itself is not a unique item in sets.
            safe_values = list(map(lambda xi: "nan" if isfloat(xi) and np.isnan(float(xi)) else xi, x[:, j]))
            unique_values = set(safe_values)
            num_unique = len(unique_values)
            converted, missing, num_missing = BasePreprocessor.convert_to_float(x[:, j])
            has_missing = num_missing != 0
            if self.imputation_mode == "zero":
                estimator = lambda _: 0  # Zero-impute missing values.
            elif self.imputation_mode == "mean":
                mean_value = np.mean(converted[missing == 0])
                estimator = lambda _: mean_value
            elif self.imputation_mode == "multiple":
                estimator = None
            else:
                raise NotImplementedError()

            self.missing_mask[:, j][missing == 1] = 1
            self.col_state.append((unique_values, num_unique, has_missing, estimator))

        # Second pass in order of multiple imputation.
        if self.imputation_mode == "multiple":
            if self.imputation_mode == "multiple":
                from covews.data_access.preprocessing.custom_imputer import CustomImputer
                self.imputer = CustomImputer(self.feature_types, random_state=self.random_state,
                                             min_value=-10, max_value=10)
            self.imputer.fit(x)

    def transform(self, x, y=None):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = len(self.col_state)
        with_feature_names = self.feature_names is not None
        with_feature_types = self.feature_types is not None

        new_x, new_feature_names, new_feature_types = [], [], []
        for j in range(num_cols):
            unique_values, num_unique, has_missing, estimator = self.col_state[j]
            if with_feature_types:
                new_feature_types.append(self.feature_types[j])
            if num_unique > self.max_num_elements_for_discretisation:
                # Case: Continuous variable.
                converted, missing, num_missing = BasePreprocessor.convert_to_float(x[:, j])

                if num_missing != 0 and not has_missing:
                    raise AssertionError(("Column {} had missing elements despite being marked as a column that does"
                                          " not have any missing elements. Perhaps your __fit__ data did not have this "
                                          " column missing whereas your __transform__ data did.").format(
                        self.feature_names[j] if self.feature_names is not None else j
                    ))

                if with_feature_names:
                    new_feature_names.append(self.feature_names[j])
                if has_missing:
                    if estimator is not None:
                        converted[missing == 1] = estimator(np.where(missing == 1)[0])

                    if self.add_missing_indicators:
                        converted = np.column_stack([converted, np.expand_dims(missing, axis=-1)])
                        if with_feature_names:
                            new_feature_names.append("MISSING_" + self.feature_names[j])
                        if with_feature_types:
                            new_feature_types.append(FeatureTypeMissingIndicator())
            else:
                converted = np.expand_dims(x[:, j], axis=-1)
                if with_feature_names:
                    new_feature_names.append(self.feature_names[j])
            new_x.append(converted)
        new_x = np.column_stack(new_x)
        not_indicators = np.where(list(map(lambda xi: not isinstance(xi, FeatureTypeMissingIndicator),
                                           new_feature_types)))[0]
        new_x[:, not_indicators] = self.imputer.transform(new_x[:, not_indicators])
        self.feature_types = new_feature_types
        assert len(self.feature_types) == new_x.shape[-1]
        if with_feature_names:
            self.feature_names = new_feature_names
            assert len(self.feature_names) == new_x.shape[-1]
        return new_x

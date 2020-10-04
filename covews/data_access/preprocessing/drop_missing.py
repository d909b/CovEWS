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


class DropMissing(BasePreprocessor):
    def __init__(self, max_fraction_missing=float("inf")):
        super(DropMissing, self).__init__()
        self.max_fraction_missing = max_fraction_missing
        self.col_state = None
        self.dropped_feature_names = None

    def fit(self, x, y=None):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = x.shape[-1]

        self.col_state = []
        for j in range(num_cols):
            missing = np.array(list(map(lambda xi: 1 if isfloat(xi) and np.isnan(float(xi)) else 0, x[:, j])))
            num_missing = np.count_nonzero(missing == 1)
            fraction_missing = float(num_missing) / len(missing)

            # Drop features with too many missing values.
            drop = fraction_missing > self.max_fraction_missing
            self.col_state.append(drop)

    def transform(self, x, y=None):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = len(self.col_state)
        with_feature_names = self.feature_names is not None

        new_x, new_feature_names, dropped_feature_names = [], [], []
        for j in range(num_cols):
            drop = self.col_state[j]
            if drop:
                if with_feature_names:
                    dropped_feature_names.append(self.feature_names[j])
            else:
                if with_feature_names:
                    new_feature_names.append(self.feature_names[j])
                new_x.append(x[:, j])

        new_x = np.column_stack(new_x)
        if with_feature_names:
            self.feature_names = new_feature_names
            self.dropped_feature_names = dropped_feature_names
            assert len(self.feature_names) == new_x.shape[-1]
        return new_x

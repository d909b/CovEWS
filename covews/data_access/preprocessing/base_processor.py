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
from sklearn.base import TransformerMixin, BaseEstimator


class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None
        self.feature_types = None

    @staticmethod
    def convert_to_float(converted):
        try:
            converted = converted.astype(float)
        except ValueError:
            # Mark any entries that could not be converted to numeric values at this point to missing.
            converted = np.array(list(map(lambda xi: float("nan") if not isfloat(xi) else xi,
                                          np.squeeze(converted))))
            converted = converted.astype(float)

        # Recalculate missingness to reflect the changes.
        missing = np.isnan(converted)*1
        num_missing = np.count_nonzero(missing == 1)
        converted = np.expand_dims(converted, axis=-1)
        return converted, missing, num_missing

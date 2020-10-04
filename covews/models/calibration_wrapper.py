"""
Copyright (C) 2020  Patrick Schwab, F. Hoffmann-La Roche Ltd

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
from bisect import bisect_left


class PercentileWrapper(object):
    def __init__(self):
        self.quantiles = []

    def fit(self, y_pred, y_true):
        self.quantiles = np.quantile(y_pred[:, -1], np.linspace(0, 1, 100))

    def transform(self, y):
        indices = np.array([bisect_left(self.quantiles, yi) for yi in y[:, -1]]) / 100.
        return indices


class CalibrationWrapper(object):
    def __init__(self, model):
        self.model = model
        self.percentile = PercentileWrapper()

    def predict_proba(self, input):
        return self.predict(input)

    def predict(self, input):
        probits = self.model.predict_proba(input)
        transformed = self.percentile.transform(probits)
        transformed = np.concatenate([1.-transformed[:, np.newaxis], transformed[:, np.newaxis]], axis=-1)
        return transformed

    def fit(self, y_pred, y_true):
        self.percentile.fit(y_pred, y_true)

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
from covews.models.baselines.time_varying_cox import TimeVaryingCox


class NonlinearTimeVaryingCox(TimeVaryingCox):
    def __init__(self, l2_weight=0.1, feature_names=[], learning_rate=0.001, num_units=12, num_layers=1,
                 batch_size=16, p_dropout=0.1, num_epochs=100):
        super(NonlinearTimeVaryingCox, self).__init__(l2_weight=l2_weight,
                                                      feature_names=feature_names,
                                                      model_type="nonlinear",
                                                      num_units=num_units,
                                                      num_layers=num_layers,
                                                      p_dropout=p_dropout,
                                                      batch_size=batch_size,
                                                      num_epochs=num_epochs,
                                                      learning_rate=learning_rate)

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "l2_weight": (0.0,),
            "num_units": (64, 128, 256),
            "num_layers": (1, 2),
            "learning_rate": (0.001,),
            "batch_size": (100,),
            "dropout": (0.1, 0.2,),
            "num_epochs": (100,)
        }
        return ranges

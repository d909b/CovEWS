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
import pandas as pd
from sklearn.base import BaseEstimator
from lifelines.utils import coalesce, interpolate_at_times
from covews.models.baselines.base_model import PickleableBaseModel, HyperparamMixin
from covews.models.nonlinear_cox.cox_time_varying_fitter import CoxTimeVaryingFitter
from covews.models.nonlinear_cox.nonlinear_time_varying_cox import CoxNonLinearTimeVaryingFitter


class TimeVaryingCox(BaseEstimator, PickleableBaseModel, HyperparamMixin):
    def __init__(self, l2_weight=0.1, feature_names=[], model_type=None,
                 learning_rate=0.001, num_units=12, num_layers=1,
                 batch_size=16, p_dropout=0.1, num_epochs=100, output_directory=""):
        self.model = None
        self.feature_names = feature_names
        self.classes_ = [0., 1.]
        self.prediction_time_frame_in_hours = 128.
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.num_units = num_units
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_directory = output_directory
        if model_type is None:
            self.model_type = CoxTimeVaryingFitter
        elif model_type == "nonlinear":
            self.model_type = CoxNonLinearTimeVaryingFitter

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "l2_weight": (0.01, 0.1, 1.0),
        }
        return ranges

    @staticmethod
    def prepare_dataset(x, y, p):
        current_pid, start_offset, contains_event, start, stop, pids = 0, 0, [], [], [], []
        for idx, (xi, (has_event, duration), pi) in enumerate(zip(x, y, p)):
            pids.append(current_pid)
            start.append(start_offset)

            start_offset += duration
            this_stop = start_offset

            assert this_stop > start[-1]
            stop.append(this_stop)

            if idx == len(p) - 1 or p[idx+1] != pi:
                contains_event.append(has_event)
                start_offset = 0
                current_pid += 1
            else:
                contains_event.append(False)
        assert len(contains_event) == len(start) == len(stop) == len(x)
        return np.array(contains_event), np.array(start), np.array(stop), np.array(pids)

    def get_data_frame(self, x, y, train_ids):
        contains_event, start, stop, pids = TimeVaryingCox.prepare_dataset(x, y, train_ids)
        prepared_dataset = np.concatenate([
            x,
            contains_event[:, np.newaxis],
            pids[:, np.newaxis],
            start[:, np.newaxis],
            stop[:, np.newaxis]
        ], axis=-1)
        train_df = pd.DataFrame(prepared_dataset,
                                columns=self.feature_names + ["event", "id", "start", "stop"])
        return train_df

    def fit(self, x, y, validation_data=None, train_ids=None):
        train_df = self.get_data_frame(x, y, train_ids)

        np.random.seed(909)
        if self.model_type == CoxNonLinearTimeVaryingFitter:
            self.model = self.model_type(
                l2_weight=self.l2_weight,
                num_units=self.num_units,
                num_layers=self.num_layers,
                p_dropout=self.p_dropout,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                output_directory=self.output_directory
            )
            val_df = self.get_data_frame(validation_data[0], validation_data[1], validation_data[2])
            self.model.fit(train_df, id_col="id", event_col="event", start_col="start", stop_col="stop",
                           show_progress=True, val_df=val_df)
        else:
            self.model = self.model_type(penalizer=self.l2_weight)
            self.model.fit(train_df, id_col="id", event_col="event", start_col="start", stop_col="stop",
                           show_progress=True)
        self.model.print_summary()

    def predict(self, x):
        if self.model is None:
            raise AssertionError("Model must be fit before calling predict.")

        np.random.seed(909)

        if isinstance(self.prediction_time_frame_in_hours, tuple):
            earlier = self.predict_at_time(x, self.prediction_time_frame_in_hours[0])
            later = self.predict_at_time(x, self.prediction_time_frame_in_hours[1])
            y_pred = later - earlier
        else:
            times_to_evaluate_at = coalesce(self.prediction_time_frame_in_hours,
                                            self.model.baseline_cumulative_hazard_.index)
            y_pred = self.predict_at_time(x, times_to_evaluate_at)

        y_pred = np.concatenate([1 - y_pred[..., np.newaxis], y_pred[..., np.newaxis]], axis=-1)
        y_pred = y_pred[0]
        return y_pred

    def predict_at_time(self, x, times_to_evaluate_at):
        if isinstance(times_to_evaluate_at, float) or len(times_to_evaluate_at) != len(x):
            times_to_evaluate_at = np.tile(times_to_evaluate_at, (len(x), 1))
        c_0 = interpolate_at_times(self.model.baseline_cumulative_hazard_, times_to_evaluate_at).T
        v = self.model.predict_partial_hazard(x)

        y_pred = c_0*v.values
        return y_pred

    def predict_proba(self, x):
        return self.predict(x)

    def attribute(self, x, baseline):
        if hasattr(self.model, "attribute"):
            return self.model.attribute(x, baseline)
        else:
            return None

    def get_config(self):
        config = {
            "l2_weight": self.l2_weight,
            "feature_names": self.feature_names,
        }
        return config


#!/usr/bin/env python3
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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import glob
import inspect
import importlib
import numpy as np
import pandas as pd
from os.path import join
from sklearn.pipeline import Pipeline
from covews.apps.util import info, warn
from covews.apps.util import time_function
from covews.apps.util import clip_percentage
from covews.apps.parameters import parse_parameters
from covews.apps.base_application import BaseApplication
from covews.models.model_evaluation import ModelEvaluation
from covews.data_access.pickle_data_loader import PickleDataLoader
from covews.data_access.generator import make_generator, get_last_row_id

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class MainApplication(BaseApplication):
    def __init__(self, args):
        super(MainApplication, self).__init__(args)
        self.training_set, self.validation_set, self.test_set, self.input_shape, \
          self.output_dim, self.feature_names, self.feature_types, self.output_names, self.output_types = [None]*9
        self.train_patients, self.val_patients, self.test_patients = None, None, None
        self.output_preprocessors, self.preprocessors = None, None
        self.load_data()

    def load_data(self):
        resample_with_replacement = self.args["resample_with_replacement"]
        seed = int(np.rint(self.args["seed"]))

        self.training_set, self.validation_set, self.test_set, self.input_shape, \
         self.output_dim, self.feature_names, self.feature_types, self.output_names, self.output_types,\
          self.train_patients, self.val_patients, self.test_patients, self.preprocessors, self.output_preprocessors = \
            self.get_data(seed=seed, resample=resample_with_replacement, resample_seed=seed)

    def setup(self):
        super(MainApplication, self).setup()

    def get_loader(self):
        dataset = self.args["dataset"].lower()
        dataset_cache = self.args["dataset_cache"].lower()
        loader = self.args["loader"]

        if loader.lower() == "pickle":
            loader = PickleDataLoader(dataset_cache=dataset_cache, task_name=dataset)
        return loader

    def get_data(self, seed=0, resample=False, resample_seed=0):
        loader_name = self.args["loader"].lower()
        dataset = self.args["dataset"].lower()
        load_existing = self.args["load_existing"]
        if load_existing:
            steps_path = os.path.join(os.path.dirname(load_existing),
                                      os.path.basename(self.get_preprocessor_path()))
            output_steps_path = os.path.join(os.path.dirname(load_existing),
                                             os.path.basename(self.get_output_preprocessor_path()))
            with open(steps_path, "rb") as fp:
                steps = pickle.load(fp)
            with open(output_steps_path, "rb") as fp:
                output_steps = pickle.load(fp)
        else:
            steps, output_steps = None, None

        if dataset.lower().startswith("covews"):
            loader = self.get_loader()
        else:
            raise NotImplementedError("{:s} is not a valid dataset.".format(dataset))
        return loader.get_data(
            self.args,
            seed=seed,
            do_resample=resample,
            resample_seed=resample_seed,
            steps=steps,
            output_steps=output_steps,
            do_split=loader_name != "trinetx"
        )

    def get_num_losses(self):
        return 1

    def make_train_generator(self, randomise=True, stratify=True):
        batch_size = int(np.rint(self.args["batch_size"]))
        seed = int(np.rint(self.args["seed"]))
        num_losses = self.get_num_losses()

        train_generator, train_steps = make_generator(
            dataset=self.training_set,
            batch_size=batch_size,
            num_losses=num_losses,
            shuffle=randomise,
            seed=seed
        )
        return train_generator, train_steps

    def make_validation_generator(self, randomise=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        val_generator, val_steps = make_generator(
            dataset=self.validation_set,
            batch_size=batch_size,
            num_losses=num_losses,
            shuffle=randomise
        )
        return val_generator, val_steps

    def make_test_generator(self, randomise=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        test_generator, test_steps = make_generator(
            dataset=self.test_set,
            batch_size=batch_size,
            num_losses=num_losses,
            shuffle=randomise
        )
        return test_generator, test_steps

    def get_visualisation_output_directory(self):
        output_directory = self.args["output_directory"]
        visualisations_output_directory = os.path.join(output_directory, "visualisations")
        if not os.path.exists(visualisations_output_directory):
            os.mkdir(visualisations_output_directory)
        return visualisations_output_directory

    def get_best_model_path(self):
        model_class = self.get_model_type_for_method_name()
        return join(self.args["output_directory"], "model" + model_class.get_save_file_type())

    def get_calibration_wrapper_path(self):
        return join(self.args["output_directory"], "calibrated.pickle")

    def get_preprocessor_path(self):
        return join(self.args["output_directory"], "preprocessor.pickle")

    def get_output_preprocessor_path(self):
        return join(self.args["output_directory"], "output_preprocessor.pickle")

    def get_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.tsv")

    def get_calibration_name(self, set_name):
        return set_name + "_calibration.pdf"

    def get_roc_name(self, set_name):
        return set_name + "_roc_{}.pdf"

    def get_thresholded_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.thresholded.tsv")

    def get_attribution_path(self, prefix=""):
        return join(self.args["output_directory"], prefix + "attributions.tsv")

    def get_hyperopt_parameters(self):
        hyper_params = {}

        resample_with_replacement = self.args["resample_with_replacement"]
        if resample_with_replacement:
            base_params = {
                "seed": [0, 2**32-1],
            }
        else:
            cls = self.get_model_type_for_method_name()
            if cls is not None:
                base_params = cls.get_hyperparameter_ranges()
            else:
                warn("Unable to retrieve class for provided method name [", self.args["method"], "].")
                base_params = {}

        hyper_params.update(base_params)
        return hyper_params

    def get_model_type_for_method_name(self):
        from covews.models.baselines.base_model import BaseModel

        method = self.args["method"]
        baseline_package_path = os.path.dirname(inspect.getfile(BaseModel))

        for module_path in glob.glob(baseline_package_path + "/*.py"):
            modname = os.path.basename(module_path)[:-3]
            fully_qualified_name = BaseModel.__module__
            fully_qualified_name = fully_qualified_name[:fully_qualified_name.rfind(".")] + "." + modname
            mod = importlib.import_module(fully_qualified_name)
            if hasattr(mod, method):
                cls = getattr(mod, method)
                return cls
        return None

    def get_model_for_method_name(self, model_params):
        cls = self.get_model_type_for_method_name()
        if cls is not None:
            instance = cls()
            available_model_params = {k: model_params[k] if k in model_params else instance.get_params()[k]
                                      for k in instance.get_params().keys()}
            instance = instance.set_params(**available_model_params)

            return instance
        else:
            return None

    @time_function("train_model")
    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        info("Started training model.")

        dropout = float(self.args["dropout"])
        seed = int(np.rint(self.args["seed"]))
        do_calibrate = self.args["do_calibrate"]
        l2_weight = float(self.args["l2_weight"])
        best_model_path = self.get_best_model_path()
        num_units = int(np.rint(self.args["num_units"]))
        output_directory = self.args["output_directory"]
        learning_rate = float(self.args["learning_rate"])
        num_epochs = int(np.rint(self.args["num_epochs"]))
        batch_size = int(np.rint(self.args["batch_size"]))
        num_layers = int(np.rint(self.args["num_layers"]))
        early_stopping_patience = int(np.rint(self.args["early_stopping_patience"]))

        model_params = {
            "output_directory": output_directory,
            "early_stopping_patience": early_stopping_patience,
            "num_layers": num_layers,
            "num_units": num_units,
            "p_dropout": dropout,
            "input_shape": self.input_shape,
            "output_dim": self.output_dim,
            "feature_names": self.feature_names,
            "output_names": self.output_names,
            "preprocessors": self.preprocessors,
            "batch_size": batch_size,
            "best_model_path": best_model_path,
            "l2_weight": l2_weight,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "seed": seed,
            "random_state": seed,
            "probability": True,
        }

        assert train_steps > 0, "You specified a batch_size that is bigger than the size of the train set."
        assert val_steps > 0, "You specified a batch_size that is bigger than the size of the validation set."

        if self.args["load_existing"]:
            info("Loading existing model from", self.args["load_existing"])
            model_class = self.get_model_type_for_method_name()
            model = model_class.load(self.args["load_existing"])

            if do_calibrate:
                model_directory = os.path.dirname(self.args["load_existing"])
                wrapper_model_name = os.path.basename(self.get_calibration_wrapper_path())
                wrapper_model_path = os.path.join(model_directory, wrapper_model_name)

                info("Loading calibration model from", wrapper_model_path)
                with open(wrapper_model_path, "rb") as load_file:
                    wrapper = pickle.load(load_file)
                model = wrapper
        else:
            model = self.get_model_for_method_name(model_params)

        if self.args["do_train"]:
            history = model.fit(self.training_set[0], self.training_set[1],
                                validation_data=(self.validation_set[0],
                                                 self.validation_set[1],
                                                 self.validation_set[2]),
                                train_ids=self.training_set[2])
            model.predict(self.training_set[0])

            if isinstance(model, Pipeline):
                model.steps[1][1].save(best_model_path)
            else:
                model.save(best_model_path)

            info("Saved model to", best_model_path)

            preprocessor_path = self.get_preprocessor_path()
            output_preprocessor_path = self.get_output_preprocessor_path()

            with open(preprocessor_path, "wb") as fp:
                pickle.dump(self.preprocessors, fp)
            info("Saved preprocessors to", preprocessor_path)

            with open(output_preprocessor_path, "wb") as fp:
                pickle.dump(self.output_preprocessors, fp)
            info("Saved output preprocessors to", output_preprocessor_path)

            if do_calibrate:
                from covews.models.calibration_wrapper import CalibrationWrapper

                wrapper = CalibrationWrapper(model)
                y_pred_val = model.predict(self.validation_set[0])
                wrapper.fit(y_pred_val, self.validation_set[1][:, 0])

                wrapper_model_path = self.get_calibration_wrapper_path()
                with open(wrapper_model_path, "wb") as load_file:
                    pickle.dump(wrapper, load_file, pickle.HIGHEST_PROTOCOL)

                info("Saved calibration model to", wrapper_model_path)
                model = wrapper
        else:
            history = {
                "val_acc": [],
                "val_loss": [],
                "val_combined_loss": [],
                "acc": [],
                "loss": [],
                "combined_loss": []
            }
        return model, history

    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="test", threshold=None,
                       plot_roc=True, prediction_time_frames=[0]):
        if with_print:
            info("Started evaluation.")

        scores = ModelEvaluation.evaluate_classifier(model, test_generator, test_steps, set_name,
                                                     threshold=threshold, with_print=with_print,
                                                     num_multilabel_outputs=self.output_dim,
                                                     output_directory=self.args["output_directory"],
                                                     prediction_time_frames=prediction_time_frames)
        return scores

    def save_predictions(self, model, threshold=None, prediction_time_frames=[24.]):
        info("Saving model predictions.")

        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])

        generators = [self.make_train_generator, self.make_validation_generator, self.make_test_generator]
        generator_names = ["train", "val", "test"]
        for generator_fun, generator_name in zip(generators, generator_names):
            generator, steps = generator_fun(randomise=False)
            steps = int(np.rint(steps * fraction_of_data_set))

            y_true, predictions, inputs = [], [], []
            for step in range(steps):
                x, y = next(generator)
                last_id = get_last_row_id()
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(x)[:, -1]
                else:
                    y_pred = model.predict(x)
                y_pred = np.squeeze(y_pred)
                if y_pred.size == 1:
                    y_pred = [y_pred]

                inputs.append(x)
                y_true.append(y)
                for current_id, current_y_pred in zip(last_id, y_pred):
                    predictions.append([current_id, current_y_pred])
            row_ids = np.hstack(list(map(lambda x: x[0], predictions)))
            outputs = np.vstack(list(map(lambda x: x[1], predictions)))
            file_path = self.get_prediction_path(generator_name)

            num_predictions = 1 if len(outputs.shape) == 1 else outputs.shape[-1]
            assert num_predictions == len(self.output_names) - 1
            columns = self.output_names[:-1]

            df = pd.DataFrame(outputs, columns=columns, index=row_ids)
            df.index.name = "PATIENTID"
            df.to_csv(file_path, sep="\t")
            info("Saved raw model predictions to", file_path)

            if threshold is not None:
                thresholded_file_path = self.get_thresholded_prediction_path(generator_name)
                df = pd.DataFrame((outputs > threshold[0][0]).astype(int), columns=columns, index=row_ids)
                df.index.name = "PATIENTID"
                df.to_csv(thresholded_file_path, sep="\t")
                info("Saved thresholded model predictions to", thresholded_file_path)


if __name__ == '__main__':
    app = MainApplication(parse_parameters())
    app.run()

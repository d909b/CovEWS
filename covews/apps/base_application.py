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
import sys
import six
import numpy as np
from os.path import join
from datetime import datetime
from abc import ABCMeta, abstractmethod
from covews.apps.parameters import parse_parameters
from covews.apps.util import info, warn, init_file_logger

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


@six.add_metaclass(ABCMeta)
class BaseApplication(object):
    def __init__(self, args):
        super(BaseApplication, self).__init__()

        self.args = args

        log_path = os.path.join(self.args["output_directory"], "log.txt")
        init_file_logger(log_path)

        info(sys.argv)
        info("Args are:", self.args)
        info("Running at", str(datetime.now()))
        self.init_seeds()
        self.setup()

    def init_seeds(self):
        import torch
        import random as rn

        seed = int(np.rint(self.args["seed"]))
        info("Seed is", seed)

        os.environ['PYTHONHASHSEED'] = '0'
        rn.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def setup(self):
        pass

    @abstractmethod
    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        return None, None

    @abstractmethod
    def evaluate_model(self, model, test_generator, test_steps, with_print=True,
                       set_name="", threshold=None, prediction_time_frames=[0]):
        return None

    @abstractmethod
    def make_train_generator(self):
        return None, None

    @abstractmethod
    def make_validation_generator(self):
        return None, None

    @abstractmethod
    def make_test_generator(self):
        return None, None

    @abstractmethod
    def get_visualisation_output_directory(self):
        return ""

    @abstractmethod
    def get_best_model_path(self):
        return ""

    @abstractmethod
    def get_preprocessor_path(self):
        return ""

    @abstractmethod
    def get_output_preprocessor_path(self):
        return ""

    @abstractmethod
    def get_calibration_wrapper_path(self):
        return ""

    @abstractmethod
    def get_prediction_path(self, set_name):
        return ""

    @abstractmethod
    def get_calibration_name(self, set_name):
        return ""

    @abstractmethod
    def get_roc_name(self, set_name):
        return ""

    @abstractmethod
    def get_thresholded_prediction_path(self, set_name):
        return ""

    @abstractmethod
    def get_attribution_path(self, prefix=""):
        return ""

    @abstractmethod
    def save_predictions(self, model, threshold=None, prediction_time_frames=[24.]):
        return

    @abstractmethod
    def load_data(self):
        return

    def run(self):
        evaluate_against = self.args["evaluate_against"]
        if evaluate_against not in ("test", "val"):
            warn("Specified wrong argument for --evaluate_against. Value was:", evaluate_against,
                 ". Defaulting to: val")
            evaluate_against = "val"
        return self.run_single(evaluate_against=evaluate_against)

    def run_single(self, evaluate_against="test"):
        info("Run with args:", self.args)

        save_predictions = self.args["save_predictions"]

        train_generator, train_steps = self.make_train_generator()
        val_generator, val_steps = self.make_validation_generator()
        test_generator, test_steps = self.make_test_generator()

        info("Built generators with", train_steps,
             "training steps, ", val_steps,
             "validation steps and", test_steps, "test steps.")

        model, history = self.train_model(train_generator,
                                          train_steps,
                                          val_generator,
                                          val_steps)

        loss_file_path = join(self.args["output_directory"], "losses.pickle")
        info("Saving loss history to", loss_file_path)
        pickle.dump(history, open(loss_file_path, "wb"), pickle.HIGHEST_PROTOCOL)

        args_file_path = join(self.args["output_directory"], "args.pickle")
        info("Saving args to", loss_file_path)
        pickle.dump(self.args, open(args_file_path, "wb"), pickle.HIGHEST_PROTOCOL)

        time_frames = [0., 1., 2., 4., 8., 16., 24., 48., 96., 192., 384.]

        threshold = None
        if self.args["do_evaluate"]:
            if evaluate_against == "test":
                thres_generator, thres_steps = val_generator, val_steps
                eval_generator, eval_steps = test_generator, test_steps
                eval_set = self.test_set
            else:
                thres_generator, thres_steps = train_generator, train_steps
                eval_generator, eval_steps = val_generator, val_steps
                eval_set = self.validation_set

            if self.args["load_existing"]:
                with open(os.path.join(os.path.dirname(self.args["load_existing"]), "eval_score.pickle"), "rb") as fp:
                    thres_score = pickle.load(fp)
            else:
                # Get threshold from train or validation set.
                thres_score = self.evaluate_model(model, thres_generator, thres_steps,
                                                  with_print=False, set_name="threshold",
                                                  prediction_time_frames=time_frames)
            threshold = [[thres_score["{:d}.threshold.{:d}".format(idx, int(t))] for t in time_frames]
                         for idx in range(self.output_dim)]

            eval_score = self.evaluate_model(model, eval_generator, eval_steps,
                                             set_name=evaluate_against, threshold=threshold,
                                             prediction_time_frames=time_frames)
        else:
            eval_score = None
            thres_score = None

        if save_predictions:
            self.save_predictions(model, threshold=threshold, prediction_time_frames=time_frames)

        if self.args["do_evaluate"]:
            if eval_score is None:
                test_score = self.evaluate_model(model, test_generator, test_steps,
                                                 with_print=evaluate_against == "val", set_name="test")
                eval_score = test_score
            else:
                test_score = eval_score
                eval_score = thres_score
        else:
            test_score = None
        BaseApplication.save_score_dicts(eval_score, test_score, self.args["output_directory"])

        return eval_score, test_score

    @staticmethod
    def save_score_dicts(eval_score, test_score, output_directory):
        eval_score_path = join(output_directory, "eval_score.pickle")
        with open(eval_score_path, "wb") as fp:
            pickle.dump(eval_score, fp, pickle.HIGHEST_PROTOCOL)
        test_score_path = join(output_directory, "test_score.pickle")
        with open(test_score_path, "wb") as fp:
            pickle.dump(test_score, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app = BaseApplication(parse_parameters())
    app.run()

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
from argparse import ArgumentParser


def parse_parameters():
    parser = ArgumentParser(description='COVID-19 Early Warning System (CovEWS).')
    parser.add_argument("--loader", default="pickle",
                        help="The loader to be used. One of: (pickle).")
    parser.add_argument("--dataset", default="covews",
                        help="The data files to be loaded. One of: (covews).")
    parser.add_argument("--dataset_cache", default="",
                        help="Cache file path to load the dataset from. Caching not used if empty.")
    parser.add_argument("--filter_subgroup", default="",
                        help="Subgroup filter to apply. One of (None, hispanic, asian, caucasian, black)")
    parser.add_argument("--evaluate_against", default="test",
                        help="Fold to evaluate trained models against. One of: (val, test).")
    parser.add_argument("--method", default="NonlinearTimeVaryingCox",
                        help="Predictive model to be used."
                             "One of: (NonlinearTimeVaryingCox, TimeVaryingCox).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for the random number generator.")
    parser.add_argument("--output_directory", default="./",
                        help="Base directory of all output files.")
    parser.add_argument("--model_name", default="model.h5.npz",
                        help="Base directory of all output files.")
    parser.add_argument("--load_existing", default="",
                        help="Existing model to load.")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="Learning rate to use for training.")
    parser.add_argument("--l2_weight", default=0.0, type=float,
                        help="L2 weight decay used on neural network weights.")
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size to use for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=12,
                        help="Number of stale epochs to wait before terminating training")
    parser.add_argument("--num_units", type=int, default=8,
                        help="Number of neurons to use in neural network layers.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers to use in neural networks.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="Value of the dropout parameter used in training in the network.")
    parser.add_argument("--fraction_of_data_set", type=float, default=1,
                        help="Fraction of time_series to use for folds.")
    parser.add_argument("--validation_set_fraction", type=float, default=0.2,
                        help="Fraction of time_series to hold out for the validation set.")
    parser.add_argument("--test_set_fraction", type=float, default=0.3,
                        help="Fraction of time_series to hold out for the test set.")

    parser.set_defaults(do_train=False)
    parser.add_argument("--do_train", dest='do_train', action='store_true',
                        help="Whether or not to train a model.")
    parser.set_defaults(do_evaluate=False)
    parser.add_argument("--do_evaluate", dest='do_evaluate', action='store_true',
                        help="Whether or not to evaluate a model.")
    parser.set_defaults(resample_with_replacement=False)
    parser.add_argument("--resample_with_replacement", dest='resample_with_replacement', action='store_true',
                        help="Whether or not to use resampling w/ replacement in the patient generator.")
    parser.set_defaults(save_predictions=True)
    parser.add_argument("--do_not_save_predictions", dest='save_predictions', action='store_false',
                        help="Whether or not to save predictions.")
    parser.set_defaults(do_calibrate=False)
    parser.add_argument("--do_calibrate", dest='do_calibrate', action='store_true',
                        help="Whether or not to calibrate the predictive model using percentile scaling.")

    return vars(parser.parse_args())

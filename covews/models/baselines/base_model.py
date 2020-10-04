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
import json
from abc import ABCMeta, abstractmethod

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


@six.add_metaclass(ABCMeta)
class BaseModel(object):
    @abstractmethod
    def save(self, file_path):
        raise NotImplementedError()

    @staticmethod
    def get_save_file_type():
        raise NotImplementedError()

    @staticmethod
    def save_config(file_path, config, config_file_name, overwrite, outer_class):
        directory_path = os.path.dirname(os.path.abspath(file_path))

        already_exists_exception_message = "__directory_path__ already contains a saved" + outer_class.__name__ + \
                                           " instance and __overwrite__ was set to __False__. Conflicting file: {}"
        config_file_path = os.path.join(directory_path, config_file_name)
        if os.path.exists(config_file_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(config_file_path))
        else:
            with open(config_file_path, "w") as fp:
                json.dump(config, fp)


class PickleableBaseModel(BaseModel):
    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as load_file:
            return pickle.load(load_file)

    def save(self, file_path):
        with open(file_path, "wb") as save_file:
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_save_file_type():
        return ".pickle"


class HyperparamMixin(object):
    @staticmethod
    def get_hyperparameter_ranges():
        raise NotImplementedError()

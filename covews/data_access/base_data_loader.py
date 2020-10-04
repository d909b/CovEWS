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
import six
import numpy as np
import pandas as pd
from collections import Counter
from covews.apps.util import info
from abc import ABCMeta, abstractmethod
from covews.apps.util import time_function
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from covews.data_access.meta_data.feature_types import FeatureTypeUnknown


@six.add_metaclass(ABCMeta)
class BaseDataLoader(object):
    @abstractmethod
    def get_patients(self):
        raise NotImplementedError()

    @abstractmethod
    def transform_patients(self, patients, with_print=True):
        raise NotImplementedError()

    @abstractmethod
    def select_task(self, y, output_names, output_types):
        raise NotImplementedError()

    @abstractmethod
    def get_covariate_preprocessors(self, seed):
        raise NotImplementedError()

    @abstractmethod
    def get_output_preprocessors(self, seed):
        raise NotImplementedError()

    def preprocess_covariates(self, x, feature_names=None, steps=None, seed=909):
        return self.preprocess_variables(x, self.get_covariate_preprocessors,
                                         feature_names=feature_names, steps=steps, seed=seed)

    def preprocess_outputs(self, x, feature_names=None, steps=None, seed=909):
        return self.preprocess_variables(x, self.get_output_preprocessors,
                                         feature_names=feature_names, steps=steps, seed=seed)

    def preprocess_variables(self, x, get_fun, feature_names=None, steps=None, seed=909):
        load_new = steps is None
        if load_new:
            steps = get_fun(seed=seed)

        prior_feature_types = [FeatureTypeUnknown() for _ in range(x.shape[-1])]
        prior_feature_names = feature_names
        for step in steps:
            step.feature_names = prior_feature_names
            step.feature_types = prior_feature_types
            if load_new:
                step.fit(x)
            x = step.transform(x)
            prior_feature_names = step.feature_names
            prior_feature_types = step.feature_types

        if feature_names is not None:
            return x, steps, prior_feature_names, prior_feature_types
        else:
            return x, steps

    @abstractmethod
    def get_split_values(self, patients):
        raise NotImplementedError()

    @abstractmethod
    def get_discrete_covariate_names(self):
        raise NotImplementedError()

    @abstractmethod
    def get_continuous_covariate_names(self):
        raise NotImplementedError()

    @staticmethod
    def make_synthetic_labels_for_stratification(num_bins=5, min_synthetic_group_size=100, max_num_unique_values=10,
                                                 label_candidates=list([])):
        synthetic_labels = []
        for arg_idx in range(len(label_candidates)):
            arg = label_candidates[arg_idx]
            if len(arg) == 0:
                raise ValueError("Length of synthetic label inputs should not be zero.")

            if len(set(arg)) <= max_num_unique_values:
                if len(set(arg)) == 2:
                    more_labels = np.array(arg).astype(int)
                else:
                    more_labels = to_categorical(np.array(arg).astype(int), num_classes=len(set(arg)))
            else:
                arg = arg.astype(float)
                nan_indices = list(map(lambda xi: np.isnan(xi), arg))
                assignments = np.digitize(arg, np.linspace(np.min(arg[np.logical_not(nan_indices)]),
                                                           np.max(arg[np.logical_not(nan_indices)]), num_bins)) - 1

                while True:
                    if len(assignments) < min_synthetic_group_size:
                        break

                    counts = Counter(assignments)
                    has_reset_any = False
                    for key, num_instances in counts.items():
                        if num_instances < min_synthetic_group_size:
                            new_key = key - 1 if key != 0 else key + 1
                            assignments[assignments == key] = new_key
                            has_reset_any = True
                            break
                    if not has_reset_any:
                        break

                more_labels = to_categorical(assignments,
                                             num_classes=num_bins)

            synthetic_labels.append(more_labels)
        synthetic_labels = np.column_stack(synthetic_labels)
        return synthetic_labels

    def merge_groups_smaller_than(self, converted_synthetic_labels, min_group_size=100):
        counts = Counter(converted_synthetic_labels)
        sorted_keys = sorted(range(len(counts)), key=lambda key: counts[key])
        for k in range(len(sorted_keys)):
            key = sorted_keys[k]
            if counts[key] < min_group_size:
                converted_synthetic_labels[converted_synthetic_labels == key] = sorted_keys[k + 1]
                counts[sorted_keys[k + 1]] += counts[key]
                counts[key] = 0
        return converted_synthetic_labels

    def split_dataset(self, rows, num_validation_samples, num_test_samples,
                      random_state=1, max_num_unique_values=10):
        split_properties, split_ids = self.get_split_values(rows)
        synthetic_labels = []
        for j in range(split_properties.shape[-1]):
            unique_values = sorted(list(set(split_properties[:, j])))
            if len(unique_values) < max_num_unique_values:
                index_map = dict(zip(unique_values, range(len(unique_values))))
                labels = list(map(lambda xi: index_map[xi], split_properties[:, j]))
            else:
                labels = split_properties[:, j]
            synthetic_labels.append(labels)

        x = np.arange(len(rows))

        for bin_size in reversed([2, 3, 4, 5]):
            converted_synthetic_labels = BaseDataLoader.make_synthetic_labels_for_stratification(
                label_candidates=synthetic_labels,
                num_bins=bin_size,
                min_synthetic_group_size=100,
                max_num_unique_values=max_num_unique_values
            )
            converted_synthetic_labels = np.squeeze(
                converted_synthetic_labels.astype(int).dot(2**np.arange(converted_synthetic_labels.shape[-1])[::-1])
            )
            synthetic_labels_map = dict(zip(sorted(list(set(converted_synthetic_labels))),
                                            range(len(set(converted_synthetic_labels)))))
            converted_synthetic_labels = np.array(list(
                map(lambda xi: synthetic_labels_map[xi], converted_synthetic_labels)
            ))
            converted_synthetic_labels = self.merge_groups_smaller_than(converted_synthetic_labels, min_group_size=30)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_samples, random_state=random_state)
            try:
                rest_index, test_index = next(sss.split(x, converted_synthetic_labels))
            except ValueError:
                continue  # Reset if split fails.
            x_rest = np.array([x[idx] for idx in rest_index])

            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation_samples, random_state=random_state)
            try:
                train_index, val_index = next(sss.split(x_rest, converted_synthetic_labels[rest_index]))
            except ValueError:
                continue  # Reset if split fails.
            train_index = x_rest[train_index]
            val_index = x_rest[val_index]

        assert len(set(train_index).intersection(set(val_index))) == 0
        assert len(set(train_index).intersection(set(test_index))) == 0

        return split_ids[train_index], split_ids[val_index], split_ids[test_index]

    @staticmethod
    def group_by_mean(train_x, column_names, group_by_key):
        df = pd.DataFrame(train_x, columns=column_names)
        mean_per_id = df.groupby(group_by_key).agg([np.nanmedian])
        return mean_per_id.values

    @staticmethod
    def filter_patient_subgroup(patients, subgroup_name):
        if subgroup_name != "":
            from covews.data_access.data_model.patient import Patient

            def include_patient(p_id):
                if subgroup_name == "hispanic":
                    attr = "ethnicity"
                    check = Patient.ETHNICITY_HISPANIC
                elif subgroup_name == "hospitalised":
                    attr = ["hospital_admissions", "icu_admissions"]
                    check = [None, None]
                else:
                    attr = "race"
                    if subgroup_name == "asian":
                        check = Patient.RACE_ASIAN
                    elif subgroup_name == "caucasian":
                        check = Patient.RACE_CAUCASIAN
                    elif subgroup_name == "black":
                        check = Patient.RACE_AFRICAN_AMERICAN
                if isinstance(attr, list):
                    value = []
                    for a in attr:
                        this_value = getattr(patients[p_id], a)
                        value.append(this_value)
                else:
                    value = getattr(patients[p_id], attr)
                if check is None:
                    return value is not None
                else:
                    return value == check

            included_patients = list(filter(lambda pt_id: include_patient(pt_id),
                                            list(patients.keys())))
            included_patients = dict(zip(included_patients, [patients[pt_id] for pt_id in included_patients]))
            return included_patients

    @time_function("load_data")
    def get_data(self, args, do_resample=False, seed=0, resample_seed=0, steps=None, output_steps=None, do_split=True):
        output_directory = args["output_directory"]
        filter_subgroup = args["filter_subgroup"]

        patients = self.get_patients()

        num_patients = len(patients)
        patient_ids = np.squeeze(list(patients.keys()))

        info("Loaded", num_patients, "patients.")

        if do_resample:
            random_state = np.random.RandomState(resample_seed)
            resampled_samples = random_state.randint(0, num_patients, size=num_patients)
            patients = patients.iloc[resampled_samples]
            patient_ids = [patient_ids[idx] for idx in resampled_samples]

        num_test_samples = int(np.rint(args["test_set_fraction"] * num_patients))
        num_validation_samples = int(np.rint(args["validation_set_fraction"] * num_patients))

        if do_split:
            train_index, val_index, test_index = self.split_dataset(patients,
                                                                    num_validation_samples,
                                                                    num_test_samples,
                                                                    random_state=seed)
        else:
            train_index, val_index, test_index = patient_ids, patient_ids, patient_ids

        train_patients = {patient_id: patients[patient_id] for patient_id in train_index}
        val_patients = {patient_id: patients[patient_id] for patient_id in val_index}
        test_patients = {patient_id: patients[patient_id] for patient_id in test_index}

        if filter_subgroup != "":
            train_patients = BaseDataLoader.filter_patient_subgroup(train_patients, filter_subgroup)
            val_patients = BaseDataLoader.filter_patient_subgroup(val_patients, filter_subgroup)
            test_patients = BaseDataLoader.filter_patient_subgroup(test_patients, filter_subgroup)

        p_train, x_train, y_train, feature_names, output_names = self.transform_patients(train_patients)
        p_val, x_val, y_val, _, _ = self.transform_patients(val_patients)
        p_test, x_test, y_test, _, _ = self.transform_patients(test_patients)

        unique_ptid = np.unique(p_train)
        ptid_map = dict(zip(unique_ptid, np.arange(len(unique_ptid))))
        pt_id_numeric = list(map(lambda x: ptid_map[x], p_train))

        group_by_key = "ptid"
        x_train_grouped = self.group_by_mean(np.concatenate([np.array(pt_id_numeric)[:, np.newaxis], x_train], axis=-1),
                                             [group_by_key] + feature_names,
                                             group_by_key=group_by_key)
        _, diagnoses, _ = self.get_initial_feature_names()
        var_name_idx_map = dict(zip(feature_names, range(len(feature_names))))
        for diagnosis in diagnoses:
            dx_idx = var_name_idx_map[diagnosis]
            indices = np.where(x_train_grouped[:, dx_idx] == 0.5)[0]
            x_train_grouped[indices, dx_idx] = 1.0

        # Train preprocessing on grouped rows to ensure no bias towards patients with more covariate changes.
        if steps is None:
            _, steps, feature_names, feature_types = self.preprocess_covariates(x_train_grouped, feature_names,
                                                                                seed=seed)
        else:
            _, _, feature_names, feature_types = self.preprocess_covariates(x_train_grouped, feature_names,
                                                                            steps=steps)

        x_train, _ = self.preprocess_covariates(x_train, steps=steps)
        x_val, _ = self.preprocess_covariates(x_val, steps=steps)
        x_test, _ = self.preprocess_covariates(x_test, steps=steps)

        if output_steps is None:
            y_train, output_steps, output_names, output_types = self.preprocess_outputs(y_train, output_names,
                                                                                        seed=seed)
        else:
            y_train, output_steps, output_names, output_types = self.preprocess_outputs(y_train, output_names,
                                                                                        steps=output_steps)
        y_val, _ = self.preprocess_outputs(y_val, steps=output_steps)
        y_test, _ = self.preprocess_outputs(y_test, steps=output_steps)

        assert x_train.shape[-1] == x_val.shape[-1] and x_val.shape[-1] == x_test.shape[-1]
        assert x_train.shape[-1] == len(feature_names)

        input_shape = (x_train.shape[-1],)

        y_train, _, _ = self.select_task(y_train, output_names, output_types)
        y_val, _, _ = self.select_task(y_val, output_names, output_types)
        y_test, output_names, output_types = self.select_task(y_test, output_names, output_types)

        return (x_train.astype(float), np.squeeze(y_train), np.squeeze(p_train)), \
               (x_val.astype(float), np.squeeze(y_val), np.squeeze(p_val)), \
               (x_test.astype(float), np.squeeze(y_test), np.squeeze(p_test)), \
               input_shape, len(output_names) - 1, feature_names, feature_types, output_names, output_types, \
               train_patients, val_patients, test_patients, steps, output_steps

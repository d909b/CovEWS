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
import pickle
import numpy as np
from functools import reduce
from covews.apps.util import info
from bisect import bisect_left, bisect_right
from covews.data_access.base_data_loader import BaseDataLoader
from covews.data_access.data_model.patient import \
    Patient, Diagnosis, Procedure, Observation
from covews.data_access.preprocessing import ImputeMissing, Standardise, DropMissing, OneHotDiscretise


class PickleDataLoader(BaseDataLoader):
    def __init__(self, dataset_cache, task_name="covews_tests"):
        self.covariates_names = None
        self.task_name = task_name
        self.dataset_cache = dataset_cache
        self.with_durations = True

    def get_initial_feature_names(self):
        time_varying = [
            "temperature", "pulse", "respiratory_rate", "systolic_blood_pressure", "diastolic_blood_pressure",
            "lymphocytes", "CRP", "hsCRP", "lactate_dehydrogenase", "procalcitonin", "fibrin_d_dimer",
            "white_blood_cell_count", "cardiac_troponin_i", "creatinine", "albumin", "platelets", "hco3", "ph",
            "neutrophil", "eosinophil", "basophil", "gamma_glutamyl_transferase",
            "aspartate_aminotransferase", "ferritin", "creatine_kinase",
            "pco2", "pao2", "spo2", "co2", "il6", "bilirubin"
        ]

        diagnoses = [
            "kidney_disease",
            "ischemic_heart_diseases", "cerebovascular_diseases", "other_heart_diseases", "pulmonary_embolism",
            "connective_tissue_diseases", "inflamatory_bowel_disease",
            "hyperlipidemia", "hypertension", "diabetes", "cancer", "copd",
            "asthma", "osteoarthritis",
            "rheumatoid_arthritis", "hiv", "dyspnea", "intubated"
        ]
        feature_names = [
            "age", "gender", "weight", "height", "bmi", "smoke"
        ] + diagnoses + time_varying
        return feature_names, diagnoses, time_varying

    def transform_patients(self, patients, with_print=True):
        feature_names, diagnoses, time_varying = self.get_initial_feature_names()
        var_name_idx_map = dict(zip(feature_names, range(len(feature_names))))

        num_diagnoses, before_origin = 0, 0
        p_train, x_train, y_train, num_no_dose = [], [], [], 0
        for offset, patient_id in enumerate(sorted(patients.keys())):
            patient = patients[patient_id]

            all_elems = [[(var_name, elem) for elem in getattr(patient, var_name)]
                         for var_name in time_varying if getattr(patient, var_name) is not None]
            if len(all_elems) == 0:
                timeline = []
            else:
                timeline = sorted(reduce(lambda a, b: a+b, all_elems),
                                  key=lambda x: x[1].timestamp)
            initial_vars = [
                patient.age,
                Patient.GENDER_MALE if patient.gender == "Male" or patient.gender == "M" else Patient.GENDER_FEMALE,
                patient.weight[-1].value if patient.weight is not None else None,
                patient.height[-1].value if patient.height is not None else None,
                patient.bmi[-1].value if patient.bmi is not None else None,
                patient.smoke[-1].value if patient.smoke is not None else None,
            ] + [int(False) for _ in diagnoses] + [None for _ in time_varying]  # Initialise with None.

            for diagnosis_name in diagnoses:
                if diagnosis_name == "intubated":
                    attr_name = "intubations"
                else:
                    attr_name = diagnosis_name
                attr = getattr(patient, attr_name)
                if attr is not None:
                    if isinstance(attr, list):
                        attr = attr[0]
                    insert_idx = bisect_left(list(map(lambda x: x[1].timestamp, timeline)), attr.timestamp)
                    timeline.insert(insert_idx, (diagnosis_name, attr))

            origin_name = "ORIGIN"
            origin_idx = bisect_left(list(map(lambda x: x[1].timestamp, timeline)), patient.covid_status.timestamp)
            timeline.insert(origin_idx, (origin_name, patient.covid_status))

            if (self.task_name == "covews_icu" and patient.icu_admissions is not None) or \
               (self.task_name == "covews_admit" and patient.hospital_admissions is not None):
                reference = patient.icu_admissions[0].timestamp \
                    if self.task_name == "covews_icu" else patient.hospital_admissions[0].timestamp
                end_idx = bisect_right(list(map(lambda x: x[1].timestamp, timeline)), reference)
                timeline = timeline[:end_idx]

            for idx, (event_name, event) in enumerate(timeline):
                increment = isinstance(event, Diagnosis)*1
                if event_name != origin_name:
                    var_idx = var_name_idx_map[event_name]
                    if isinstance(event, Observation):
                        value = event.value
                    elif isinstance(event, Procedure) or isinstance(event, Diagnosis):
                        value = 1
                        num_diagnoses += increment
                        before_origin += increment
                    else:
                        value = event.test_value
                    initial_vars[var_idx] = value

                # Do not emit before origin time, and for events recorded at the same time.
                if idx >= origin_idx and (idx == len(timeline) - 1 or timeline[idx+1][1].timestamp > event.timestamp):
                    before_origin -= increment
                    x_train.append(list(initial_vars))

                    midday_offset = 12.
                    seconds_to_hours = 1./60./60.
                    mortality_tte = \
                        (patient.date_of_death - event.timestamp).total_seconds()*seconds_to_hours \
                        if patient.date_of_death is not None and patient.covid_status is not None else None
                    admissions_tte = \
                        (patient.hospital_admissions[0].timestamp - event.timestamp).total_seconds()*seconds_to_hours \
                        if patient.hospital_admissions is not None and patient.covid_status is not None else None
                    icu_tte = \
                        (patient.icu_admissions[0].timestamp - event.timestamp).total_seconds()*seconds_to_hours \
                        if patient.icu_admissions is not None and patient.covid_status is not None else None
                    intub_tte = \
                        (patient.intubations[0].timestamp - event.timestamp).total_seconds()*seconds_to_hours \
                        if patient.intubations is not None and patient.covid_status is not None else None

                    if self.with_durations:
                        if not (idx == len(timeline) - 1):
                            mortality_tte = admissions_tte = icu_tte = intub_tte = False
                            duration = \
                                (timeline[idx+1][1].timestamp - event.timestamp).total_seconds()*seconds_to_hours
                        else:
                            if mortality_tte is None:
                                duration = \
                                    (patient.last_observed_date - event.timestamp).total_seconds()*seconds_to_hours +\
                                    midday_offset
                            else:
                                duration = mortality_tte
                            mortality_tte = mortality_tte is not None
                            admissions_tte = admissions_tte is not None
                            icu_tte = icu_tte is not None
                            intub_tte = intub_tte is not None

                        if duration == 0.0:
                            duration = 1./60.

                    y_train.append((
                        mortality_tte,
                        int(patient.covid_status is not None),
                        icu_tte,
                        admissions_tte,
                        intub_tte,
                        duration,
                    ))
                    p_train.append(patient_id)

        total_obs_time_in_years = np.sum(np.array(y_train)[:, -1]) / 24. / 365.25

        if with_print:
            info("Total observation time is {:.2f} years.".format(total_obs_time_in_years))

        output_names = ["mortality", "covid", "icu", "admit", "intubation", "duration"]
        assert len(x_train) == len(y_train) == len(p_train)
        assert len(x_train[0]) == len(feature_names)
        assert len(y_train[0]) == len(output_names)
        return p_train, np.array(x_train, dtype=float), np.array(y_train, dtype=float), feature_names, output_names

    def select_task(self, y, output_names, output_types):
        task_index_map = {
            "covews_tests": 1,
            "covews_admit": 3,
            "covews_icu": 2,
            "covews_intubation": 4,
            "covews_mortality": 0,
        }

        task_index = task_index_map[self.task_name]
        y = np.concatenate([y[:, task_index:task_index+1], y[:, -1:]], axis=-1)
        output_names = output_names[task_index:task_index+1] + output_names[-1:]
        output_types = output_types[task_index:task_index+1] + output_types[-1:]
        return y, output_names, output_types

    def get_covariate_preprocessors(self, seed):
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(),
            Standardise(),
            ImputeMissing(random_state=seed, add_missing_indicators=False),
        ]
        return preprocessors

    def get_output_preprocessors(self, seed):
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(),
        ]
        return preprocessors

    def get_discrete_covariate_names(self):
        _, diagnoses, _ = self.get_initial_feature_names()
        covariate_names = [
            "gender", "smoke", "date_of_death", "covid_status",
            "hospital_admissions", "icu_admissions", "intubations"
        ] + diagnoses
        return covariate_names

    def get_continuous_covariate_names(self):
        _, _, time_varying = self.get_initial_feature_names()
        covariate_names = ["age", "weight", "height", "bmi"] + time_varying
        return covariate_names

    def get_split_values(self, patients):
        split_values, ids = [], []
        for patient_id in patients.keys():
            patient = patients[patient_id]
            values = (
                patient.age,
                patient.gender,
                patient.date_of_death is None,
                patient.intubations is None,
                patient.covid_status is not None,
                patient.icu_admissions is None,
                patient.hiv is not None,
            )
            split_values.append(values)
            ids.append(patient_id)
        return np.array(split_values), np.array(ids)

    def get_patients(self):
        with open(self.dataset_cache, "rb") as fp:
            patients = pickle.load(fp)
        return patients

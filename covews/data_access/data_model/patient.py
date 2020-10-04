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


class Patient(object):
    GENDER_MALE = 0
    GENDER_FEMALE = 1

    RACE_UNKNOWN_OTHER = 0
    RACE_AFRICAN_AMERICAN = 1
    RACE_CAUCASIAN = 2
    RACE_ASIAN = 3

    ETHNICITY_OTHER = 0
    ETHNICITY_HISPANIC = 1

    def __init__(self, patient_id, birth_year, gender, date_of_death, region, covid_status, covid_tests=None,
                 admissions=None, icu_admissions=None, intubations=None, diagnoses=None, observations=None, dexa=None,
                 race=None, ethnicity=None):
        self.patient_id = patient_id
        self.age = 2020 - int(birth_year) if birth_year is not None else None
        self.gender = gender
        self.date_of_death = date_of_death
        self.region = region
        self.covid_status = covid_status
        self.covid_tests = covid_tests
        self.icu_admissions = icu_admissions
        self.hospital_admissions = admissions
        self.intubations = intubations
        self.dexa = dexa
        self.diagnoses = diagnoses
        self.observations = observations
        self.race = race
        self.ethnicity = ethnicity


class Procedure(object):
    def __init__(self, code, timestamp):
        self.code = code
        self.timestamp = timestamp

    def __repr__(self):
        return 'code={:}, time={:})'.format(self.code, self.timestamp)


class Diagnosis(object):
    def __init__(self, code, timestamp, coding_scheme="icd10"):
        self.code = code
        self.timestamp = timestamp
        self.coding_scheme = coding_scheme

    def __repr__(self):
        return 'code={:}, time={:})'.format(self.code, self.timestamp)


class Visit(object):
    def __init__(self, area_name, timestamp):
        self.area_name = area_name
        self.timestamp = timestamp

    def __repr__(self):
        return 'area_name={:}, time={:})'.format(self.area_name, self.timestamp)


class Observation(object):
    SMOKE_NEVER = 0
    SMOKE_PREVIOUSLY = 1
    SMOKE_CURRENTLY = 2
    SMOKE_UNKNOWN = 3

    def __init__(self, obs_type, value, unit, timestamp):
        self.type = obs_type
        self.value = value
        self.unit = unit
        self.timestamp = timestamp

    def __repr__(self):
        return 'type={:}, value={:}, unit={:}, time={:}'.format(self.type, self.value, self.unit, self.timestamp)


class LabTest(object):
    TEST_NEGATIVE = 0
    TEST_POSITIVE = 1
    TEST_UNKNOWN = 2

    def __init__(self, code, collected_timestamp, test_value, test_unit=None, result_timestamp=None):
        self.code = code
        self.timestamp = collected_timestamp
        self.test_value = test_value
        self.test_unit = test_unit
        self.result_timestamp = result_timestamp

    @property
    def value(self):
        return self.test_value

    def __repr__(self):
        return 'code={:}, time={:}, value={:})'.format(self.code, self.timestamp, self.test_value)

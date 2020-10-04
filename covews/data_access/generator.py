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
from covews.apps.util import info


LAST_ID = -1


def get_last_row_id():
    return LAST_ID


def inner_generator(x, y, p_ids, resample_with_replacement, shuffle, random_state,
                    num_samples, num_steps, num_losses, batch_size):
    global LAST_ID

    if resample_with_replacement:
        patient_ids = set(p_ids)
        num_patients = len(patient_ids)
        resampled_samples = random_state.randint(0, num_patients, size=num_patients)
        resampled_ids = np.array(list(sorted(list(patient_ids))))[resampled_samples]
        indices = []
        for resampled_id in resampled_ids:
            indices += list(np.where(p_ids == resampled_id)[0])
        x = x[indices]
        y = y[indices]
        p_ids = p_ids[indices]
        num_samples = len(x)
        num_steps = int(np.ceil(num_samples / float(batch_size)))
    else:
        resampled_samples = np.arange(num_samples)

    while True:
        if shuffle:
            if resample_with_replacement:
                samples = random_state.permutation(resampled_samples)
            else:
                samples = random_state.permutation(num_samples)
        else:
            samples = np.arange(num_samples)

        for _ in range(num_steps):
            batch_indices = samples[:batch_size]
            samples = samples[batch_size:]

            LAST_ID = [p_ids[idx] for idx in batch_indices]

            batch_x = np.array([x[idx] for idx in batch_indices])
            batch_y = y[batch_indices]

            if num_losses != 1:
                batch_y = [batch_y] * num_losses

            yield batch_x, batch_y


def make_generator(dataset, batch_size, shuffle=True, num_losses=1, resample_with_replacement=False, seed=909):
    x, y, p_ids = dataset

    num_steps = int(np.ceil(len(x) / float(batch_size)))
    num_samples = len(x)

    random_state = np.random.RandomState(seed)

    info("Loaded generator with", num_samples, "samples. Doing", num_steps, "steps of size", batch_size)

    return inner_generator(x, y, p_ids, resample_with_replacement, shuffle, random_state,
                           num_samples, num_steps, num_losses, batch_size), num_steps

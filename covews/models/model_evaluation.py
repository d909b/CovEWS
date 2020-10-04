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
from collections import OrderedDict
from covews.apps.util import info, warn
from covews.data_access.generator import get_last_row_id
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


class ModelEvaluation(object):
    @staticmethod
    def calculate_statistics_binary(y_true, y_pred, set_name, with_print, threshold=None):
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            if threshold is None:
                # Choose optimal threshold based on closest-to-top-left selection on ROC curve.
                dists = np.linalg.norm(np.stack((fpr, tpr)).T -
                                       np.repeat([[0., 1.]], fpr.shape[0], axis=0), axis=1)
                optimal_threshold_idx = np.argmin(dists)
                threshold = thresholds[optimal_threshold_idx]
            else:
                optimal_threshold_idx = None

            if with_print:
                info("Using threshold at", threshold)

            y_pred_thresholded = (y_pred >= threshold).astype(np.int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()
            auc_score = roc_auc_score(y_true, y_pred)

            if auc_score < 0.5:
                if with_print:
                    info("Inverting AUC.")
                auc_score = 1. - auc_score

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = auc(recall, precision)

            specificity = float(tn) / (tn + fp) if (tn + fp) != 0 else 0
            sensitivity = float(tp) / (tp + fn) if (tp + fn) != 0 else 0
            ppv = float(tp) / (tp + fp) if (tp + fp) != 0 else 0
            npv = float(tn) / (tn + fn) if (tn + fn) != 0 else 0

            if optimal_threshold_idx is not None:
                assert np.isclose(tpr[optimal_threshold_idx], sensitivity)
                assert np.isclose(1 - specificity, fpr[optimal_threshold_idx])

            accuracy = accuracy_score(y_true, y_pred_thresholded)
            f1_value = f1_score(y_true, y_pred_thresholded)

            if with_print:
                info(
                    "Performance on", set_name,
                    "AUROC =", auc_score,
                    ", with AUPRC =", auprc_score,
                    ", with accuracy =", accuracy,
                    ", with mean =", np.mean(y_true),
                    ", with f1 =", f1_value,
                    ", with specificity =", specificity,
                    ", with sensitivity =", sensitivity,
                    ", with PPV =", ppv,
                    ", with NPV =", npv,
                    ", n =", len(y_true),
                    ", TP =", tp,
                    ", FP =", fp,
                    ", TN =", tn,
                    ", FN =", fn,
                )
            return {
                "auroc": auc_score,
                "auprc": auprc_score,
                "f1": f1_value,
                "ppv": ppv,
                "npv": npv,
                "specificity": specificity,
                "sensitivity": sensitivity,
                "threshold": threshold,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            }
        except:
            if with_print:
                warn("Score calculation failed. Most likely, there was only one class present in y_true.")
            return None

    @staticmethod
    def get_alt_prediction(model, xi, start_offset, this_duration):
        if np.isclose(start_offset, this_duration):
            time_frame = this_duration
        else:
            time_frame = (start_offset - this_duration, start_offset)
        if hasattr(model, "base_estimator") and \
                hasattr(model.base_estimator, "prediction_time_frame_in_hours"):
            model.base_estimator.prediction_time_frame_in_hours = time_frame
        elif hasattr(model, "prediction_time_frame_in_hours"):
            model.prediction_time_frame_in_hours = time_frame
        y_pred_alt = model.predict_proba(xi)
        return y_pred_alt

    @staticmethod
    def prepare_dataset(model, x, y, y_pred, p, min_prediction_horizon=0.):
        test_x, test_y, new_y_pred, duration, start, start_offset = [], [], [], 0, [], 0
        for idx, (xi, (has_event, this_duration), y_pred_i, pi) in enumerate(zip(x, y, y_pred, p)):
            start_offset += this_duration

            if idx == len(p) - 1 or p[idx + 1] != pi:
                if this_duration >= min_prediction_horizon:
                    start.append(start_offset - this_duration)
                    test_x.append(xi)
                    test_y.append(has_event)
                    new_y_pred.append(y_pred_i)
                else:
                    tmp_idx, tmp_duration = idx, 0
                    while tmp_idx >= 0 and p[tmp_idx] == pi:
                        xi, (_, this_duration), y_pred_i = x[tmp_idx], y[tmp_idx], y_pred[tmp_idx]
                        tmp_duration += this_duration
                        if tmp_duration >= min_prediction_horizon:
                            start.append(start_offset - tmp_duration)
                            test_x.append(xi)
                            test_y.append(has_event)
                            new_y_pred.append(y_pred_i)
                            break
                        tmp_idx -= 1
                start_offset = 0
        return np.array(test_x), np.array(test_y), np.array(new_y_pred), np.array(start)

    @staticmethod
    def evaluate_classifier(model, generator, num_steps, set_name="Test set", num_multilabel_outputs=3,
                            selected_slices=list([-1]), with_print=True, threshold=None, output_directory="",
                            prediction_time_frames=[128.]):
        final_score_dict = OrderedDict()
        for j in range(num_multilabel_outputs):
            all_inputs, all_outputs, all_num_tasks, all_ids = [], [], [], []
            for _ in range(num_steps):
                generator_outputs = next(generator)
                if len(generator_outputs) == 3:
                    batch_input, labels_batch, sample_weight = generator_outputs
                else:
                    batch_input, labels_batch = generator_outputs

                ids = get_last_row_id()

                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(batch_input)[:, 1]
                else:
                    y_pred = model.predict(batch_input)
                all_inputs.append(batch_input)
                all_outputs.append((y_pred, labels_batch))
                all_ids.append(ids)

            for i, selected_slice in enumerate(selected_slices):
                all_x, y_pred, y_true, this_ids = [], [], [], []
                output_dim = model.output[i].shape[-1] if hasattr(model, "output") else 1
                for current_step in range(num_steps):
                    model_outputs, labels_batch = all_outputs[current_step]

                    if isinstance(model_outputs, list):
                        model_outputs = model_outputs[selected_slice]

                    if isinstance(labels_batch, list):
                        labels_batch = labels_batch[selected_slice]
                    all_x.append(all_inputs[current_step])
                    y_pred.append(model_outputs)
                    y_true.append(labels_batch)
                    this_ids.append(all_ids[current_step])

                all_x = np.concatenate(all_x, axis=0)
                y_true_before = np.concatenate(y_true, axis=0)
                y_pred_before = np.concatenate(y_pred, axis=0)
                p_ids_before = np.concatenate(this_ids, axis=0)

                score_dict = {}
                for t_idx, min_prediction_time_frame in enumerate(prediction_time_frames):
                    _, y_true, y_pred, starts = ModelEvaluation.prepare_dataset(model,
                                                                                all_x,
                                                                                y_true_before,
                                                                                y_pred_before,
                                                                                p_ids_before,
                                                                                min_prediction_time_frame)

                    if output_dim != 1:
                        y_true = y_true.reshape((-1, output_dim))
                        y_pred = y_pred.reshape((-1, output_dim))
                    else:
                        y_pred = np.squeeze(y_pred)

                    if (y_true.ndim == 2 and y_true.shape[-1] == 1) and \
                       (y_pred.ndim == 1 and y_pred.shape[0] == y_true.shape[0]):
                       y_pred = np.expand_dims(y_pred, axis=-1)

                    assert y_true.shape[-1] == y_pred.shape[-1]
                    assert y_true.shape[0] == y_pred.shape[0]

                    this_score_dict = ModelEvaluation.calculate_statistics_binary(y_true, y_pred,
                                                                                  set_name + ".{}.{:d}h"
                                                                                  .format(
                                                                                      j,
                                                                                      int(min_prediction_time_frame)
                                                                                  ),
                                                                                  with_print,
                                                                                  threshold=
                                                                                  threshold[j][t_idx]
                                                                                  if threshold is not None else
                                                                                  threshold)

                    if this_score_dict is not None:
                        for k, v in this_score_dict.items():
                            # Add prefix for each min time frame.
                            score_dict[k + "." + str(int(min_prediction_time_frame))] = v
                    else:
                        return None

                if score_dict is None:
                    final_score_dict = None
                else:
                    for k, v in score_dict.items():
                        final_score_dict["{}.".format(j) + k] = v
        return final_score_dict

"""
Modified from https://github.com/shreydesai/calibration
"""


import os
import numpy as np

def check_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    return

class ECE:
    def __init__(self, buckets=10) -> None:
        self.buckets = buckets

    def get_bucket_scores(self, y_score):
        """
        Organizes real-valued posterior probabilities into buckets.
        For example, if we have 10 buckets, the probabilities 0.0, 0.1,
        0.2 are placed into buckets 0 (0.0 <= p < 0.1), 1 (0.1 <= p < 0.2),
        and 2 (0.2 <= p < 0.3), respectively.
        """
        bucket_values = [[] for _ in range(self.buckets)]
        bucket_indices = [[] for _ in range(self.buckets)]
        for i, score in enumerate(y_score):
            for j in range(self.buckets):
                if score < float((j + 1) / self.buckets):
                    break
            bucket_values[j].append(score)
            bucket_indices[j].append(i)
        return (bucket_values, bucket_indices)
    
    def get_bucket_confidence(self, bucket_values):
        """
        Computes average confidence for each bucket. If a bucket does
        not have predictions, returns -1.
        """

        return [
            np.mean(bucket)
            if len(bucket) > 0 else -1.
            for bucket in bucket_values
        ]


    def get_bucket_accuracy(self, bucket_values, y_true, y_pred):
        """
        Computes accuracy for each bucket. If a bucket does
        not have predictions, returns -1.
        """

        per_bucket_correct = [
            [int(y_true[i] == y_pred[i]) for i in bucket]
            for bucket in bucket_values
        ]
        return [
            np.mean(bucket)
            if len(bucket) > 0 else -1.
            for bucket in per_bucket_correct
        ]
    
    def calculate_error(self, n_samples, bucket_values, bucket_confidence, bucket_accuracy):
        """
        Computes several metrics used to measure calibration error:
            - Expected Calibration Error (ECE): \sum_k (b_k / n) |acc(k) - conf(k)|
            - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
            - Total Calibration Error (TCE): \sum_k |acc(k) - conf(k)|
        """

        assert len(bucket_values) == len(bucket_confidence) == len(bucket_accuracy)
        assert sum(map(len, bucket_values)) == n_samples

        expected_error, max_error, total_error = 0., 0., 0.
        for (bucket, accuracy, confidence) in zip(
            bucket_values, bucket_accuracy, bucket_confidence
        ):
            if len(bucket) > 0:
                delta = abs(accuracy - confidence)
                expected_error += (len(bucket) / n_samples) * delta
                max_error = max(max_error, delta)
                total_error += delta
        return (expected_error * 100., max_error * 100., total_error * 100.)


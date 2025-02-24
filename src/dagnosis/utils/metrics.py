# third party

# dagnosis absolute
from dagnosis.utils.conformal import analyse_conformal_dict


class MetricsComputer:
    def compute_tp(self, conformal_dict):
        """Compute true positives from conformal predictions."""
        df = analyse_conformal_dict(conformal_dict)
        tp = len(df[df["inconsistent"] == True])

        n_samples = len(df)
        return tp, conformal_dict, n_samples

    def compute_metrics(self, conformal_dict_clean, conformal_dict_corrupted):
        """Compute precision, recall, and F1 score for a single method."""
        # Compute false positives on clean data
        fp, _, _ = self.compute_tp(conformal_dict_clean)

        # Compute true positives on corrupted data
        tp, conf_dict, n_samples = self.compute_tp(conformal_dict_corrupted)

        # Calculate metrics
        recall = tp / n_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 / (1 / precision + 1 / recall) if precision > 0 and recall > 0 else 0

        metrics = {"precision": precision, "recall": recall, "f1": f1}

        return metrics, conf_dict

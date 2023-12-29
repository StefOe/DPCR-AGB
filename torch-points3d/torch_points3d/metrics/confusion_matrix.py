import os

import numpy as np
import torch


class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions. 
    Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, cls_names):
        self.cls_names = np.array(cls_names)
        self.n_cls = len(cls_names)
        self.confusion_matrix = None

    @staticmethod
    def create_from_matrix(confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        matrix = ConfusionMatrix(confusion_matrix.shape[0])
        matrix.confusion_matrix = confusion_matrix
        return matrix

    def count_predicted_batch(self, ground_truth_vec, predicted):
        assert predicted.max() < self.n_cls
        batch_confusion = torch.bincount(
            self.n_cls * ground_truth_vec.int() + predicted, minlength=self.n_cls ** 2
        ).reshape(self.n_cls, self.n_cls)
        if self.confusion_matrix is None:
            self.confusion_matrix = batch_confusion
        else:
            self.confusion_matrix += batch_confusion

    def get_count(self, ground_truth, predicted):
        """labels are integers from 0 to number_of_labels-1"""
        return self.confusion_matrix[ground_truth][predicted]

    def get_confusion_matrix(self):
        """returns list of lists of integers; use it as result[ground_truth][predicted]
            to know how many samples of class ground_truth were reported as class predicted"""
        return self.confusion_matrix

    def get_stats(self):
        cmat = self.confusion_matrix
        stats = {}
        class_stats = {}
        numel = cmat.sum(1)
        mask = numel > 0
        if mask.sum() == 0:  # nothing to log
            return stats
        tp = torch.diag(cmat)[mask]
        stats["tp"] = tp.sum().item()
        fp = (cmat.sum(0)[mask] - tp)
        stats["fp"] = fp.sum().item()
        fn = (cmat.sum(1)[mask] - tp)
        stats["acc"] = (tp.sum() / numel.sum()).item()

        # macro statistics
        acc = (tp / numel[mask])
        stats["macc"] = acc.mean().item()

        precision = tp / (tp + fp + torch.finfo(torch.float32).eps)
        stats["precision"] = precision.mean().item()

        recall = tp / (tp + fn + torch.finfo(torch.float32).eps)
        stats["recall"] = recall.mean().item()

        f1 = 2 * ((precision * recall) / (precision + recall + torch.finfo(torch.float32).eps))
        stats["f1"] = f1.mean().item()

        # class stats
        for i, cls_name in enumerate(self.cls_names[mask.cpu()]):
            class_stats["acc", cls_name] = acc[i].item()
            class_stats["tp", cls_name] = tp[i].item()
            class_stats["recall", cls_name] = recall[i].item()
            class_stats["precision", cls_name] = precision[i].item()
            class_stats["f1", cls_name] = f1[i].item()

        """
        # normalize conf matrix
        cmat_sum = cmat.sum(axis=1, keepdim=True)
        cmat_sum += cmat_sum == 0  # avoid nans by displaying 0
        cmatn = cmat / cmat_sum
        """
        return stats, class_stats, cmat


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=5)

    template_path = os.path.join(path2save, "{}.svg")
    # PRECISION
    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    # RECALL
    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve

__all__ = ["mean_PPVn", "find_optimal_threshold", "evaluate_metrics"]

def mean_PPVn(values_true, values_pred, topk=None):
    '''
    NOTE: I cross-checked with the official implementation of
    "Deep neural networks predict class I major histocompatibility complex epitope presentation and
    transfer learn neoepitope immunogenicity"

    https://github.com/KarchinLab/bigmhc/blob/6d894cb359fab57b7ccdd7b688a1ac15d44063bd/nb/makefigs.ipynb#L156
    '''

    assert len(values_true) == len(values_pred)

    # Sorting by score in descending order
    sorting_idx = np.argsort(values_pred)[::-1]
    values_true = values_true[sorting_idx]

    # Calculate cumulative true positives
    cum_true_positives = np.cumsum(values_true)

    # Total possible positives in top n
    total_predictions = np.arange(1, len(values_true) + 1)

    # Calculating PPVn
    ppvn = cum_true_positives / total_predictions

    # Mean PPVn across all n
    num_positives = int(values_true.sum())

    if topk is None:
        mean_ppvn = np.mean(ppvn[:num_positives])
    elif topk >= len(ppvn[:num_positives]):
        mean_ppvn = np.mean(ppvn[:num_positives])
        print(f"`mean_PPVn`: topk ({topk}) bigger than number of positive samples ({num_positives}).")
    else:
        mean_ppvn = np.mean(ppvn[:num_positives][:topk])

    return mean_ppvn

def find_optimal_threshold(y_true, y_prob):
    '''
    Maximizing Youden's J statistic (sensitivity + specificity - 1).
    '''
    # Compute the ROC curve and the corresponding false positive rates (FPR) and true positive rates (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Calculate Youden's J statistic for each threshold
    youden_j = tpr - fpr

    # Find the index of the maximum Youden's J statistic
    optimal_idx = np.argmax(youden_j)

    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_idx]

    assert optimal_threshold >= 0 and optimal_threshold <= 1

    return optimal_threshold


def evaluate_metrics(true_targets, predicted_probs, optimal_threshold):
    roc_auc = roc_auc_score(true_targets, predicted_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(true_targets, predicted_probs)
    pr_auc = auc(recall_curve, precision_curve)
    accuracy = accuracy_score(true_targets, predicted_probs >= 0.5)
    accuracy_op = accuracy_score(true_targets, predicted_probs >= optimal_threshold)
    f1 = f1_score(true_targets, predicted_probs >= 0.5)
    f1_op = f1_score(true_targets, predicted_probs >= optimal_threshold)
    precision = precision_score(true_targets, predicted_probs >= 0.5)
    precision_op = precision_score(true_targets, predicted_probs >= optimal_threshold)
    recall = recall_score(true_targets, predicted_probs >= 0.5)
    recall_op = recall_score(true_targets, predicted_probs >= optimal_threshold)
    ppvn = mean_PPVn(true_targets, predicted_probs >= 0.5)
    ppvn_op = mean_PPVn(true_targets, predicted_probs >= optimal_threshold)
    ppv30 = mean_PPVn(true_targets, predicted_probs >= 0.5, topk=30)
    ppv30_op = mean_PPVn(true_targets, predicted_probs >= optimal_threshold, topk=30)

    print('metrics')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')
    print(f'Accuracy @0.5: {accuracy:.4f}')
    print(f'Accuracy @op: {accuracy_op:.4f}')
    print(f'F1 Score @0.5: {f1:.4f}')
    print(f'F1 Score @op: {f1_op:.4f}')
    print(f'Precision @0.5: {precision:.4f}')
    print(f'Precision @op: {precision_op:.4f}')
    print(f'Recall @0.5: {recall:.4f}')
    print(f'Recall @op: {recall_op:.4f}')
    print(f'Mean PPVn @0.5: {ppvn:.4f}')
    print(f'Mean PPVn @op: {ppvn_op:.4f}')
    print(f'PPVn (n=30) @0.5: {ppv30:.4f}')
    print(f'PPVn (n=30) @op: {ppv30_op:.4f}')
    
    output_dict = {
        "optimal_threshold": optimal_threshold,
        "accuracy": accuracy,
        "accuracy_op": accuracy_op,
        "f1": f1,
        "f1_op": f1_op,
        "precision": precision,
        "precision_op": precision_op,
        "recall": recall,
        "recall_op": recall_op,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ppvn": ppvn,
        "ppvn_op": ppvn_op,
        "ppv30": ppv30,
        "ppv30_op": ppv30_op,
    }

    return output_dict

if __name__ == '__main__':
    values_pred = np.random.rand(100)
    values_true = np.uint8( np.random.rand(100) > 0.5)
    values_true[20:50] = np.uint8(values_pred[20:50] > 0.5)
    ppvn = mean_PPVn(values_true, values_pred, topk=None)
    ppv30 = mean_PPVn(values_true, values_pred, topk=30)
    print('Mean PPVn:', ppvn, '\nPPVn (n=30)', ppv30)

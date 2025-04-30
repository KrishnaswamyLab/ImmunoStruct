import numpy as np
import torch
from .metric import evaluate_metrics, find_optimal_threshold
from .clinical_validation import inference_clinical_only

__all__ = ["inference", "inference_comparative","inference_clinical_only"]


def inference(config, model, data_loader, device, clinical_loader=None, return_raw_preds=False, fig_save_folder=None, optimal_threshold=None):
    model.eval()
    true_targets = []
    predicted_probs = []  # Store raw probabilities for ROC AUC calculation

    with torch.no_grad():
        for graph_data, sequence_data, target, peptide_property in data_loader:
            graph_data = graph_data.to(device)
            sequence_data, target, peptide_property = sequence_data.to(device), target.to(device), peptide_property.to(device)

            _, _, _, final_output = model(graph_data, sequence_data, peptide_property)

            # Convert to probabilities
            probs = torch.sigmoid(final_output).squeeze()

            # Handle the case where probs is a scalar
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)  # Make it a 1-element tensor

            probs = probs.detach().cpu().numpy()

            true_targets.extend(target.detach().cpu().numpy())
            predicted_probs.extend(probs.tolist())  # Convert to list before extending

    # Calculate metrics
    true_targets = np.array(true_targets)
    predicted_probs = np.array(predicted_probs)

    if optimal_threshold is None:
        optimal_threshold = find_optimal_threshold(true_targets, predicted_probs)

    output_dict = evaluate_metrics(true_targets, predicted_probs, optimal_threshold)

    if return_raw_preds:
        output_dict["predicted_probs"] = predicted_probs
        output_dict["true_targets"] = true_targets

    if clinical_loader:
        clinical_output = inference_clinical_only(config, model, device, clinical_loader=clinical_loader, fig_save_folder=fig_save_folder)
        output_dict.update(clinical_output)

    return output_dict


def inference_comparative(config, model, data_loader, device, clinical_loader=None, return_raw_preds=False, fig_save_folder=None, optimal_threshold=None):
    model.eval()
    true_targets = []
    predicted_probs = []  # Store raw probabilities for ROC AUC calculation

    with torch.no_grad():
        for graph_data_pair, sequence_data_pair, target, peptide_property_pair in data_loader:

            graph_data_cancer, graph_data_wt = graph_data_pair
            sequence_data_cancer, sequence_data_wt = sequence_data_pair
            peptide_property_cancer, peptide_property_wt = peptide_property_pair

            graph_data_cancer, graph_data_wt = graph_data_cancer.to(device), graph_data_wt.to(device)
            sequence_data_cancer, sequence_data_wt = sequence_data_cancer.to(device), sequence_data_wt.to(device)
            peptide_property_cancer, peptide_property_wt = peptide_property_cancer.to(device), peptide_property_wt.to(device)
            target = target.to(device)

            _, _, _, _, final_output = model.forward_comparative((graph_data_cancer, graph_data_wt),
                                                                 (sequence_data_cancer, sequence_data_wt),
                                                                 (peptide_property_cancer, peptide_property_wt))

            # Convert to probabilities
            probs = torch.sigmoid(final_output).squeeze()

            # Handle the case where probs is a scalar
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)  # Make it a 1-element tensor

            probs = probs.detach().cpu().numpy()

            true_targets.extend(target.detach().cpu().numpy())
            predicted_probs.extend(probs.tolist())  # Convert to list before extending

    # Calculate metrics
    true_targets = np.array(true_targets)
    predicted_probs = np.array(predicted_probs)

    if optimal_threshold is None:
        optimal_threshold = find_optimal_threshold(true_targets, predicted_probs)

    output_dict = evaluate_metrics(true_targets, predicted_probs, optimal_threshold)

    if return_raw_preds:
        output_dict["predicted_probs"] = predicted_probs
        output_dict["true_targets"] = true_targets

    if clinical_loader:
        clinical_output = inference_clinical_only(config, model, device, clinical_loader=clinical_loader, fig_save_folder=fig_save_folder)
        output_dict.update(clinical_output)

    return output_dict

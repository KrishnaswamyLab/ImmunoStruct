import os
import platform
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

__all__ = ["clinical_pvalues", "inference_clinical_only"]


def convert_patient_code(patient_code):
    mapping = {
        'mUC': 'BC',
        'MM': 'Neye',
        'RH': 'RH'
        # You can add more mappings here if needed
    }

    prefix, number = patient_code.split('-')
    if prefix in mapping:
        return f"{mapping[prefix]}-{number}"
    else:
        return patient_code


def clinical_pvalues(predicted_probs_clinical, fig_save_path=None):

    current_platform = platform.system()
    if current_platform == 'Windows':
        ROOT_DIR = "\\".join(os.path.realpath(__file__).split("\\")[:-3])
    elif current_platform == 'Linux':
        ROOT_DIR = "/".join(os.path.realpath(__file__).split("/")[:-3])
    else:
        raise NotImplementedError(f"Cannot support the current platform: {current_platform}.")

    seq_df = pd.read_table(os.path.join(ROOT_DIR, 'data', 'hadrup_cancer_df_29K.txt'))
    clin_df = pd.read_table(os.path.join(ROOT_DIR, 'data', 'All_samples_clinical.txt'))

    seq_df['patient_ID'] = seq_df['patient'].apply(convert_patient_code)

    # This is where to insert the predicted labels for each pMHC sample.
    seq_df['ImmunoStruct_predicted'] = predicted_probs_clinical

    # Drop the rows that have nan values.
    seq_df = seq_df.dropna(subset=['ImmunoStruct_predicted'])

    # Aggregate ImmunoStruct_predicted by patient_ID to create ImmunoStruct_predicted_load
    # For example, summing the predicted scores for each patient
    immunostruct_load_df = seq_df.groupby('patient_ID')['ImmunoStruct_predicted'].sum().reset_index()
    immunostruct_load_df = immunostruct_load_df.sort_values('patient_ID')

    # Replace commas with decimal points in the PFS.Time and OS.Time columns
    clin_df['PFS.Time'] = clin_df['PFS.Time'].astype(str).str.replace(',', '.')
    clin_df['OS.Time'] = clin_df['OS.Time'].astype(str).str.replace(',', '.')

    # Optionally, convert the columns back to floats
    clin_df['PFS.Time'] = clin_df['PFS.Time'].astype(float)
    clin_df['OS.Time'] = clin_df['OS.Time'].astype(float)

    clin_df['patient_ID'] = clin_df['Patient']
    clin_df = clin_df.sort_values('patient_ID')

    # Merge this new ImmunoStruct_predicted_load back into the main clinical dataframe
    clin_df['ImmunoStruct_predicted_load'] = immunostruct_load_df['ImmunoStruct_predicted'].tolist()
    clin_df.to_csv(os.path.join(ROOT_DIR, 'results', 'clinical_results.csv'))
    
    # Define a threshold to split ImmunoStruct_predicted_load into low and high groups
    low_group_threshold = np.percentile(clin_df['ImmunoStruct_predicted_load'], 50)
    high_group_threshold = np.percentile(clin_df['ImmunoStruct_predicted_load'], 50)
    assert low_group_threshold <= high_group_threshold
    low_immuno_struct = clin_df[clin_df['ImmunoStruct_predicted_load'] <= low_group_threshold]
    high_immuno_struct = clin_df[clin_df['ImmunoStruct_predicted_load'] >= high_group_threshold]

    # Perform the log-rank test for OS
    result_os = logrank_test(low_immuno_struct['OS.Time'], high_immuno_struct['OS.Time'],
                             event_observed_A=low_immuno_struct['OS.Event'],
                             event_observed_B=high_immuno_struct['OS.Event'])

    # Get the OS p-value
    os_p_value = result_os.p_value

    # Perform the log-rank test for PFS
    result_pfs = logrank_test(low_immuno_struct['PFS.Time'], high_immuno_struct['PFS.Time'],
                              event_observed_A=low_immuno_struct['PFS.Event'],
                              event_observed_B=high_immuno_struct['PFS.Event'])

    # Get the PFS p-value
    pfs_p_value = result_pfs.p_value

    if fig_save_path is not None:
        if os_p_value > 0.1 and pfs_p_value > 0.1:
            print('Not plotting the clinical validation figures since both p-values are higher than 0.1.')
        else:
            os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
            plot_clinical_validation(low_immuno_struct, high_immuno_struct, fig_save_path)

    # Output the p-values
    return os_p_value, pfs_p_value


def plot_clinical_validation(low_immuno_struct, high_immuno_struct, fig_save_path):
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'serif'

    # Create three subplots: OS, PFS, and RECIST
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Initialize the Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Fit and plot for low mutational load (OS)
    kmf.fit(low_immuno_struct['OS.Time'], event_observed=low_immuno_struct['OS.Event'], label='Low Predicted Immunogenicity')
    kmf.plot_survival_function(ax=axes[0], ci_show=False, linewidth=2.5, color='mediumblue')

    # Fit and plot for high mutational load (OS)
    kmf.fit(high_immuno_struct['OS.Time'], event_observed=high_immuno_struct['OS.Event'], label='High Predicted Immunogenicity')
    kmf.plot_survival_function(ax=axes[0], ci_show=False, linewidth=2.5, color='firebrick')

    # Perform log-rank test for OS
    result = logrank_test(low_immuno_struct['OS.Time'], high_immuno_struct['OS.Time'],
                          event_observed_A=low_immuno_struct['OS.Event'], event_observed_B=high_immuno_struct['OS.Event'])
    p_value = result.p_value

    # Add p-value annotation for OS plot
    axes[0].text(0.6, 0.12, f'p-value = {p_value:.4f}', transform=axes[0].transAxes, fontsize=16)

    # Set title and labels for OS plot
    axes[0].set_title('OS Kaplan-Meier Curve Stratified by ImmunoStruct', fontsize=16)
    axes[0].set_xlabel('Time (months)', fontsize=18)
    axes[0].set_ylabel('Survival Probability', fontsize=18)

    # Fit and plot for low mutational load (PFS)
    kmf.fit(low_immuno_struct['PFS.Time'], event_observed=low_immuno_struct['PFS.Event'], label='Low Predicted Immunogenicity')
    kmf.plot_survival_function(ax=axes[1], ci_show=False, linewidth=2.5, color='mediumblue')

    # Fit and plot for high mutational load (PFS)
    kmf.fit(high_immuno_struct['PFS.Time'], event_observed=high_immuno_struct['PFS.Event'], label='High Predicted Immunogenicity')
    kmf.plot_survival_function(ax=axes[1], ci_show=False, linewidth=2.5, color='firebrick')

    # Perform log-rank test for PFS
    result_pfs = logrank_test(low_immuno_struct['PFS.Time'], high_immuno_struct['PFS.Time'],
                              event_observed_A=low_immuno_struct['PFS.Event'], event_observed_B=high_immuno_struct['PFS.Event'])
    p_value_pfs = result_pfs.p_value

    # Add p-value annotation for PFS plot
    axes[1].text(0.6, 0.12, f'p-value = {p_value_pfs:.4f}', transform=axes[1].transAxes, fontsize=16)

    # Set title and labels for PFS plot
    axes[1].set_title('PFS Kaplan-Meier Curve Stratified by ImmunoStruct', fontsize=16)
    axes[1].set_xlabel('Time (months)', fontsize=18)
    axes[1].set_ylabel('Survival Probability', fontsize=18)

    # Increase font sizes of the legends
    axes[0].legend(fontsize=16)
    axes[1].legend(fontsize=16)

    # Display the plots
    fig.tight_layout(pad=2)
    fig.savefig(fig_save_path)

def inference_clinical_only(config, model, device, clinical_loader=None, fig_save_folder=None):
    model.eval()
    predicted_probs_clinical = []  # Store raw probabilities for clinical p values
    output_dict = {}

    with torch.no_grad():
        # NOTE: Currently, we do not have the wt sequences for the clinical validation dataset.
        # In this case, the cancer features will be copied to the wt features,
        # It will be hacky if --use-wt-for-downstream is true.
        for graph_data, sequence_data, target, peptide_property in clinical_loader:

            graph_data = graph_data.to(device)
            sequence_data, peptide_property = sequence_data.to(device), peptide_property.to(device)

            if config.self_supervision:
                _, _, _, final_output, _ = model(graph_data, sequence_data, peptide_property)
            else:
                _, _, _, final_output = model(graph_data, sequence_data, peptide_property)

            # Convert to probabilities
            probs = torch.sigmoid(final_output).squeeze()

            # Handle the case where probs is a scalar
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)  # Make it a 1-element tensor

            probs = probs.detach().cpu().numpy()
            peptide_property = peptide_property.detach().cpu().numpy()

            nan_indices = np.asarray(np.where(np.isnan(peptide_property[:, 0]))).reshape(-1)
            probs[nan_indices] = np.nan

            predicted_probs_clinical.extend(probs.tolist())  # Convert to list before extending

    predicted_probs_clinical = np.array(predicted_probs_clinical)
    os_p_value, pfs_p_value = clinical_pvalues(predicted_probs_clinical,
                                                fig_save_path=os.path.join(fig_save_folder, 'clinical_p_value.png'))

    print('clinical metrics')
    print(f'OS p-value: {os_p_value:.4f}')
    print(f'PFS p-value: {pfs_p_value:.4f}')
    output_dict["os_p_value"] = os_p_value
    output_dict["pfs_p_value"] = pfs_p_value

    return output_dict

if __name__ == '__main__':
    os_p_value, pfs_p_value = clinical_pvalues([1 for _ in range(29484)], fig_save_path='test_clinical_validation.png')
    print(f'OS p-value: {os_p_value:.4f}')
    print(f'PFS p-value: {pfs_p_value:.4f}')

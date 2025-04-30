import os
import argparse
import torch
import numpy as np
from dgl.dataloading import GraphDataLoader

from data import ImmunoPredInferDataset, collate, SplitDataset, ImmunoPredInferDatasetComparative
from models.mapping import model_map
from utils import seed_everything, update_paths
from procedures import inference, inference_comparative

# python infer_IEDB_or_Cancer.py --model HybridModelv2 --model-dir D:/Edward/Smita_Lab_Work/ImmunoPred/results/PropIEDB_ImmunoIEDB_final --model-filename HybridModelv2-lr_pt_0.001-lr_ft_0.0001-ep_40-bs_150-fseq_True-seql_True-fs_23-cs_3-seed_2_finetune.pt --full-sequence --infer_dataset IEDB --hla-path D:\Edward\Smita_Lab_Work\HLA_27_seqs_csv.csv --graph-dir-IEDB D:\Edward\Smita_Lab_Work\IEDB\PyGs --property-path-IEDB D:\Edward\Smita_Lab_Work\complete_score_Mprops_1_2_smoothed_sasa_v2.txt
# python infer_IEDB_or_Cancer.py --model HybridModelv2_Comparative --model-dir D:/Edward/Smita_Lab_Work/ImmunoPred/results/PropIEDB_PropCancer_ImmunoCancer_final --model-filename HybridModelv2_Comparative-wtds_True-lr_pt_0.001-lr_ft_0.0001-cc_0-ssl_False-ep_40-bs_128-fseq_True-seql_True-fs_23-cs_3-seed_1_finetune.pt --full-sequence --infer_dataset Cancer --hla-path D:\Edward\Smita_Lab_Work\HLA_27_seqs_csv.csv --graph-dir-cancer D:\Edward\Smita_Lab_Work\Cancer_2 --property-path-cancer D:\Edward\Smita_Lab_Work\cedar_data_final_with_mprop1_mprop2_v2.txt --property-path-wildtype D:\Edward\Smita_Lab_Work\cedar_data_final_WILD_TYPE_with_mprop1_mprop2_v2_has_mprop10_11_v2.txt --graph-dir-wildtype D:\Edward\Smita_Lab_Work\graph_pyg_Cancer_WT --comparative --use-wt-for-downstream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry point.")
    # Model parameters
    parser.add_argument("--model", default="StructureModel", type=str)
    parser.add_argument("--model-dir", default="$ROOT/results/PropIEDB_PropCancer_ImmunoCancer/", type=str)
    parser.add_argument("--model-filename", default="HybridModel_Comparative-wtds_False-lr_pt_0.001-lr_ft_0.0001-cc_0.01-ssl_False-ep_40-bs_128-fseq_True-seql_False-fs_23-cs_3-seed_1_finetune.pt")
    parser.add_argument("--use-wt-for-downstream", action='store_true')

    # Dataset parameters
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--full-sequence", action="store_true")
    parser.add_argument("--infer_dataset", default="IEDB", type=str) # IEDB or Cancer
    parser.add_argument("--comparative", action="store_true") # whether dataset includes comparative data

    # Training parameters
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)

    # Data paths
    parser.add_argument("--graph-dir-IEDB", default="$ROOT/data/graph_pyg_IEDB/", type=str)
    parser.add_argument("--graph-dir-cancer", default="$ROOT/data/graph_pyg_Cancer/", type=str)
    parser.add_argument("--graph-dir-wildtype", default="$ROOT/data/graph_pyg_Cancer_WT/", type=str) # only used for comparative
    parser.add_argument("--property-path-IEDB", default="$ROOT/data/complete_score_Mprops_1_2_smoothed_sasa_v2.txt", type=str)
    parser.add_argument("--property-path-cancer", default="$ROOT/data/cedar_data_final_with_mprop1_mprop2_v2.txt", type=str)
    parser.add_argument("--property-path-wildtype", default="$ROOT/data/cedar_data_final_WILD_TYPE_with_mprop1_mprop2_v2.txt", type=str) # only used for comparative
    parser.add_argument("--hla-path", default="$ROOT/data/HLA_27_seqs_csv.csv", type=str)

    config = parser.parse_args()

    # Update path.
    update_paths(config)

    # Model save paths.
    model_path = os.path.join(config.model_dir, config.model_filename)
    print(f'SAVED MODEL PATH: {model_path}')

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    # Define model.
    # input_dim = sequence length (11 for peptide, 283 for sequence) * embedding
    input_dim = 283 * 21 if config.full_sequence else 11 * 21
    model = model_map[config.model](vae_input_dim=input_dim, device=device, use_wt_for_downstream=config.use_wt_for_downstream)
    model.load_trained(model_path, new_head=False, map_location=device)
    model.to(device)

    print('Retrieving dataset')
    dataset_ft = None
    if config.infer_dataset == "IEDB": 
        dataset_ft = ImmunoPredInferDataset(config,
                                                graph_directory=config.graph_dir_IEDB,
                                                property_path=config.property_path_IEDB,
                                                hla_path=config.hla_path)
    else:
        if config.comparative:
            dataset_ft = ImmunoPredInferDatasetComparative(config,
                                                            graph_directory_cancer=config.graph_dir_cancer,
                                                            property_path_cancer=config.property_path_cancer,
                                                            graph_directory_wt=config.graph_dir_wildtype,
                                                            property_path_wt=config.property_path_wildtype,
                                                            hla_path=config.hla_path)

        else:
            dataset_ft = ImmunoPredInferDataset(config,
                                                    graph_directory=config.graph_dir_cancer,
                                                    property_path=config.property_path_cancer,
                                                    hla_path=config.hla_path)

    _, _, test_dataset_ft = torch.utils.data.random_split(dataset_ft, [0.8, 0.1, 0.1], generator)

    # `binary=True` --> Using immunogenicity.
    if config.comparative:
        test_split_dataset = SplitDataset(test_dataset_ft, "test", binary=True, comparative=True, full=config.full_sequence)
    else:
        test_split_dataset = SplitDataset(test_dataset_ft, "test", binary=True, full=config.full_sequence)

    test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)

    print('running inference')
    if config.comparative:
        test_stats = inference_comparative(config, model, test_loader, device, return_raw_preds=True)
    else:
        test_stats = inference(config, model, test_loader, device, return_raw_preds=True)

    sequences = dataset_ft.raw_full_sequence[np.array(test_dataset_ft.indices)]
    np.savetxt(f"{config.model_dir}/predictions_PPI.txt", np.stack([test_stats["predicted_probs"], test_stats["true_targets"], sequences], axis=1),
               delimiter="\t", fmt="%s", header="Predicted Immunogenicity\tTrue Immunogenicity\tSequence", comments="")
    print('DONE')

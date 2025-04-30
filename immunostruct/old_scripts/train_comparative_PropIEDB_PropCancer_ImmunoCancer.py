import os
import argparse
import torch
import wandb
from dgl.dataloading import GraphDataLoader

from data import ImmunoPredDataset, ImmunoPredDatasetComparative, collate, SplitDataset, ExtendedDataset, collate_amino_acid, ClinicalDataset
from models.mapping import model_map
from utils import Losses, seed_everything, LinearWarmupCosineAnnealingLR, update_paths
from procedures import train_model, train_model_SSL, inference_comparative, train_model_comparative, train_model_comparative_SSL, inference_comparative_SSL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry point.")
    parser.add_argument("--model", default="HybridModelv2_Comparative", type=str)
    parser.add_argument("--use-wt-for-downstream", action='store_true')
    parser.add_argument("--learning-rate-pretrain", default=1e-3, type=float)
    parser.add_argument("--learning-rate-finetune", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--full-sequence", action="store_true")
    parser.add_argument("--sequence-loss", action="store_true")
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--min-finetuning-batches", default=64, type=int)
    parser.add_argument("--model-save-dir", default="$ROOT/checkpoints/comparative_PropIEDB_PropCancer_ImmunoCancer/", type=str)
    parser.add_argument("--figure-save-dir", default="$ROOT/figures/comparative_PropIEDB_PropCancer_ImmunoCancer/", type=str)
    parser.add_argument("--graph-dir-IEDB", default="$ROOT/data/graph_pyg_IEDB/", type=str)
    parser.add_argument("--graph-dir-cancer", default="$ROOT/data/graph_pyg_Cancer/", type=str)
    parser.add_argument("--graph-dir-wildtype", default="$ROOT/data/graph_pyg_Cancer_WT/", type=str)
    parser.add_argument("--graph-dir-clinical", default="$ROOT/data/graph_pyg_Clinical/", type=str)
    parser.add_argument("--property-path-IEDB", default="$ROOT/data/complete_score_Mprops_1_2_smoothed_sasa_v2.txt", type=str)
    parser.add_argument("--property-path-cancer", default="$ROOT/data/cedar_data_final_with_mprop1_mprop2_v2.txt", type=str)
    parser.add_argument("--property-path-wildtype", default="$ROOT/data/cedar_data_final_WILD_TYPE_with_mprop1_mprop2_v2.txt", type=str)
    parser.add_argument("--seq-path-clinical", default="$ROOT/data/hadrup_cancer_df_29K.txt", type=str)
    parser.add_argument("--hla-path", default="$ROOT/data/HLA_27_seqs_csv.csv", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--wandb-username", default=None, type=str)
    parser.add_argument("--sequence-pad-count", default=0, type=int)
    parser.add_argument("--structure-pad-count", default=0, type=int)
    parser.add_argument("--self-supervision", action="store_true") # if the model is an SSL model this should be set to true
    parser.add_argument("--coeff-contrastive", default=0, type=float)
    config = parser.parse_args()

    update_paths(config)

    # Model save paths.
    model_str = f"{config.model}-wtds_{config.use_wt_for_downstream}" + \
        f"-lr_pt_{config.learning_rate_pretrain}-lr_ft_{config.learning_rate_finetune}" + \
        f"-cc_{config.coeff_contrastive}-ssl_{config.self_supervision}" + \
        f"-ep_{config.num_epochs}-bs_{config.batch_size}-fseq_{config.full_sequence}-seql_{config.sequence_loss}" + \
        f"-fs_{config.feature_size}-cs_{config.coord_size}-seed_{config.seed}"
    config.model_save_path_pretrain = os.path.join(config.model_save_dir, model_str + "_pretrain.pt")
    config.model_save_path_finetune = os.path.join(config.model_save_dir, model_str + "_finetune.pt")
    config.fig_save_folder = os.path.join(config.figure_save_dir, model_str)

    # Weights and biases.
    wandb.init(
        project="ImmunoPred-Cancer-Paper-2", # set the wandb project where this run will be logged
        entity=config.wandb_username,
        name=f"Comparative-PropIEDB_PropCancer_ImmunoCancer:{model_str}",  # display name of the run
        config=config,   # track hyperparameters and run metadata
    )
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    # Define model.
    # input_dim = sequence length (11 for peptide, 283 for sequence) * embedding
    input_dim = 283 * 21 if config.full_sequence else 11 * 21
    model = model_map[config.model](vae_input_dim=input_dim, device=device, use_wt_for_downstream=config.use_wt_for_downstream)
    model.to(device)

    # Pretraining and Finetuning datasets.
    dataset_pt1 = ImmunoPredDataset(config,
                                    graph_directory=config.graph_dir_IEDB,
                                    property_path=config.property_path_IEDB,
                                    hla_path=config.hla_path)
    dataset_pt2 = dataset_ft = ImmunoPredDatasetComparative(config,
                                                            graph_directory_cancer=config.graph_dir_cancer,
                                                            property_path_cancer=config.property_path_cancer,
                                                            graph_directory_wt=config.graph_dir_wildtype,
                                                            property_path_wt=config.property_path_wildtype,
                                                            hla_path=config.hla_path)
    clinical_dataset = ClinicalDataset(config,
                                       graph_directory=config.graph_dir_clinical,
                                       seq_path=config.seq_path_clinical)

    train_dataset_pt1, val_dataset_pt1, test_dataset_pt1 = torch.utils.data.random_split(dataset_pt1, [0.8, 0.1, 0.1], generator)
    train_dataset_pt2, val_dataset_pt2, test_dataset_pt2 = torch.utils.data.random_split(dataset_pt2, [0.8, 0.1, 0.1], generator)
    train_dataset_ft, val_dataset_ft, test_dataset_ft = train_dataset_pt2, val_dataset_pt2, test_dataset_pt2
    print("Pretraining stage 1 train/val/test size:", len(train_dataset_pt1), len(val_dataset_pt1), len(test_dataset_pt1))
    print("Pretraining stage 2 train/val/test size:", len(train_dataset_pt2), len(val_dataset_pt2), len(test_dataset_pt2))
    print("Finetuning train/val/test size:", len(train_dataset_ft), len(val_dataset_ft), len(test_dataset_ft))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_pretrain, weight_decay=1e-6)
    losses = Losses(input_dim, dataset_pt1.class_weights, sequence=config.sequence_loss)


    # `binary=False` --> Using property.
    train_split_dataset = SplitDataset(train_dataset_pt1, "train", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_pt1, "val", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_pt1, "test", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True, num_workers=config.num_workers)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss_SSL)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True, num_workers=config.num_workers)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss)

    print("DONE PRE-TRAINING Stage 1")
    del train_split_dataset, val_split_dataset, test_split_dataset
    del train_loader, val_loader, test_loader
    del optimizer

    model.load_trained(config.model_save_path_pretrain, new_head=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_pretrain, weight_decay=1e-6)
    losses = Losses(input_dim, dataset_pt2.class_weights, sequence=config.sequence_loss)

    # `binary=False` --> Using property.
    train_split_dataset = SplitDataset(train_dataset_pt2, "train", binary=False, full=config.full_sequence, comparative=True, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_pt2, "val", binary=False, full=config.full_sequence, comparative=True, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_pt2, "test", binary=False, full=config.full_sequence, comparative=True, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True, num_workers=config.num_workers)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model_comparative_SSL(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss_SSL)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True, num_workers=config.num_workers)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model_comparative(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss)

    print("DONE PRE-TRAING Stage 2")
    del train_split_dataset, val_split_dataset, test_split_dataset
    del train_loader, val_loader, test_loader
    del optimizer

    model.load_trained(config.model_save_path_pretrain, new_head=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_finetune, weight_decay=1e-6)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=config.num_epochs//4,
                                              warmup_start_lr=config.learning_rate_finetune/100,
                                              max_epochs=config.num_epochs)
    losses = Losses(input_dim, dataset_ft.class_weights, sequence=config.sequence_loss)

    # `binary=True` --> Using immunogenicity.
    train_split_dataset = SplitDataset(train_dataset_ft, "train", binary=True, full=config.full_sequence, comparative=True, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_ft, "val", binary=True, full=config.full_sequence, comparative=True, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_ft, "test", binary=True, full=config.full_sequence, comparative=True, return_amino_acid=config.self_supervision)
    
    clinical_dataset = SplitDataset(clinical_dataset, "infer", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=False)
    clinical_loader = GraphDataLoader(clinical_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)

    min_datapoints = config.min_finetuning_batches * config.batch_size
    if len(train_split_dataset) < min_datapoints:
        train_split_dataset = ExtendedDataset(train_split_dataset, desired_len=min_datapoints)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True, num_workers=config.num_workers)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model_comparative_SSL(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss_SSL, scheduler, stage="finetune")
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True, num_workers=config.num_workers)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model_comparative(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss, scheduler, stage="finetune")

    print("DONE FINE TUNING")

    model.load_trained(config.model_save_path_finetune, new_head=False)

    train_stats = {}
    test_states = {}

    if config.self_supervision:
        train_stats = inference_comparative_SSL(config, model, train_loader, device)
        test_stats = inference_comparative_SSL(config, model, test_loader, device,
                                               clinical_loader=clinical_loader,
                                               fig_save_folder=config.fig_save_folder,
                                               optimal_threshold=train_stats["optimal_threshold"])
    else:
        train_stats = inference_comparative(config, model, train_loader, device)
        test_stats = inference_comparative(config, model, test_loader, device,
                                           clinical_loader=clinical_loader,
                                           fig_save_folder=config.fig_save_folder,
                                           optimal_threshold=train_stats["optimal_threshold"])

    wandb.log({
        "Train ROC AUC": train_stats["roc_auc"],
        "Train PR AUC": train_stats["pr_auc"],
        "Train Accuracy @0.5": train_stats["accuracy"],
        "Train Accuracy @op": train_stats["accuracy_op"],
        "Train F1 Score @0.5": train_stats["f1"],
        "Train F1 Score @op": train_stats["f1_op"],
        "Train Precision @0.5": train_stats["precision"],
        "Train Precision @op": train_stats["precision_op"],
        "Train Recall @0.5": train_stats["recall"],
        "Train Recall @op": train_stats["recall_op"],
        "Train Mean PPVn @0.5": train_stats["ppvn"],
        "Train Mean PPVn @op": train_stats["ppvn_op"],
        "Train PPVn (n=30) @0.5": train_stats["ppv30"],
        "Train PPVn (n=30) @op": train_stats["ppv30_op"],
    })

    wandb.log({
        "Test ROC AUC": test_stats["roc_auc"],
        "Test PR AUC": test_stats["pr_auc"],
        "Test Accuracy @0.5": test_stats["accuracy"],
        "Test Accuracy @op": test_stats["accuracy_op"],
        "Test F1 Score @0.5": test_stats["f1"],
        "Test F1 Score @op": test_stats["f1_op"],
        "Test Precision @0.5": test_stats["precision"],
        "Test Precision @op": test_stats["precision_op"],
        "Test Recall @0.5": test_stats["recall"],
        "Test Recall @op": test_stats["recall_op"],
        "Test Mean PPVn @0.5": test_stats["ppvn"],
        "Test Mean PPVn @op": test_stats["ppvn_op"],
        "Test PPVn (n=30) @0.5": test_stats["ppv30"],
        "Test PPVn (n=30) @op": test_stats["ppv30_op"],
        "OS p-value": test_stats["os_p_value"],
        "PFS p-value": test_stats["pfs_p_value"],
    })

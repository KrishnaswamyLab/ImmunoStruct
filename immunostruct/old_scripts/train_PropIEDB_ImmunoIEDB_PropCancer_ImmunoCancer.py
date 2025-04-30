import os
import argparse
import torch
import wandb
from dgl.dataloading import GraphDataLoader

from data import ImmunoPredDataset, collate, SplitDataset, collate_amino_acid
from models.mapping import model_map
from utils import Losses, seed_everything, update_paths
from procedures import inference, train_model, inference_SSL, train_model_SSL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry point.")
    parser.add_argument("--model", default="StructureModel", type=str)
    parser.add_argument("--learning-rate-pretrain", default=1e-3, type=float)
    parser.add_argument("--learning-rate-finetune", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=150, type=int)
    parser.add_argument("--full-sequence", action="store_true")
    parser.add_argument("--sequence-loss", action="store_true")
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--model-save-dir", default="$ROOT/results/PropIEDB_ImmunoIEDB_PropCancer_ImmunoCancer/", type=str)
    parser.add_argument("--graph-dir-IEDB", default="$ROOT/data/graph_pyg_IEDB/", type=str)
    parser.add_argument("--graph-dir-cancer", default="$ROOT/data/graph_pyg_Cancer/", type=str)
    parser.add_argument("--property-path-IEDB", default="$ROOT/data/complete_score_Mprops_1_2_smoothed_sasa_v2.txt", type=str)
    parser.add_argument("--property-path-cancer", default="$ROOT/data/cedar_data_final_with_mprop1_mprop2_v2.txt", type=str)
    parser.add_argument("--hla-path", default="$ROOT/data/HLA_27_seqs_csv.csv", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--wandb-username", default=None, type=str)
    parser.add_argument("--sequence-pad-count", default=0, type=int)
    parser.add_argument("--structure-pad-count", default=0, type=int)
    parser.add_argument("--self-supervision", action="store_true") # if the model is an SSL model this should be set to true
    config = parser.parse_args()

    update_paths(config)

    # Model save paths.
    model_str = f"{config.model}-lr_pt_{config.learning_rate_pretrain}-lr_ft_{config.learning_rate_finetune}" + \
        f"-ep_{config.num_epochs}-bs_{config.batch_size}-fseq_{config.full_sequence}-seql_{config.sequence_loss}" + \
        f"-fs_{config.feature_size}-cs_{config.coord_size}-seed_{config.seed}"
    config.model_save_path_pretrain = os.path.join(config.model_save_dir, model_str + "_pretrain.pt")
    config.model_save_path_finetune = os.path.join(config.model_save_dir, model_str + "_finetune.pt")

    # Weights and biases.
    wandb.init(
        project="ImmunoPred-Scratch", # set the wandb project where this run will be logged
        entity=config.wandb_username,
        name=f"PropIEDB_ImmunoIEDB_PropCancer_ImmunoCancer:{model_str}",  # display name of the run
        config=config,   # track hyperparameters and run metadata
    )
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    # Define model.
    # input_dim = sequence length (11 for peptide, 283 for sequence) * embedding
    input_dim = 283 * 21 if config.full_sequence else 11 * 21
    model = model_map[config.model](vae_input_dim=input_dim, device=device)
    model.to(device)

    # Pretraining and Finetuning datasets.
    dataset_pt = dataset_ft1 = ImmunoPredDataset(config,
                                                 graph_directory=config.graph_dir_IEDB,
                                                 property_path=config.property_path_IEDB,
                                                 hla_path=config.hla_path)
    dataset_ft2 = dataset_ft3 = ImmunoPredDataset(config,
                                                  graph_directory=config.graph_dir_cancer,
                                                  property_path=config.property_path_cancer,
                                                  hla_path=config.hla_path)

    train_dataset_pt, val_dataset_pt, test_dataset_pt = torch.utils.data.random_split(dataset_pt, [0.7, 0.15, 0.15], generator)
    train_dataset_ft1, val_dataset_ft1, test_dataset_ft1 = train_dataset_pt, val_dataset_pt, test_dataset_pt
    train_dataset_ft2, val_dataset_ft2, test_dataset_ft2 = torch.utils.data.random_split(dataset_ft2, [0.7, 0.15, 0.15], generator)
    train_dataset_ft3, val_dataset_ft3, test_dataset_ft3 = train_dataset_ft2, val_dataset_ft2, test_dataset_ft2
    print("Pretraining train/val/test size:", len(train_dataset_pt), len(val_dataset_pt), len(test_dataset_pt))
    print("Finetuning stage 1 train/val/test size:", len(train_dataset_ft1), len(val_dataset_ft1), len(test_dataset_ft1))
    print("Finetuning stage 2 train/val/test size:", len(train_dataset_ft2), len(val_dataset_ft2), len(test_dataset_ft2))
    print("Finetuning stage 3 train/val/test size:", len(train_dataset_ft3), len(val_dataset_ft3), len(test_dataset_ft3))

    # `binary=False` --> Using property.
    train_split_dataset = SplitDataset(train_dataset_pt, "train", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_pt, "val", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_pt, "test", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_pretrain)
    losses = Losses(input_dim, dataset_pt.class_weights, sequence=config.sequence_loss)

    if config.self_supervision:
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss_SSL)
    else:
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss)

    print("DONE PRE-TRAINING")
    del train_split_dataset, val_split_dataset, test_split_dataset
    del train_loader, val_loader, test_loader
    del optimizer

    model.load_trained(config.model_save_path_pretrain, new_head=True)

    # `binary=True` --> Using immunogenicity.
    train_split_dataset = SplitDataset(train_dataset_ft1, "train", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_ft1, "val", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_ft1, "test", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_finetune, weight_decay=1e-6)
    losses = Losses(input_dim, dataset_ft1.class_weights, sequence=config.sequence_loss)

    if config.self_supervision:
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss_SSL, stage="finetune")
    else:
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss, stage="finetune")

    print("DONE FINE TUNING Stage 1")
    del train_split_dataset, val_split_dataset, test_split_dataset
    del train_loader, val_loader, test_loader
    del optimizer

    model.load_trained(config.model_save_path_finetune, new_head=True)

    # `binary=False` --> Using property.
    train_split_dataset = SplitDataset(train_dataset_ft2, "train", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_ft2, "val", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_ft2, "test", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_finetune, weight_decay=1e-6)
    losses = Losses(input_dim, dataset_ft2.class_weights, sequence=config.sequence_loss)

    if config.self_supervision:
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss_SSL, stage="finetune")
    else:
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss, stage="finetune")

    print("DONE FINE TUNING Stage 2")
    del train_split_dataset, val_split_dataset, test_split_dataset
    del train_loader, val_loader, test_loader
    del optimizer

    model.load_trained(config.model_save_path_finetune, new_head=True)

    # `binary=True` --> Using immunogenicity.
    train_split_dataset = SplitDataset(train_dataset_ft3, "train", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset = SplitDataset(val_dataset_ft3, "val", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    test_split_dataset = SplitDataset(test_dataset_ft3, "test", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True)
        val_loader = GraphDataLoader(val_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)
        test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_finetune, weight_decay=1e-6)
    losses = Losses(input_dim, dataset_ft3.class_weights, sequence=config.sequence_loss)

    if config.self_supervision:
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss_SSL, stage="finetune")
    else:
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss, stage="finetune")

    print("DONE FINE TUNING Stage 3")

    model.load_trained(config.model_save_path_finetune, new_head=False)

    if config.self_supervision:
        accuracy, precision, recall, f1, roc_auc, pr_auc, ppvn, ppv30 = inference_SSL(config, model, train_loader, device)
    else:
        accuracy, precision, recall, f1, roc_auc, pr_auc, ppvn, ppv30 = inference(config, model, train_loader, device)
    wandb.log({
        "Train Accuracy": accuracy,
        "Train Precision": precision,
        "Train Recall": recall,
        "Train F1 Score": f1,
        "Train ROC AUC": roc_auc,
        "Train PR AUC": pr_auc,
        "Train Mean PPVn": ppvn,
        "Train PPVn (n=30)": ppv30,
    })

    if config.self_supervision:
        accuracy, precision, recall, f1, roc_auc, pr_auc, ppvn, ppv30 = inference_SSL(config, model, test_loader, device)
    else:
        accuracy, precision, recall, f1, roc_auc, pr_auc, ppvn, ppv30 = inference(config, model, test_loader, device)
    wandb.log({
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1 Score": f1,
        "Test ROC AUC": roc_auc,
        "Test PR AUC": pr_auc,
        "Test Mean PPVn": ppvn,
        "Test PPVn (n=30)": ppv30,
    })

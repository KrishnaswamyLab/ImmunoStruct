import os
import torch
import wandb
from utils import PairedContrastiveLoss

__all__ = ["train_model_SSL", "train_model_comparative_SSL"]

# Training loop.
# stage can be either "pretrain" or "finetune"
def train_model_SSL(config, device, model, train_loader, val_loader, optimizer, loss_function, scheduler=None, stage="pretrain"):
    train_losses = []
    val_losses = []

    lowest_val_loss = float("inf")
    for epoch_idx in range(config.num_epochs):
        model.train()
        train_loss = 0
        for graph_data, sequence_data, target, peptide_property, amino_acid in train_loader:

            graph_data = graph_data.to(device)
            sequence_data, target, peptide_property = sequence_data.to(device), target.to(device), peptide_property.to(device)
            amino_acid = amino_acid.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar, final_output, pred_amino_acid = model(graph_data, sequence_data, peptide_property)

            loss = loss_function(recon_batch, sequence_data, mu, logvar, final_output, target, pred_amino_acid, amino_acid)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        if scheduler is not None:
            scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graph_data, sequence_data, target, peptide_property, _ in val_loader:
                graph_data = graph_data.to(device)
                sequence_data, target, peptide_property = sequence_data.to(device), target.to(device), peptide_property.to(device)

                recon_batch, mu, logvar, final_output, _ = model(graph_data, sequence_data, peptide_property)
                loss = loss_function(recon_batch, sequence_data, mu, logvar, final_output, target, torch.tensor([]), torch.tensor([]))
                val_loss += loss.item()

        if val_loss < lowest_val_loss:
            if stage == "pretrain":
                os.makedirs(os.path.dirname(config.model_save_path_pretrain), exist_ok=True)
                torch.save(model.state_dict(), config.model_save_path_pretrain)
            else:
                os.makedirs(os.path.dirname(config.model_save_path_finetune), exist_ok=True)
                torch.save(model.state_dict(), config.model_save_path_finetune)
            lowest_val_loss = val_loss

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        wandb.log({
            stage + "_train_loss": train_loss,
            stage + "_val_loss": val_loss
        })

        print(f"Epoch {epoch_idx+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Training loop.
def train_model_comparative_SSL(config, device, model, train_loader, val_loader, optimizer, loss_function, scheduler=None, stage="pretrain"):
    train_losses = []
    val_losses = []

    if 'coeff_contrastive' in vars(config).keys() and config.coeff_contrastive > 0:
        coeff_contrastive = config.coeff_contrastive
        paired_contrastive_loss = PairedContrastiveLoss(device=device, embedding_dim=104)
    else:
        coeff_contrastive = 0

    lowest_val_loss = float("inf")
    for epoch_idx in range(config.num_epochs):
        model.train()
        train_loss = 0
        for graph_data_pair, sequence_data_pair, target, peptide_property_pair, amino_acid in train_loader:

            graph_data_cancer, graph_data_wt = graph_data_pair
            sequence_data_cancer, sequence_data_wt = sequence_data_pair
            peptide_property_cancer, peptide_property_wt = peptide_property_pair

            graph_data_cancer, graph_data_wt = graph_data_cancer.to(device), graph_data_wt.to(device)
            sequence_data_cancer, sequence_data_wt = sequence_data_cancer.to(device), sequence_data_wt.to(device)
            peptide_property_cancer, peptide_property_wt = peptide_property_cancer.to(device), peptide_property_wt.to(device)
            target = target.to(device)
            amino_acid = amino_acid.to(device)

            embedding_pair, recon_batch_pair, mu_pair, logvar_pair, final_output, pred_amino_acid = \
                model.forward_comparative((graph_data_cancer, graph_data_wt),
                                          (sequence_data_cancer, sequence_data_wt),
                                          (peptide_property_cancer, peptide_property_wt))

            embedding_cancer, embedding_wt = embedding_pair
            recon_batch_cancer, recon_batch_wt = recon_batch_pair
            mu_cancer, mu_wt = mu_pair
            logvar_cancer, logvar_wt = logvar_pair

            loss_cancer = loss_function(recon_batch_cancer, sequence_data_cancer, mu_cancer, logvar_cancer, final_output, target, pred_amino_acid, amino_acid)
            loss_wt = loss_function(recon_batch_wt, sequence_data_wt, mu_wt, logvar_wt, final_output, target, pred_amino_acid, amino_acid)
            loss = (loss_cancer + loss_wt) / 2

            if coeff_contrastive > 0:
                loss_contrastive = paired_contrastive_loss(embedding_cancer, embedding_wt, target)
                loss = loss + coeff_contrastive * loss_contrastive

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        if scheduler is not None:
            scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():

            for graph_data_pair, sequence_data_pair, target, peptide_property_pair, _ in val_loader:

                graph_data_cancer, graph_data_wt = graph_data_pair
                sequence_data_cancer, sequence_data_wt = sequence_data_pair
                peptide_property_cancer, peptide_property_wt = peptide_property_pair

                graph_data_cancer, graph_data_wt = graph_data_cancer.to(device), graph_data_wt.to(device)
                sequence_data_cancer, sequence_data_wt = sequence_data_cancer.to(device), sequence_data_wt.to(device)
                peptide_property_cancer, peptide_property_wt = peptide_property_cancer.to(device), peptide_property_wt.to(device)
                target = target.to(device)

                embedding_pair, recon_batch_pair, mu_pair, logvar_pair, final_output, _ = \
                    model.forward_comparative((graph_data_cancer, graph_data_wt),
                                              (sequence_data_cancer, sequence_data_wt),
                                              (peptide_property_cancer, peptide_property_wt))

                embedding_cancer, embedding_wt = embedding_pair
                recon_batch_cancer, recon_batch_wt = recon_batch_pair
                mu_cancer, mu_wt = mu_pair
                logvar_cancer, logvar_wt = logvar_pair

                loss_cancer = loss_function(recon_batch_cancer, sequence_data_cancer, mu_cancer, logvar_cancer, final_output, target, torch.tensor([]), torch.tensor([]))
                loss_wt = loss_function(recon_batch_wt, sequence_data_wt, mu_wt, logvar_wt, final_output, target, torch.tensor([]), torch.tensor([]))
                loss = (loss_cancer + loss_wt) / 2

                if coeff_contrastive > 0:
                    loss_contrastive = paired_contrastive_loss(embedding_cancer, embedding_wt, target)
                    loss = loss + coeff_contrastive * loss_contrastive

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < lowest_val_loss:
            if stage == "pretrain":
                os.makedirs(os.path.dirname(config.model_save_path_pretrain), exist_ok=True)
                torch.save(model.state_dict(), config.model_save_path_pretrain)
            else:
                os.makedirs(os.path.dirname(config.model_save_path_finetune), exist_ok=True)
                torch.save(model.state_dict(), config.model_save_path_finetune)
            lowest_val_loss = val_loss

        wandb.log({
            stage + "_train_loss": train_loss,
            stage + "_val_loss": val_loss
        })

        print(f"Epoch {epoch_idx+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

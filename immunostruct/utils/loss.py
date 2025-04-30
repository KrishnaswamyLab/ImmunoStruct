import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Losses:
    def __init__(self, vae_input_dim, class_weights, sequence = True):
        self.vae_input_dim = vae_input_dim
        self.sequence = sequence

        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        self.pos_weight = torch.tensor(float(class_weights[0])/ float(class_weights[1])).float()

    def regression_loss(self, recon_x, x, mu, logvar, final_output, y):
        regression = F.mse_loss(final_output.squeeze(), y.squeeze(), reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return 2.0 * regression + 0.5 * MSE + 0.5 * KLD
        else:
            return regression

    def BCE_loss(self, recon_x, x, mu, logvar, final_output, y):
        BCE = F.binary_cross_entropy_with_logits(final_output.view(-1), y.view(-1), pos_weight=self.pos_weight, reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return 5.0 * BCE + 0.1 * MSE + 0.1 * KLD
        else:
            return BCE
        
    def regression_loss_SSL(self, recon_x, x, mu, logvar, final_output, y, pred_amino_acid, amino_acid):
        amino_classification = 0

        if pred_amino_acid.numel(): 
            amino_classification = F.cross_entropy(pred_amino_acid, amino_acid)

        regression = F.mse_loss(final_output.squeeze(), y.squeeze(), reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return 2.0 * regression + 0.5 * MSE + 0.5 * KLD + amino_classification
        else:
            return regression + amino_classification

    def BCE_loss_SSL(self, recon_x, x, mu, logvar, final_output, y, pred_amino_acid, amino_acid):
        amino_classification = 0

        if pred_amino_acid.numel(): 
            amino_classification = F.cross_entropy(pred_amino_acid, amino_acid)

        BCE = F.binary_cross_entropy_with_logits(final_output.view(-1), y.view(-1), pos_weight=self.pos_weight, reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return 5.0 * BCE + 0.1 * MSE + 0.1 * KLD + amino_classification
        else:
            return BCE + amino_classification

# Plot the loss curves
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

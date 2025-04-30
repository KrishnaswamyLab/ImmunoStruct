import torch

__all__ = ['PairedContrastiveLoss']


class PairedContrastiveLoss(torch.nn.Module):
    '''
    This is inspired by the InfoNCE loss (Equation 1 in the SimCLR paper https://arxiv.org/abs/2002.05709),
    and the barlow twins loss (Algorithm 1 in the Barlow Twins paper https://arxiv.org/pdf/2103.03230).

    We are interested in:
        Pulling together the cancer/WT embedding pair for non-immunogenic proteins.
        Pushing apart the cancer/WT embedding pair for immunogenic proteins.
        Encouraging all non-pairs to be orthogonal to each other.
        Reducing redundancy among feature dimensions.
    '''

    def __init__(self,
                 embedding_dim: int = 104,
                 z_dim: int = 128,
                 lambda_off_diag: float = 1e-2,
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        self.z_dim = z_dim
        self.lambda_off_diag = lambda_off_diag
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, z_dim, bias=False),
            torch.nn.BatchNorm1d(z_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(z_dim, z_dim, bias=False)
        )
        self.device = device

        self.projector.to(self.device)

    def forward(self, embedding_cancer: torch.Tensor, embedding_wt: torch.Tensor, is_immunogenic: torch.Tensor):
        if len(is_immunogenic.unique()) == 2:
            is_immunogenic = is_immunogenic > is_immunogenic.mean()
        else:
            # Either in the pretraining stage with continuous foreignness score,
            # or during finetuning stage with immunogenicity but no diversity in batch.
            return 0  # Nothing to contrast.

        assert embedding_cancer.shape == embedding_wt.shape

        z_cancer = self.projector(embedding_cancer)
        z_wt = self.projector(embedding_wt)

        batch_size = z_cancer.shape[0]
        assert self.z_dim == z_cancer.shape[1]

        # Mean centering.
        z_cancer = z_cancer - z_cancer.mean(0)                              # batch_size x z_dim
        z_wt = z_wt - z_wt.mean(0)                                          # batch_size x z_dim

        # Unit variance regularization.
        std_z_cancer = torch.sqrt(z_cancer.var(dim=0) + 0.0001)
        std_z_wt = torch.sqrt(z_wt.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_z_cancer)) / 2 \
                 + torch.mean(torch.nn.functional.relu(1 - std_z_wt)) / 2

        # Pair similarity matrix.
        pair_sim = torch.mm(z_cancer, z_wt.T) / self.z_dim                  # batch_size x batch_size

        # Cross-correlation matrix.
        cross_corr = torch.mm(z_cancer.T, z_wt) / batch_size                # z_dim x z_dim

        # Push and pull based on immunogenicity.
        pair_sim_ideal = torch.eye(batch_size, device=self.device)          # batch_size x batch_size
        pair_sim_ideal[~is_immunogenic] = 0
        pair_sim_diff = (pair_sim - pair_sim_ideal).pow(2)                  # batch_size x batch_size
        # Down-weigh the off-diagonal items.
        pair_sim_diff[~torch.eye(batch_size, dtype=bool)] *= self.lambda_off_diag

        # Encourage pair correlation and reduce feature redundancy.
        cross_corr_ideal = torch.eye(self.z_dim, device=self.device)        # z_dim x z_dim
        cross_corr_diff = (cross_corr - cross_corr_ideal).pow(2)            # z_dim x z_dim
        # Down-weigh the off-diagonal items.
        cross_corr_diff[~torch.eye(self.z_dim, dtype=bool)] *= self.lambda_off_diag

        loss = pair_sim_diff.sum() + cross_corr_diff.sum() + std_loss

        return loss
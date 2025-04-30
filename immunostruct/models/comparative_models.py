import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from dgl.nn import EGNNConv
from .layers import SelfAttention, MultiHeadAttention

__all__ = ["HybridModel_Comparative", "HybridModelv2_Comparative", "HybridModelv2_Comparative_SSL", "HybridModel_Comparative_SSL"]


class HybridModel_Comparative(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim : int = 8,
                 use_wt_for_downstream: bool = True):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.use_wt_for_downstream = use_wt_for_downstream

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = SelfAttention(gat_hidden_channels)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        if self.use_wt_for_downstream:
            input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2
        else:
            input_dim = self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        return self.vae_fc21(h1), self.vae_fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward_item(self, graph_data, sequence_data, peptide_property):
        x, node_feat, coord_feat, edge_feat = \
            graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(graph_data.device)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        return mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        # Fusion and final layers
        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)
        if self.use_wt_for_downstream:
            combined = torch.cat([embeddings_cancer, embeddings_wt], dim=1)
        else:
            combined = embeddings_cancer

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """
        This is mainly for pre-training purposes.
        Using this as `forward` for backward compatibility.
        """

        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        # Fusion and final layers
        if self.use_wt_for_downstream:
            # NOTE: repeating features as a hot fix for classifier dimension.
            combined = torch.cat([x_gat_node, z_vae, x_gat_node, z_vae], dim=1)
        else:
            combined = torch.cat([x_gat_node, z_vae], dim=1)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output

class HybridModel_Comparative_SSL(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim : int = 8,
                 use_wt_for_downstream: bool = True,
                 mlp_features: int = 32):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.use_wt_for_downstream = use_wt_for_downstream
        self.mlp_features = mlp_features

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = SelfAttention(gat_hidden_channels)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20) # number of amino acids

    def get_classifier(self):
        if self.use_wt_for_downstream:
            input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2
        else:
            input_dim = self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        return self.vae_fc21(h1), self.vae_fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward_item(self, graph_data, sequence_data, peptide_property):
        x, node_feat, coord_feat, edge_feat = \
            graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(graph_data.device)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        return mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        # Fusion and final layers
        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)
        if self.use_wt_for_downstream:
            combined = torch.cat([embeddings_cancer, embeddings_wt], dim=1)
        else:
            combined = embeddings_cancer

        fusion_output = self.classifier(combined)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)


        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output, node_prediction

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output, node_prediction

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output, node_prediction

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """
        This is mainly for pre-training purposes.
        Using this as `forward` for backward compatibility.
        """

        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        # Fusion and final layers
        if self.use_wt_for_downstream:
            # NOTE: repeating features as a hot fix for classifier dimension.
            combined = torch.cat([x_gat_node, z_vae, x_gat_node, z_vae], dim=1)
        else:
            combined = torch.cat([x_gat_node, z_vae], dim=1)

        fusion_output = self.classifier(combined)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output, node_prediction


        if return_attention:
            return attention_weights_n, mu, logvar, final_output, node_prediction


        return recon_x, mu, logvar, final_output, node_prediction


class HybridModelv2_Comparative(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim : int = 8,
                 self_attention_heads: int = 1,
                 combined_attention_heads: int = 8,
                 use_wt_for_downstream: bool = True):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.use_wt_for_downstream = use_wt_for_downstream

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, self_attention_heads)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        self.combined_attention = MultiHeadAttention(32, combined_attention_heads, input_dim=1)

        # Fusion/Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        if self.use_wt_for_downstream:
            input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2
        else:
            input_dim = self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        return self.vae_fc21(h1), self.vae_fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward_item(self, graph_data, sequence_data, peptide_property):
        x, node_feat, coord_feat, edge_feat = \
            graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(graph_data.device)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        return mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        # Fusion and final layers
        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)
        if self.use_wt_for_downstream:
            combined = torch.cat([embeddings_cancer, embeddings_wt], dim=1)
        else:
            combined = embeddings_cancer

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """
        This is mainly for pre-training purposes.
        Using this as `forward` for backward compatibility.
        """

        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        # Fusion and final layers
        if self.use_wt_for_downstream:
            # NOTE: repeating features as a hot fix for classifier dimension.
            combined = torch.cat([x_gat_node, z_vae, x_gat_node, z_vae], dim=1)
        else:
            combined = torch.cat([x_gat_node, z_vae], dim=1)

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output

class HybridModelv2_Comparative_SSL(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim: int = 8,
                 self_attention_heads: int = 1,
                 combined_attention_heads: int = 8,
                 use_wt_for_downstream: bool = True,
                 mlp_features: int = 32):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.property_embedding_dim = property_embedding_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.use_wt_for_downstream = use_wt_for_downstream
        self.mlp_features = mlp_features

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, self_attention_heads)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        self.combined_attention = MultiHeadAttention(32, combined_attention_heads, input_dim=1)

        # Fusion/Classifier layers
        self.classifier = self.get_classifier()

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20) # number of amino acids

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        if self.use_wt_for_downstream:
            input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2
        else:
            input_dim = self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        return self.vae_fc21(h1), self.vae_fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward_item(self, graph_data, sequence_data, peptide_property):
        x, node_feat, coord_feat, edge_feat = \
            graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(graph_data.device)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        return mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        # Fusion and final layers
        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)
        if self.use_wt_for_downstream:
            combined = torch.cat([embeddings_cancer, embeddings_wt], dim=1)
        else:
            combined = embeddings_cancer

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        fusion_output = self.classifier(combined)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output, node_prediction

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output, node_prediction

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output, node_prediction

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """
        This is mainly for pre-training purposes.
        Using this as `forward` for backward compatibility.
        """

        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        # Fusion and final layers
        if self.use_wt_for_downstream:
            # NOTE: repeating features as a hot fix for classifier dimension.
            combined = torch.cat([x_gat_node, z_vae, x_gat_node, z_vae], dim=1)
        else:
            combined = torch.cat([x_gat_node, z_vae], dim=1)

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        fusion_output = self.classifier(combined)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output, node_prediction

        if return_attention:
            return attention_weights_n, mu, logvar, final_output, node_prediction

        return recon_x, mu, logvar, final_output, node_prediction

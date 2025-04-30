import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from dgl.nn import EGNNConv
from .layers import SelfAttention, MultiHeadAttention

__all__ = ["SequenceModel", "SequenceFpModel", "StructureModel", "StructureModel_SSL", "StructureModelv2", "DualModel"]

class SequenceModel(nn.Module):
    def __init__(self, vae_input_dim, device, gcn_layers=5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden_channels = 64, *args, **kwargs):
        super(SequenceModel, self).__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim, vae_hidden_dim) # vae_latent_dim + 1 (number of properties) is the number of features
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim, 32), # vae_latent_dim + 1 + gat_hidden_channels
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

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        recon_x = self.decode_vae(z_vae)

        # Fusion and final layers
        final_output = self.classifier(z_vae)

        return recon_x, mu, logvar, final_output

class SequenceFpModel(nn.Module):
    def __init__(self, vae_input_dim, device, gcn_layers=5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden_channels = 64, *args, **kwargs):
        super(SequenceFpModel, self).__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + 2, vae_hidden_dim) # vae_latent_dim + 1 (number of properties) is the number of features
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + 2, 32), # vae_latent_dim + 1 + gat_hidden_channels
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

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        # Fusion and final layers
        final_output = self.classifier(z_vae)

        return recon_x, mu, logvar, final_output

class StructureModel(nn.Module):
    def __init__(self, vae_input_dim, device, gcn_layers=5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden_channels = 64, *args, **kwargs):
        super(StructureModel, self).__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, 8)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.gat_hidden_channels, 32), # vae_latent_dim + 1 + gat_hidden_channels
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

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

        # Fusion and final layers
        final_output = self.classifier(x_gat_node)
        return 0, 0, 0, final_output

class StructureModel_SSL(nn.Module):
    def __init__(self, vae_input_dim, device, gcn_layers = 5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden_channels = 64, mlp_features = 32, *args, **kwargs):
        super(StructureModel_SSL, self).__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.mlp_features = mlp_features

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, 8)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20) # number of amino acids

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.gat_hidden_channels, 32),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

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

        # Fusion and final layers
        fusion_output = self.classifier(x_gat_node)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        return 0, 0, 0, final_output, node_prediction

class StructureModelv2(nn.Module):
    def __init__(self, vae_input_dim, device, gcn_layers = 5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden_channels = 64, mlp_features = 32, *args, **kwargs):
        super(StructureModelv2, self).__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.mlp_features = mlp_features

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, 8)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20) # number of amino acids

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.gat_hidden_channels*2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

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
        x_gat_node_mean = global_mean_pool(attention_output_n, batch_tensor.flatten())
        x_gat_node_max = global_max_pool(attention_output_n, batch_tensor.flatten())

        x_gat_node = torch.cat([x_gat_node_mean, x_gat_node_max], dim=-1)

        # Fusion and final layers
        fusion_output = self.classifier(x_gat_node)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        return 0, 0, 0, final_output, node_prediction

class DualModel(nn.Module):
    def __init__(self, vae_input_dim, device, gcn_layers=5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden_channels = 64):
        super(DualModel, self).__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = SelfAttention(gat_hidden_channels)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim, vae_hidden_dim) # vae_latent_dim + 1 (number of properties) is the number of features
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + self.gat_hidden_channels, 32), # vae_latent_dim + gat_hidden_channels
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

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

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

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        recon_x = self.decode_vae(z_vae)

        # Fusion and final layers
        combined = torch.cat([x_gat_node, z_vae], dim=1)
        combined_gat_only = torch.cat([x_gat_node], dim=1)
        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output

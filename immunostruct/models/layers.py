import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights
    
# https://github.com/hyunwoongko/transformer/blob/master/models/layers/scale_dot_product_attention.py
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = F.softmax(score, dim=-1)

        # 4. multiply with Value
        v = score @ v

        return v, score

# https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py
class MultiHeadAttention(nn.Module):

    def __init__(self, feature_dim, n_head, input_dim=None):
        super(MultiHeadAttention, self).__init__()

        assert feature_dim % n_head == 0, "Embedding dimension must be 0 modulo number of heads."

        if not input_dim:
            input_dim = feature_dim
        self.n_head = n_head
        self.w_q = nn.Linear(input_dim, feature_dim)
        self.w_k = nn.Linear(input_dim, feature_dim)
        self.w_v = nn.Linear(input_dim, feature_dim)
        self.w_concat = nn.Linear(feature_dim, feature_dim)
        self.attention = ScaleDotProductAttention()

    def forward(self, x, mask=None):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        q, k, v = self.split(q), self.split(k), self.split(v)

        # attention here is multi head attention
        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))

class MixedScoreAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']
        mix1_weight = torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(sample_shape=(head_num, 2, ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(sample_shape=(head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        self.mix1_bias = nn.Parameter(mix1_bias)

        mix2_weight = torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(sample_shape=(head_num, ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(sample_shape=(head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        self.mix2_bias = nn.Parameter(mix2_bias)

    def forward(self, q, k, v, problem):

        batch_size = q.size(0)
        row_size = q.size(2)
        col_size = k.size(2)
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']
        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head, row, col)
        dot_product_score = dot_product.unsqueeze(-1) / sqrt_qkv_dim
        problem_score = (problem[:, None, :, :].expand(batch_size, head_num, row_size, col_size)).unsqueeze(-1)
        cat_score = torch.cat((dot_product_score, problem_score), dim=4)
        transposed_score = cat_score.transpose(1, 2)
        mix_score1 = torch.matmul(transposed_score, self.mix1_weight)
        # shape: (batch, row, head, col, hidden)
        mix_score1 = mix_score1 + self.mix1_bias[None, None, :, None, :]
        mix_score1_activated = F.relu(mix_score1)

        mix_score2 = torch.matmul(mix_score1_activated, self.mix2_weight)
        # shape: (batch, row, head, col, 1)
        mix_score2 = mix_score2 + self.mix2_bias[None, None, :, None, :]

        mixed_score = mix_score2.transpose(1, 2)
        mixed_score = mixed_score.squeeze(4)
        # shape: (batch, head, row, col)
        weights = nn.Softmax(dim=3)(mixed_score)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, row_size, head_num * qkv_dim)
        return out


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)
    return picked_nodes

def reshape_by_heads(qkv, head_num):
    # q shape: (batch, n, head * qkv)
    batch_size = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_size, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head, n, qkv)
    return q_transposed

import torch
import torch.nn as nn
import torch.nn.functional as F

from ACVRP_Model_LIB import AddAndInstanceNormalization, FeedForward, MixedScoreAttention, _get_encoding, reshape_by_heads

class ACVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = ACVRP_Encoder(**self.model_params)
        self.decoder = ACVRP_Decoder(**self.model_params)
        self.encoded_row = None
        self.encoded_col = None
        # shape: (batch, node, embedding)

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        # shape: (batch, node, node)
        batch_size = problems.size(0)
        node_num = problems.size(1)
        embedding_dim = self.model_params['embedding_dim']
        row_emb = torch.zeros(size=(batch_size, node_num, embedding_dim))
        col_emb = torch.zeros(size=(batch_size, node_num, embedding_dim))
        seed_num = self.model_params['one_hot_seed_num']
        rand = torch.rand(batch_size, seed_num)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :node_num]

        batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, node_num)
        node_idx = torch.arange(node_num)[None, :].expand(batch_size, node_num)
        col_emb[batch_idx, node_idx, rand_idx] = 1

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        self.decoder.set_kv(self.encoded_col)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_row = _get_encoding(self.encoded_row, selected)
            self.decoder.set_q1(encoded_first_row)
        elif state.selected_count == 1:
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node)
            all_job_probs = self.decoder(encoded_current_row, mask=state.state_mask)
            if self.training:
                while True:
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)

                    prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = all_job_probs.argmax(dim=2)
                prob = None

        return selected, prob


class ACVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params["encoder_layer_num"]
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, problem):
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, problem)

        return row_emb, col_emb

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, problem):
        row_emb_out = self.row_encoding_block(row_emb, col_emb, problem)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, problem.transpose(1, 2))
        return row_emb_out, col_emb_out

class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_attention = MixedScoreAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.normalization1 = AddAndInstanceNormalization(**model_params)
        self.feedforward = FeedForward(**model_params)
        self.normalization2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row, col, problem):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(row), head_num=head_num)
        k = reshape_by_heads(self.Wk(col), head_num=head_num)
        v = reshape_by_heads(self.Wv(col), head_num=head_num)
        # shape: (batch, head, n, key)
        out_concat = self.mixed_score_attention(q, k, v, problem)
        mixed_score_attention = self.multi_head_combine(out_concat)
        out1 = self.normalization1(row, mixed_score_attention)
        out2 = self.feedforward(out1)
        out3 = self.normalization2(out1, out2)
        return out3

class ACVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.Wq0 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None

    def set_kv(self, encoded_col):
        head_num = self.model_params["head_num"]
        self.k = reshape_by_heads(self.Wk(encoded_col), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_col), head_num=head_num)
        # shape: (batch, head_num, node, )
        self.single_head_key = encoded_col.transpose(1, 2)
        # shape: (batch, embedding, node)

    def set_q1(self, first_row):
        head_num = self.model_params["head_num"]
        self.q1 = reshape_by_heads(self.Wq1(first_row), head_num=head_num)

    def _decoder_attention(self, q, k, v, mask=None):
        batch_size = q.size(0)
        n = q.size(2)
        node_num = k.size(2)
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        sqrt_qkv_dim = self.model_params["sqrt_qkv_dim"]
        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, pomo, node)
        score = score / sqrt_qkv_dim
        if mask is not None:
            score = score + mask[:, None, :, :].expand(batch_size, head_num, n, node_num)
        weights = nn.Softmax(dim=3)(score)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, n, head_num * qkv_dim)
        return out

    def forward(self, q0, mask):
        # q0 shape: (batch, pomo, embedding)
        # mask shape: (batch, pomo, node)
        head_num = self.model_params["head_num"]
        q0 = reshape_by_heads(self.Wq0(q0), head_num=head_num)
        # shape: (batch, head, pomo, qkv)
        q = self.q1 + q0
        out = self._decoder_attention(q, self.k, self.v, mask=mask)
        score = self.multi_head_combine(out)
        score = torch.matmul(score, self.single_head_key)
        sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]
        score = score / sqrt_embedding_dim
        score = logit_clipping * torch.tanh(score)
        score = score + mask
        probs = F.softmax(score, dim=2)
        return probs



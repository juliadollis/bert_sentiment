# model.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import CfgNode as CN

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):
    """
    Self-Attention SEM máscara causal
    (BERT enxerga tokens à esquerda e à direita).
    """
    def __init__(self, config):
        super().__init__()
        # Ex.: n_embd deve ser múltiplo de n_head
        assert config.n_embd % config.n_head == 0
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(self, x, attention_mask=None):
        """
        x: (B, T, C)
        attention_mask: (B, T) ou (B, 1, 1, T) se quiser ignorar tokens de padding
        BERT não usa máscara triangular (causal).
        """
        B, T, C = x.size()

        # Projeções em Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Separar cabeças: (B, T, n_head, C//n_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Matmul QK^T
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_head, T, T)

        # Caso queira ignorar tokens de padding, é aqui que se faz mask
        if attention_mask is not None:
            # Ex.: se attention_mask for (B, T), transmitimos para (B, 1, 1, T)
            # e colocamos -inf onde é 0 ou algo assim
            expanded_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            att = att.masked_fill(expanded_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class BERT(nn.Module):
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'bert'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.num_labels = 2  # Classificação binária
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # se quiser sobrescrever config de acordo com model_type, mas SEM assert de XOR
        if config.model_type is not None:
            config.merge_from_dict({
                'gpt-mini':   dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':  dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':   dict(n_layer=3, n_head=3, n_embd=48),
            }.get(config.model_type, {}))

        # Construção do "transformador"
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Cabeçalho de classificação
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.n_embd, self.num_labels)

        self.apply(self._init_weights)
        # Ajustar c_proj se quiser
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2.0**0.5 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, labels=None, attention_mask=None):
        """
        idx: (batch_size, seq_length) com IDs de token
        labels: (batch_size,) -> 0 ou 1
        attention_mask: se quiser mascarar tokens de padding (1 = token válido, 0 = token ignorado)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Seq len {t} > block_size {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x, attention_mask)

        x = self.transformer.ln_f(x)
        # Pega só o embedding do primeiro token para classificação, estilo BERT
        logits = self.classifier(x[:, 0, :])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

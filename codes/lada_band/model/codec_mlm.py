import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, BertConfig, BertModel, LlamaModel, AutoConfig

import omegaconf
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from ..utils.sampling import top_k_sampling, top_p_sampling, temperature_sampling
    from ..utils.mask import (
        scalar_to_batch_tensor,
        random_mask,
        apply_mask,
        _gamma,
        mask_by_mink,
    )
    # try:
    #     from .llama import LlamaModel
    # except:
    #     from .llama_codeclm_env import LlamaModel

    # from .conformer import Conformer (need gateloop)
except:
    import sys

    sys.path.append("../")
    from utils.sampling import top_k_sampling, top_p_sampling, temperature_sampling
    from utils.mask import (
        scalar_to_batch_tensor,
        random_mask,
        apply_mask,
        _gamma,
        mask_by_mink,
    )

    try:
        from llama import LlamaModel
    except:
        from llama_codeclm_env import LlamaModel
    # from conformer import Conformer


class SelectedHeads(nn.Module):
    def __init__(self, input_dim, n_codebooks, codebook_size):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.multihead = nn.Linear(input_dim, codebook_size * n_codebooks)

        self.codebook_offsets = (torch.arange(0, n_codebooks) * codebook_size).reshape(-1, 1, 1)

    def forward(self, embeds, q):
        '''
        embeds: (B, T, hidden_size)
        q: (B)
        '''
        B, T, _ = embeds.shape
        idxs = torch.arange(0, self.codebook_size * self.n_codebooks).repeat(B, T, 1)
        # print(idxs)
        start_idxs = self.codebook_offsets[q].repeat(1, T, 1)
        # print(start_idxs)
        end_idxs = start_idxs + self.codebook_size
        mask = (idxs >= start_idxs) & (idxs < end_idxs)
        mask = mask.to(embeds.device)
        # print(mask, mask.shape, mask.sum(-1))
        logits = self.multihead(embeds)
        logits = torch.masked_select(logits, mask)
        logits = logits.reshape(B, T, -1)
        return logits


class CodecMLMModel(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.model_name = cfg.model.name
        self.codec_name = cfg.codec.name
        model_cfg = cfg["model"]["json_path"]
        codec_cfg = cfg["codec"][self.codec_name]
        self.codec_cfg = codec_cfg
        codebook_size = codec_cfg.codebook_size
        n_codebooks = codec_cfg.n_codebooks

        if self.model_name == "llama":
            model_hg_cfg = AutoConfig.from_pretrained(model_cfg)
            self.lm = LlamaModel(model_hg_cfg)
            self.lm.embed_tokens.requires_grad_(False)  # freeze for ddp

        hidden_size = (
            model_cfg.hidden_size if model_cfg.get("hidden_size") else model_cfg.dim
        )
        self.voc_embed_layers = nn.ModuleList(
            [
                nn.Embedding(codebook_size + 1, hidden_size//2)
                for _ in range(n_codebooks)
            ]
        )
        self.acc_embed_layers = nn.ModuleList(
            [
                nn.Embedding(codebook_size + 2, hidden_size//2)
                for _ in range(n_codebooks)
            ]
        )

        self.special_tokens = {
            "voc_pad": codebook_size,
            "acc_mask": codebook_size,
            "acc_pad": codebook_size + 1,
        }

        ## codec每一层需要不同的linear层
        # self.to_logits = nn.ModuleList(
        #     [
        #         nn.Linear(hidden_size, codec_cfg.codebook_size, bias=False)
        #         for _ in range(codec_cfg.n_codebooks)
        #     ]
        # )
        self.to_logits = SelectedHeads(input_dim=hidden_size, n_codebooks=n_codebooks, codebook_size=codebook_size)


    def cal_parameter_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, voc_ids, acc_ids, q, attention_mask=None):  # TODO mulan embedding
        """
        voc_ids, acc_ids: [B, n_codebook, T]
        q: [B]
        """
        if attention_mask is None:
            attention_mask = (voc_ids[:, 0] != self.special_tokens["voc_pad"]).to(self.device)

        ## cal codec embeddings for each codebooks and cat
        voc_embeds = []
        for i, layer in enumerate(self.voc_embed_layers):
            voc_embeds.append(layer(voc_ids[:, i]))
        voc_embeds = torch.sum(torch.stack(voc_embeds), dim=0)

        acc_embeds = []
        for i, layer in enumerate(self.acc_embed_layers):
            acc_embeds.append(layer(acc_ids[:, i]))
        acc_embeds = torch.sum(torch.stack(acc_embeds), dim=0)

        input_embeds = torch.cat([voc_embeds, acc_embeds], dim=-1)
        out = self.lm(inputs_embeds=input_embeds, attention_mask=attention_mask).last_hidden_state
        logits = self.to_logits(out, q)

        return logits

    @torch.no_grad()
    def generate(  # TODO
        self,
        voc_ids,
        sampling_schedule=[16, 1, 1, 1],
        top_p=None,
        top_k=None,
        temperature=1.0,
        mask=None,
        mask_temperature=10.5,
    ):
        """
        batch_size: 1
        """
        B, n_q, seq_len = voc_ids.shape
        assert B == 1
        assert n_q == len(sampling_schedule)
        voc_ids = voc_ids.squeeze(0)

        ## mask all acc_ids in the beginning
        acc_ids_masked = torch.full_like(voc_ids, self.special_tokens["acc_mask"])
        for q in range(n_q):
            sampling_steps = sampling_schedule[q]
            for i in tqdm(range(sampling_steps)):
                logits = self.forward(
                    voc_ids=voc_ids[None],
                    acc_ids=acc_ids_masked[None],
                    q=[q]
                ).squeeze(0)
                if i < sampling_steps - 1:
                    if top_p is not None:
                        logits = top_p_sampling(logits, top_p=top_p)  # TODO top_p bug
                    if top_k is not None:
                        logits = top_k_sampling(logits, top_k=top_k)
                    probs = temperature_sampling(logits=logits, temp=temperature)
                    tokens = torch.multinomial(probs, 1).long()  # (seq_len, 1)
                    # print(probs.shape, tokens.shape)
                    selected_probs = torch.gather(probs, dim=-1, index=tokens).squeeze(-1)
                    tokens = tokens.squeeze(-1)
                    # print(selected_probs.shape)
                else:
                    ## greedy search on the last iteration
                    selected_probs, tokens = logits.max(-1)

                ## add back unmasked acc_ids
                mask = acc_ids_masked[q] == self.special_tokens["acc_mask"]
                tokens = torch.where(mask, tokens, acc_ids_masked[q])
                ## ignore masked idxs on probs
                selected_probs = torch.where(mask, selected_probs, torch.inf)

                ## mask according to the schedule
                r = torch.tensor([(i + 1) / sampling_steps])  # cos schedule
                num_to_mask = torch.floor(_gamma(r) * seq_len).long()
                if i != sampling_steps - 1:
                    num_to_mask = torch.max(
                        torch.tensor(1), torch.min((mask.sum() - 1).cpu(), num_to_mask)
                    )
                # print(num_to_mask)
                # print(selected_probs)
                mask = mask_by_mink(
                    num_to_mask,
                    selected_probs.cpu(),
                    temperature=mask_temperature * (1 - r),
                )  # TODO: 一开始随机性最大，之后变小？
                acc_ids_masked[q] = torch.where(
                    mask.to(tokens.device), self.special_tokens["acc_mask"], tokens
                )

        return acc_ids_masked


if __name__ == "__main__":
    # heads = SelectedHeads(input_dim=1024, n_codebooks=4, codebook_size=2048)
    # input_data = torch.randn(4, 1500, 1024)
    # logits = heads(input_data, [2,1,3,0])
    # print(logits.shape)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load("../conf/mlm_codec.yaml")
    # cfg = OmegaConf.load('../conf/mlm.yaml')
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg.model.llama.num_hidden_layers = 2
    model = CodecMLMModel(cfg).to(device)
    print("parameter num", model.cal_parameter_num())

    voc_ids = (torch.rand((4, 4, 1500)) * 2049).long().to(device)
    acc_ids = (torch.rand((4, 4, 1500)) * 2050).long().to(device)

    ## voc_ids[0, 0] = 1024
    ## acc_ids[0, 0] = 1025
    # with torch.no_grad():
    #     out = model(voc_ids, acc_ids, [2,1,3,0])
    # print(out, out.shape)  # (B, T, codebook_size)

    # acc_mask = torch.ones(1, 1).bool().to(device)

    voc_ids = (torch.rand(1, 4, 1500) * 2049).long().to(device)
    model.eval()
    with torch.no_grad():
        generate_out = model.generate(voc_ids)
    print(generate_out, generate_out.shape)

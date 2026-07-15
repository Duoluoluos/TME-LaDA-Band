import torch
import torch.nn as nn
from transformers import AutoModel, Qwen3PreTrainedModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationMixin

class AudioAdapter(nn.Module):
    """
    音频适配器：将音频 Codec 的离散 Token 映射到 Qwen3 的 Hidden Size。
    Mini-Omni2 论文中使用了简单的 Linear 或 MLP 结构 [cite: 119, 290]。
    """
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # 可以增加一个 Projector 层来增强非线性表达，类似 Mini-Omni2 的 adapter 设计
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.adapter(x)
        return self.ln(x)


class OmniQwenMusicModel(Qwen3PreTrainedModel, GenerationMixin):
    """
    基于 Qwen3 架构的 Omni 伴奏生成模型
    继承自 Qwen3PreTrainedModel 以复用权重加载和梯度检查机制
    """
    def __init__(self, model_path: str, attn_implementation: str = "sdpa"):
        super().__init__(AutoConfig.from_pretrained(model_path))
        print("Load Qwen3 from ", model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        self.model.config.voc_vocab_size = 10001
        self.model.config.acc_vocab_size = 10002
        # --- 修改点 1: 从 config 中分别读取 voc 和 acc 的大小 ---
        # 如果 config 里没有，就给默认值，防止报错
        voc_vocab_size = self.model.config.voc_vocab_size
        acc_vocab_size = self.model.config.acc_vocab_size

        # --- 修改点 2: 分别初始化 Adapters ---
        # Vocal Adapter: 需要支持 0~10000
        self.vocal_adapter = AudioAdapter(voc_vocab_size, self.model.config.hidden_size)
        self.vocal_adapter.to(dtype=torch.bfloat16)
        # Acc Adapter: 需要支持 0~10001
        self.acc_adapter = AudioAdapter(acc_vocab_size, self.model.config.hidden_size)
        self.acc_adapter.to(dtype=torch.bfloat16)
        # --- 修改点 3: Output Head ---
        # 我们预测的是 Acc，所以输出维度对应 acc_vocab_size
        self.audio_head = nn.Linear(self.model.config.hidden_size, acc_vocab_size, bias=False)
        self.audio_head.to(dtype=torch.bfloat16)
        # 初始化权重 (Qwen3PreTrainedModel 会自动处理)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 1. 如果 inputs_embeds 存在（第一步），直接使用 embedding，不需要 input_ids
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 2. 如果是自回归生成的后续步骤 (past_key_values 不为空)
            # 此时 input_ids 包含的是上一步生成的 [B, 1] 的音频 Token
            model_inputs = {"input_ids": input_ids}

            # --- 关键 Hack ---
            # 我们需要在 forward 里知道这些 input_ids 是音频，而不是文本。
            # 我们可以通过添加一个 flag 参数传递给 forward
            model_inputs["is_generating_audio"] = True

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


    def forward(
        self,
        input_ids=None,         # [B, Text_Len] (Style Prompts)
        vocal_ids=None,         # [B, Vocal_Len] (Condition)
        ref_ids=None,           # [B, Ref_Len] (Condition)
        acc_ids=None,           # [B, Acc_Len] (Target)
        attention_mask=None,    # [B, Total_Len]
        labels=None,            # [B, Total_Len]
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inputs_embeds=None,
        is_generating_audio=False,
        past_key_values=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        target_dtype = self.model.dtype
        if inputs_embeds is not None:
            pass

        # A. Text (Qwen 原生)
        elif input_ids is not None:
            if not is_generating_audio:
                # 1. 正常的文本 (Training 或 Prompt 阶段)
                inputs_embeds = self.model.embed_tokens(input_ids)
            else:
                # 2. 推理生成阶段 (Autoregressive step)
                # 此时 input_ids 是生成的 Audio Token ID
                inputs_embeds = self.acc_adapter(input_ids)
        else:
            inputs_embeds = torch.tensor([], device=self.device, dtype=target_dtype)

        # B. Vocal (Source Audio)
        if vocal_ids is not None:
            vocal_embeds = self.vocal_adapter(vocal_ids)
        else:
            vocal_embeds = torch.tensor([], device=self.device, dtype=target_dtype)

        # C. Reference (Target Audio)
        if ref_ids is not None:
            ref_embeds = self.acc_adapter(ref_ids)
        else:
            ref_embeds = torch.tensor([], device=self.device, dtype=target_dtype)

        # D. Accompaniment (Target Audio)
        if acc_ids is not None:
            acc_embeds = self.acc_adapter(acc_ids)
        else:
            acc_embeds = torch.tensor([], device=self.device, dtype=target_dtype)

        # 拼接顺序: [Text Prompt] -> [Vocal Condition] -> [Ref Condition] -> [Accompaniment Generation]
        # 参考 Mini-Omni2 Fig 3(a) [cite: 134, 183]
        combined_embeds = torch.cat([inputs_embeds, vocal_embeds, ref_embeds, acc_embeds], dim=1)
        if combined_embeds.dtype != target_dtype:
            combined_embeds = combined_embeds.to(target_dtype)
        # 自动补全 Attention Mask (如果外部未提供)
        if attention_mask is None:
            attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=self.device)

        # --- 2. Qwen Backbone Forward ---
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # --- 3. 计算 Logits ---
        # 使用专门的 audio_head 预测音频 token
        logits = self.audio_head(hidden_states)
        # debug: 打印第一个batch的前50个logits
        # print("First batch logits:", logits[0, :50, :])
        # --- 4. 计算 Loss ---
        loss = None
        if labels is not None:
            # Shift Logits: 预测下一个 Token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            loss_fct = nn.CrossEntropyLoss()

            # 确保在同一设备上
            shift_logits = shift_logits.view(-1, self.model.config.acc_vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

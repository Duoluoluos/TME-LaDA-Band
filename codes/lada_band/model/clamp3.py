import torch
import torch.nn as nn
import omegaconf
import os
import sys
import numpy as np
from transformers import BertConfig, AutoTokenizer


class Clamp3(nn.Module):
    def __init__(self, precision, ckpt_path=None):
        super().__init__()

        # Import clamp3 modules and config
        from clamp3.code.utils import M3Patchilizer, CLaMP3Model

        # Import config parameters
        from clamp3.code.config import (
            MAX_AUDIO_LENGTH, AUDIO_HIDDEN_SIZE, AUDIO_NUM_LAYERS,
            MAX_TEXT_LENGTH, M3_HIDDEN_SIZE, PATCH_LENGTH, PATCH_SIZE, PATCH_NUM_LAYERS,
            TEXT_MODEL_NAME, CLAMP3_HIDDEN_SIZE, CLAMP3_LOAD_M3, CLAMP3_WEIGHTS_PATH
        )


        # Model and configuration setup
        audio_config = BertConfig(vocab_size=1,
                                hidden_size=AUDIO_HIDDEN_SIZE,
                                num_hidden_layers=AUDIO_NUM_LAYERS,
                                num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                                intermediate_size=AUDIO_HIDDEN_SIZE*4,
                                max_position_embeddings=MAX_AUDIO_LENGTH)
        symbolic_config = BertConfig(vocab_size=1,
                                    hidden_size=M3_HIDDEN_SIZE,
                                    num_hidden_layers=PATCH_NUM_LAYERS,
                                    num_attention_heads=M3_HIDDEN_SIZE//64,
                                    intermediate_size=M3_HIDDEN_SIZE*4,
                                    max_position_embeddings=PATCH_LENGTH)

        self.model = CLaMP3Model(audio_config=audio_config,
                            symbolic_config=symbolic_config,
                            text_model_name=TEXT_MODEL_NAME,
                            hidden_size=CLAMP3_HIDDEN_SIZE,
                            load_m3=CLAMP3_LOAD_M3)

        # Load model weights
        self.model.eval()

        # Download weights if not available
        if ckpt_path is None:
            ckpt_path = CLAMP3_WEIGHTS_PATH

        if not os.path.exists(ckpt_path):
            print("No CLaMP 3 weights found. Downloading from Hugging Face...")
            import requests
            from tqdm import tqdm
            checkpoint_url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
            response = requests.get(checkpoint_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(ckpt_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            print("Weights file downloaded successfully.")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint['model'])

        # Initialize tokenizer and patchilizer
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.patchilizer = M3Patchilizer()

        # Initialize MERT model and processor
        from lada_band.utils.mert import mert_init
        self.mert_model, self.mert_processor = mert_init(mert_name='MERT-v0')
        self.mert_model.eval()

        # Configuration parameters
        self.MAX_AUDIO_LENGTH = MAX_AUDIO_LENGTH
        self.AUDIO_HIDDEN_SIZE = AUDIO_HIDDEN_SIZE
        self.MAX_TEXT_LENGTH = MAX_TEXT_LENGTH
        self.PATCH_LENGTH = PATCH_LENGTH
        self.PATCH_SIZE = PATCH_SIZE

        # Precision handling
        self.precision = precision
        self.trans_precision = False

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.mert_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _get_wav_embedding(self, audio) -> torch.Tensor:
        """
        Get audio embedding using Clamp3 model
        audio: [B, T] or [B, 1, T] tensor of audio waveforms
        """
        with torch.cuda.amp.autocast(enabled=False):
            # Inference Check
            if self.precision.startswith('bf16'):
                audio = audio.to(torch.bfloat16)

            # Ensure audio is in the correct format [B, T]
            if audio.ndim == 3 and audio.shape[1] == 1:
                audio = audio.squeeze(1)

            # Extract MERT features for each audio in the batch
            mert_embeddings = []
            from lada_band.utils.mert import mert_predict

            for i in range(audio.shape[0]):
                # Get current audio sample [T]
                audio_sample = audio[i]

                # Extract MERT features using mert_predict
                mert_embed = mert_predict(self.mert_processor, self.mert_model, audio_sample)

                # Average across time to get a single embedding per audio sample
                # This is a simplification - you could use different pooling strategies
                mert_embed_avg = mert_embed.mean(dim=0)  # [T, 768] -> [768]
                mert_embeddings.append(mert_embed_avg)

            # Stack embeddings to get [B, 768]
            mert_embeddings = torch.stack(mert_embeddings)

            # Expand to match Clamp3's expected input format [B, 1, 768]
            mert_embeddings = mert_embeddings.unsqueeze(1)

            # Now process with Clamp3 model
            # Handle precision
            if self.precision == 'bf16-mixed' and not self.trans_precision:
                self.model = self.model.to(torch.float32)
                self.trans_precision = True

            if self.trans_precision:
                with torch.cuda.amp.autocast(enabled=False):
                    audio_embeds = self.model.get_audio_features(
                        audio_inputs=mert_embeddings.float(),
                        audio_masks=torch.ones(mert_embeddings.shape[0], 1),
                        get_global=True
                    )
                audio_embeds = audio_embeds.to(torch.bfloat16)
            else:
                audio_embeds = self.model.get_audio_features(
                    audio_inputs=mert_embeddings,
                    audio_masks=torch.ones(mert_embeddings.shape[0], 1),
                    get_global=True
                )

            audio_embeds = audio_embeds.unsqueeze(1)  # [B, 768] -> [B, 1, 768]
            return audio_embeds

    @torch.no_grad()
    def _get_text_embedding(self, texts) -> torch.Tensor:
        """
        Get text embedding using Clamp3 model
        texts: list of strings
        """
        # Tokenize texts
        input_data = self.tokenizer(
            texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.MAX_TEXT_LENGTH
        )
        input_ids = input_data['input_ids']
        attention_mask = input_data['attention_mask']

        # Handle precision
        if self.precision == 'bf16-mixed' and not self.trans_precision:
            self.model = self.model.to(torch.float32)
            self.trans_precision = True

        if self.trans_precision:
            with torch.cuda.amp.autocast(enabled=False):
                text_embeds = self.model.get_text_features(
                    text_inputs=input_ids,
                    text_masks=attention_mask,
                    get_global=True
                )
            text_embeds = text_embeds.to(torch.bfloat16)
        else:
            text_embeds = self.model.get_text_features(
                text_inputs=input_ids,
                text_masks=attention_mask,
                get_global=True
            )

        text_embeds = text_embeds.unsqueeze(1)  # [B, 768] -> [B, 1, 768]
        return text_embeds

    def forward(self, audio=None, texts=None):
        """
        Forward pass for Clamp3 model
        audio: [B, T] or [B, 1, T] tensor of audio waveforms
        texts: list of strings
        Returns: [B, 1, 768] tensor of embeddings
        """
        assert (audio is not None) ^ (texts is not None), "Either audio or texts must be provided"

        if texts is None:
            with torch.no_grad():
                embeds = self._get_wav_embedding(audio)
        else:
            with torch.no_grad():
                embeds = self._get_text_embedding(texts)

        return embeds
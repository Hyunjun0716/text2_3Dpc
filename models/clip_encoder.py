import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP text encoder to encode text into embeddings.
    The model is frozen (all parameters require_grad=False).
    """
    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", max_length=77, return_sequence=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.return_sequence = return_sequence
        self.freeze()

    def freeze(self):
        """Freeze all parameters"""
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """
        Args:
            text: list of strings or a single string
        Returns:
            If return_sequence=True:
                dict with:
                    'tokens': (B, seq_len, 512) - full token sequence for cross attention
                    'pool': (B, 512) - pooled output for FiLM conditioning
            If return_sequence=False:
                embeddings: (B, 512) - pooled output only (backward compatibility)
        """
        if isinstance(text, str):
            text = [text]

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        )

        tokens = batch_encoding["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.transformer(input_ids=tokens)

            if self.return_sequence:
                # Return both token sequence and pooled output
                return {
                    'tokens': outputs.last_hidden_state,  # (B, seq_len, 512) - for cross attention
                    'pool': outputs.pooler_output         # (B, 512) - for FiLM
                }
            else:
                # Backward compatibility: return pooled output only
                return outputs.pooler_output

    def encode(self, text):
        """Alias for forward"""
        return self.forward(text)

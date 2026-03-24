import torch
import torch.nn.functional as F

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("cloverlm")
class CloverLMHFLM(HFLM):
    def __init__(self, pad_multiple=128, **kwargs):
        super().__init__(**kwargs)
        self.pad_multiple = pad_multiple

    def _encode_pair(self, context, continuation):
        context_enc, continuation_enc = super()._encode_pair(context, continuation)

        if not continuation_enc and continuation:
            whole_enc = self.tok_encode(context + continuation)
            if len(whole_enc) > 1:
                continuation_enc = whole_enc[-1:]
                context_enc = whole_enc[:-1]
            elif whole_enc:
                continuation_enc = whole_enc
                context_enc = [self.prefix_token_id]
            else:
                continuation_enc = [self.prefix_token_id]
                context_enc = [self.prefix_token_id]

        return context_enc, continuation_enc

    def _model_call(self, inps: torch.Tensor, attn_mask: torch.Tensor = None, **kwargs):
        orig_len = inps.shape[1]
        remainder = orig_len % self.pad_multiple
        
        if remainder != 0:
            pad_len = self.pad_multiple - remainder
            inps = F.pad(inps, (0, pad_len), value=self.tokenizer.pad_token_id)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, pad_len), value=0)
                
        logits = super()._model_call(inps, attn_mask=attn_mask, **kwargs)
        if remainder != 0:
            logits = logits[:, :orig_len, :]
        return logits

    def _model_generate(self, context, max_length, **kwargs):
        orig_len = context.shape[1]
        remainder = orig_len % self.pad_multiple
        
        if remainder != 0:
            pad_len = self.pad_multiple - remainder
            context = F.pad(context, (pad_len, 0), value=self.tokenizer.pad_token_id)
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                kwargs["attention_mask"] = F.pad(kwargs["attention_mask"], (pad_len, 0), value=0)
                
        out = super()._model_generate(context, max_length, **kwargs)
        if remainder != 0:
            out = out[:, pad_len:]
            
        return out


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()

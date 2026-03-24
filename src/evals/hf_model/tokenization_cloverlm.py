
from typing import List, Optional
import tokenmonster
from transformers import PreTrainedTokenizer


TOKENMONSTER_URL = (
    "https://huggingface.co/gvlassis/tokenmonster/resolve/main/"
    "englishcode-32000-strict-nocapcode-v1-eot%3D14199.vocab"
    "?download=true"
)


class CloverLMTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_url: str = TOKENMONSTER_URL,
                 eot_id: int = 14199, **kwargs):
        self._tm = tokenmonster.load(vocab_url)
        self._eot_id = eot_id
        self._vocab_size = 32000

        super().__init__(
            eos_token="<eot>",
            pad_token="<eot>",
            bos_token="<eot>",
            **kwargs,
        )
        self.eos_token_id = eot_id
        self.pad_token_id = eot_id
        self.bos_token_id = eot_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        return {f"<tok_{i}>": i for i in range(self._vocab_size)}

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        ids = self._tm.tokenize(text).tolist()
        return [str(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = [int(t) for t in tokens]
        return self._tm.decode(ids)

    @property
    def all_special_tokens_extended(self):
        return [self.eos_token]

    @property
    def all_special_tokens(self):
        return [self.eos_token]

    @property
    def all_special_ids(self):
        return [self._eot_id]

    def save_vocabulary(self, save_directory: str,
                        filename_prefix: Optional[str] = None):
        return ()

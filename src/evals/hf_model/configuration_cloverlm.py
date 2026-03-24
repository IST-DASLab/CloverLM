from transformers import PretrainedConfig


class CloverLMConfig(PretrainedConfig):
    model_type = "cloverlm"

    def __init__(
        self,
        vocab_size=32000,
        num_blocks=4,
        heads=6,
        d_head=128,
        ratio=3,
        scale_type="1/sqrt(d)",
        max_context=1024,
        quartet_2_impl="pseudoquant",
        weight_tying=True,
        attn_backend="pytorch",
        # Optional: HuggingFace / vLLM tooling (defaults derived from shape)
        hidden_size=None,
        intermediate_size=None,
        max_position_embeddings=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        head_dim=None,
        **kwargs,
    ):
        self.num_blocks = num_blocks
        self.num_hidden_layers = num_blocks
        self.heads = heads
        self.d_head = d_head
        self.ratio = ratio
        self.scale_type = scale_type
        self.max_context = max_context
        self.quartet_2_impl = quartet_2_impl
        self.weight_tying = weight_tying
        self.attn_backend = attn_backend

        d_model = heads * d_head
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else 4 * d_model
        )
        self.max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else max_context
        )
        self.num_attention_heads = (
            num_attention_heads if num_attention_heads is not None else heads
        )
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else heads // ratio
        )
        self.head_dim = head_dim if head_dim is not None else d_head

        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            vocab_size=vocab_size,
            tie_word_embeddings=weight_tying,
            **kwargs,
        )

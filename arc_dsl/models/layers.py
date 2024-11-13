from tensorflow import keras
from keras import layers
from keras import ops


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(0.1)
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout_2 = layers.Dropout(0.1)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, query_mask=mask, key_mask=mask
        )

        attention_output = self.dropout_1(attention_output)
        proj_input = self.layernorm_1(inputs + attention_output)

        proj_output = self.dense_proj(proj_input)
        proj_output = self.dropout_2(proj_output)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class PositionalGridEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            name="token-embeddings",
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
        )

        # NOTE: Don't think there is any real reason to separate row/column out instead of one 900 long Embedding
        # But this is how the code ended up at the deadline
        self.row_embeddings = layers.Embedding(
            name="row-embeddings", input_dim=30, output_dim=embed_dim
        )
        self.col_embeddings = layers.Embedding(
            name="col-embeddings", input_dim=30, output_dim=embed_dim
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.supports_masking = True

    def call(self, inputs):
        row_positions = ops.repeat(ops.arange(30), repeats=30)
        col_positions = ops.concatenate([ops.arange(30) for _ in range(30)])
        embedded_tokens = self.token_embeddings(inputs)
        embedded_rows = self.row_embeddings(row_positions)
        embedded_cols = self.col_embeddings(col_positions)
        return embedded_tokens + embedded_rows + embedded_cols

    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            name="token-embeddings",
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
        )
        self.position_embeddings = layers.Embedding(
            name="pos-embeddings", input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.supports_masking = True

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(embed_dim),
            ]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(0.1)
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout_2 = layers.Dropout(0.1)
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout_3 = layers.Dropout(0.1)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        x, encoder_outputs = inputs

        if mask is None:
            decoder_inputs_mask, encoder_outputs_mask = None, None
        else:
            decoder_inputs_mask, encoder_outputs_mask = mask

        self_attn_out = self.self_attention(
            query=x,
            value=x,
            key=x,
            query_mask=decoder_inputs_mask,  # Explicit masking to remove any uncertainty about the automatic masking propagation by Keras.
            use_causal_mask=True,
        )
        self_attn_out = self.dropout_1(self_attn_out)
        out_1 = self.layernorm_1(x + self_attn_out)

        x_attn_out = self.cross_attention(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            # Again, explicit masking from both tensors
            query_mask=decoder_inputs_mask,
            key_mask=encoder_outputs_mask,
        )
        x_attn_out = self.dropout_2(x_attn_out)
        out_2 = self.layernorm_2(out_1 + x_attn_out)

        proj_output = self.dense_proj(out_2)
        proj_output = self.dropout_3(proj_output)
        return self.layernorm_3(out_2 + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

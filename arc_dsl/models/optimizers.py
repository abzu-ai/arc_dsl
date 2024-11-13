import tensorflow as tf


class AttnIsAllUNeedLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": float(self.d_model.numpy()),
            "warmup_steps": self.warmup_steps,
        }


# Have not had time to tweak the parameters. But measurements showed
# that this learned faster than default AdamW and/or ReduceLROnPlateau.
def get_attn_is_all_u_need_optimizer(dense_dim):
    learning_rate = AttnIsAllUNeedLRSchedule(dense_dim)
    return tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

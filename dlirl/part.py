from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    state: jnp.ndarray  # [B, 4, 12]
    action: jnp.ndarray  # [B, A]


class NetworkOutput(NamedTuple):
    successor_features: jax.Array  # [1, 7]
    preference_vectors: jax.Array  # [1, 7]
    # Derived outputs
    policy_params: jax.Array  # [B, num_actions]

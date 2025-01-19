from typing import Any

import haiku as hk
import jax.numpy as jnp

from dlirl.part import NetworkOutput


class Module(hk.Module):
    def __init__(self,
                 num_cumulants: int = 12,
                 batch_size: int = 128):
        super().__init__()
        self._num_cumulants = num_cumulants
        self._batch_size = batch_size

        self.deep_rnn_model1 = hk.DeepRNN([
            hk.VanillaRNN(32),
            hk.LSTM(64),
            hk.LSTM(64),
            hk.GRU(64),
        ])

        self.deep_rnn_model2 = hk.DeepRNN([
            # hk.VanillaRNN(32),
            hk.LSTM(32),
            hk.GRU(64),
        ])

    def __call__(self, pixels_observation: jnp.ndarray, action_index: int) -> jnp.ndarray:
        embedding = pixels_observation  # [B, 4, 12]
        if action_index == 0:
            deep_rnn_state = self.deep_rnn_model1.initial_state(self._batch_size)
            output = None
            for t in range(embedding.shape[1]):
                output, deep_rnn_state = self.deep_rnn_model1(embedding[:, t, :], deep_rnn_state)
            embedding = output  # [B, 64]
            embedding = hk.nets.MLP((32, 32, self._num_cumulants))(embedding)

            return embedding  # [B, 12]
        if action_index == 1:
            # embedding = jax.vmap(hk.Conv1D(32, 4), in_axes=1, out_axes=1)(embedding)
            deep_rnn_state = self.deep_rnn_model2.initial_state(self._batch_size)
            output = None
            for t in range(embedding.shape[1]):
                output, deep_rnn_state = self.deep_rnn_model2(embedding[:, t, :], deep_rnn_state)
            embedding = output  # [B, 64]
            embedding = hk.nets.MLP((64, 32, 32, 32, self._num_cumulants))(embedding)

            return embedding


class Network(hk.Module):
    def __init__(
            self,
            num_actions: int,
            num_cumulants: int = 12,
            batch_size: int = 128,
            successor_features: jnp.ndarray = None,
    ) -> None:
        super().__init__()
        self._num_actions = num_actions
        self._num_cumulants = num_cumulants
        self._batch_size = batch_size

        if successor_features is not None:
            self.successor_features = successor_features
            self.preference_vectors = hk.get_parameter(
                'preference_vectors',
                shape=(self._num_cumulants, 1),
                init=hk.initializers.RandomNormal())
        else:
            self.successor_features = hk.get_parameter(
                'successor_features',
                shape=(self._num_cumulants, 1),
                init=hk.initializers.RandomNormal())
            self.preference_vectors = jnp.ones((self._num_cumulants, 1))
        self.reshape = hk.Reshape(preserve_dims=1,
                                  output_shape=(self._num_cumulants, self._num_actions))
        self.module = Module(batch_size=batch_size)

        self.action_index = 0

    def __call__(self, state_spec: jnp.ndarray, action_index: int) -> Any:  # [B, F, S]
        embedding = self.module(state_spec, action_index)  # [B, 12]

        assert (self.successor_features.shape == (self._num_cumulants, 1)
                and self.preference_vectors.shape == (self._num_cumulants, 1))

        policy_params = jnp.einsum(
            'ca,bc->ba', self.preference_vectors * self.successor_features, embedding)  # pred

        return NetworkOutput(
            self.successor_features,
            self.preference_vectors,
            policy_params)

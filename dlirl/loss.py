import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from dlirl.part import Transition


class Loss:
    def __init__(self, network_fn, batch_size: int = 128) -> None:
        self._index_action = None
        self._network_fn = network_fn
        self._batch_size = batch_size
        self.successor_features = jnp.array([[0, 0, 0, 0, 0, 0, 0]])

    def set_index_action(self, index_action: int):
        self._index_action = index_action

    def __call__(
            self,
            hk_params: hk.Params,
            hk_state: hk.Params,
            transition: Transition,
    ) -> jax.Array:
        @jax.vmap
        def ms_loss_fn(
                pred_logits: jnp.ndarray,
                target: jnp.ndarray
        ) -> jnp.ndarray:
            error = pred_logits - target
            # error = jnp.where(jnp.isnan(error) | jnp.isinf(error), 0, error)
            return jnp.mean(error ** 2)

        state_spec = transition.state
        pred = transition.action

        network_output_t, hk_state = self._network_fn(hk_params, hk_state, state_spec, self._index_action)
        policy_logits = network_output_t.policy_params
        self.successor_features = network_output_t.successor_features

        ms_loss = ms_loss_fn(policy_logits, pred[:, self._index_action])  # [B, 1]
        ms_loss = jnp.sum(ms_loss) / self._batch_size
        ms_loss = jnp.sqrt(ms_loss)

        return ms_loss

    def get_successor_features(self) -> np.ndarray:
        s = []
        for i in range(self.successor_features.shape[0]):
            s.append(self.successor_features[i, 0].item())
        return np.array(s)

    @property
    def network_fn(self):
        return self._network_fn

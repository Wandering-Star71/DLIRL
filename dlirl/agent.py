from typing import Any, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from dlirl.loss import Loss
from dlirl.networks import Network
from dlirl.part import Transition


class Agent:
    def __init__(
            self
    ) -> None:
        self.network = None
        self.optimizer = None
        self.loss_fn = None

    def update(self,
               hk_params: hk.Params,
               hk_states: hk.Params,
               transition: Transition,  # [B, ..]
               opt_state,
               opt_update,
               get_params,
               if_train: bool) -> tuple[Any, Any, Any]:
        loss, grads = jax.value_and_grad(self.loss_fn)(hk_params, hk_states, transition)
        if if_train:
            opt_state = opt_update(0, grads, opt_state)
            return get_params(opt_state), opt_state, loss
        else:
            return None, None, loss

    def initial_params(self, rng_key: int, batch_size: int = 128, action_index: int = None, sf: Sequence[list] = None) \
            -> tuple[hk.Params, hk.Params]:
        self.network = hk.without_apply_rng(
            hk.transform_with_state(
                lambda x, y: Network(
                    num_actions=1,
                    num_cumulants=12,
                    batch_size=batch_size,
                    successor_features=jnp.array(sf[action_index]) if sf is not None else None
                )(x, y)))
        self.optimizer = optimizers.adam
        self.loss_fn = Loss(network_fn=self.network.apply,
                            batch_size=batch_size)
        self.loss_fn.set_index_action(action_index)

        params, states = self.network.init(rng_key, jnp.zeros((batch_size, 4, 12)), action_index)
        return params, states

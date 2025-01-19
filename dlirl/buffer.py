import glob
import math
import os
import pickle
import random
from typing import Sequence, Optional, Union, Any
import haiku as hk
import jax
import jax.numpy as jnp
# import scipy
from jax.random import permutation
from jax.tree_util import tree_map
from tqdm import tqdm

from dlirl.loss import Loss
from dlirl.networks import Network
from dlirl.part import Transition
from dlirl.util import Logger, save_transition_to_disk


class Dataset:
    def __init__(self, data_path: str, if_train: bool = True):
        self.tr_path = data_path + '/transition_cache/train/' if if_train \
            else data_path + '/transition_cache/test/'

    def len(self):
        file_list = glob.glob(os.path.join(self.tr_path, '*.tr.batch'))
        return len(file_list)

    def getitem(self, idx):
        return pickle.load(open(self.tr_path + str(idx) + '.tr.batch', 'rb'))


class ReplayBuffer:
    def __init__(self,
                 Transition_path: str,
                 slice_ratio: float = 0.8,
                 random_key: int = 42,
                 batch_size: int = 256) -> None:
        self.transition_cache_path = Transition_path + '/transition_cache'
        self.batch_size = batch_size
        self.transition_path = Transition_path
        self.slice_ratio = slice_ratio
        self.key = random_key

    def get_batches(self):
        assert os.path.exists(self.transition_path), 'Run Collect_xxx.py first!!! :madge:'
        if not self._if_cache():
            print('No cache found! Generating batches...')
            self.generate_batch()
        else:
            print('Cache found!')
        print('get all batches!')

    @staticmethod
    def stack_t(transitions: Sequence[Transition]) -> Transition:
        return tree_map(lambda *args: jnp.stack(args), *transitions)

    @staticmethod
    def random_t(transition: Transition, random_key: int) -> Transition:
        key = jax.random.PRNGKey(random_key)
        return Transition(permutation(key, transition.state),
                          permutation(key, transition.action))

    def generate_batch(self):
        file_num = len(os.listdir(self.transition_path))
        file_num -= 1 if os.path.exists(self.transition_path+'.DS_Store') else 0
        assert file_num > 0, 'Run Collect_xxx.py first!!! :madge:'

        os.makedirs(self.transition_cache_path + '/train/', exist_ok=True)
        os.makedirs(self.transition_cache_path + '/test/', exist_ok=True)
        t_count = 0
        train_batch_count = 0
        test_batch_count = 0
        ts = []

        for file_index in range(1, file_num+1):
            file_list = glob.glob(os.path.join(self.transition_path + '{}/'.format(file_index), '*'))
            pbar = tqdm(file_list)
            for file_name in pbar:
                pbar.set_description('generate batch of file {}/{}'.format(file_index, file_num))
                t = pickle.load(open(file_name, 'rb'))
                t_count += 1
                ts.append(t)
                if t_count == self.batch_size:
                    t_batch = self.random_t(self.stack_t(ts), self.key)
                    self.key += 1
                    t_count = 0
                    ts.clear()
                    random_number = 1 if random.random() < self.slice_ratio else 0
                    if random_number == 1:
                        save_transition_to_disk(t_batch, os.path.join(self.transition_cache_path + '/train/',
                                                                      '{}.tr.batch'.format(train_batch_count)))
                        train_batch_count += 1
                    else:
                        save_transition_to_disk(t_batch, os.path.join(self.transition_cache_path + '/test/',
                                                                      '{}.tr.batch'.format(test_batch_count)))
                        test_batch_count += 1

    def _if_cache(self) -> bool:
        return os.path.exists(self.transition_cache_path)


class BufferLoop:
    def __init__(self,
                 train_data: Dataset,  # 1 Transition per batch
                 test_data: Dataset,
                 agent,
                 logger: Logger,
                 batch_size: int = 128):
        self._train = train_data
        self._test = test_data
        self._agent = agent
        self._logger = logger
        self._batch_size = batch_size

    def run(self,
            num_episodes: int = 500,
            eval_freq: int = 5,
            train_ration: float = 1,
            seed: Optional[int] = 42,
            num_action: int = 2,
            sf: Optional[Sequence[list]] = None) -> None:
        rng_key = jax.random.PRNGKey(seed)
        random_seed = seed
        rng_key, params_key = jax.random.split(rng_key, num=2)
        assert train_ration <= 1

        for i in range(num_action):
            print('----------Start Training Action {}----------'.format(i))
            self._logger.init_module_path(i)
            params, states = self._agent.initial_params(params_key, batch_size=self._batch_size, action_index=i, sf=sf)
            opt_init, opt_update, get_params = self._agent.optimizer(1e-2)
            opt_state = opt_init(params)

            # loss = None
            # successor_features = None
            for step in range(num_episodes):
                random.seed(random_seed + step)
                pbar = tqdm(random.sample(range(self._train.len()), int(self._train.len()*train_ration)))
                loss_save = []
                for idx in pbar:
                    batch = self._train.getitem(idx)
                    pbar.set_description('train {} {}/{}'.format(i, step, num_episodes - 1))
                    params, opt_state, loss = self._agent.update(
                        params, states, batch, opt_state, opt_update, get_params, True)
                    loss_save.append(loss)
                    # print('loss: {}'.format(_loss))
                self._logger.write_csv(True, step, loss_save)
                # print('train loss: {}'.format(sum(loss_save) / len(loss_save)))

                if step > 0 and (step + 1) % eval_freq == 0:
                    loss_save.clear()
                    random.seed(random_seed + step)
                    pbar = tqdm(random.sample(range(self._test.len()), int(self._test.len()*train_ration)))
                    for idx in pbar:
                        batch = self._test.getitem(idx)
                        pbar.set_description('test {} {}'.format(i, step // eval_freq))
                        _, _, loss = self._agent.update(
                            params, states, batch, None, None, None, False)
                        loss_save.append(loss)
                    self._logger.write_hk_module(self._logger.if_better(loss_save),
                                                 params, states)
                    self._logger.write_csv(False, step, loss_save)
                    print('test loss: {}'.format(sum(loss_save) / len(loss_save)))

                if self._logger.better_count == 0 or step == num_episodes - 1:
                    print('----------Stop training of Module {}!----------'.format(i))
                    print('test last best loss is : {}'.format(self._logger.last_best_loss))
                    break


class BufferLoad:
    def __init__(self,
                 model_path: str,
                 model_path2: Optional[str] = None):
        self._model_pair1, self._model_pair2 = self._read_model(model_path)
        self._network_1, self._network_2 = self._init_model(self._model_pair1, self._model_pair2)
        self._new_net_1, self._new_net_2 = None, None
        # if model_path2 is not None:
        #     self._model2_pair1, self._model2_pair2 = self._read_model(model_path2)
        #     self._network2_1, self._network2_2 = self._init_model(self._model2_pair1, self._model2_pair2)
        #     self._new_net2_1, self._new_net2_2 = None, None
        # else:
        #     self._network2_1, self._network2_2 = None, None

        self._plot_color = ['dodgerblue', 'lightgreen', 'orange', 'yellow', 'mediumpurple']

    @staticmethod
    def _read_model(model_path: str) -> tuple[list, list]:
        hk_param1 = pickle.load(open(model_path + '/model0/param.pickle', 'rb'))
        hk_param2 = pickle.load(open(model_path + '/model1/param.pickle', 'rb'))
        hk_state1 = pickle.load(open(model_path + '/model0/state.pickle', 'rb'))
        hk_state2 = pickle.load(open(model_path + '/model1/state.pickle', 'rb'))
        return [hk_param1, hk_state1], [hk_param2, hk_state2]

    @staticmethod
    def _init_model(model_pair1: list, model_pair2: list, sf_pair=None) -> tuple[Any, Any]:
        network1 = hk.without_apply_rng(hk.transform_with_state(
            lambda x, y: Network(1, 12, 1024,
                                 jnp.array(sf_pair[0]) if sf_pair is not None else None)(x, y)))
        network2 = hk.without_apply_rng(hk.transform_with_state(
            lambda x, y: Network(1, 12, 1024,
                                 jnp.array(sf_pair[1]) if sf_pair is not None else None)(x, y)))

        loss1 = Loss(network_fn=network1.apply, batch_size=1024)
        loss1.set_index_action(0)
        loss2 = Loss(network_fn=network2.apply, batch_size=1024)
        loss2.set_index_action(1)

        m1 = lambda s: (lambda a: [a[i].item() for i in range(a.shape[0])]) \
            (loss1.network_fn(model_pair1[0], model_pair1[1], s, 0)[0].policy_params)

        m2 = lambda s: (lambda a: [a[i].item() / 10 for i in range(a.shape[0])]) \
            (loss2.network_fn(model_pair2[0], model_pair2[1], s, 1)[0].policy_params)

        return m1, m2

    def get_the_successor_feature(self) -> list:
        sf1 = self._model_pair1[0]['network']['successor_features']
        sf2 = self._model_pair2[0]['network']['successor_features']
        return [[x[0].item() for x in sf1], [x[0].item() for x in sf2]]

    @staticmethod
    def _create_new_model(model_pair1: list, model_pair2: list, sf_pair=None) -> tuple[Any, Any]:
        def n1(s, f: list):
            assert len(f) == 7
            f = jnp.array(f)
            new_network = hk.without_apply_rng(hk.transform_with_state(
                lambda x, y: Network(1, 12, 1024, f)(x, y)))
            new_loss = Loss(network_fn=new_network.apply, batch_size=1024)
            new_loss.set_index_action(0)
            a = new_loss.network_fn(model_pair1[0], model_pair1[1], s, 0)[0].policy_params
            return [a[i].item() for i in range(a.shape[0])]

        def n2(s, f: list):
            assert len(f) == 7
            f = jnp.array(f)
            new_network = hk.without_apply_rng(hk.transform_with_state(
                lambda x, y: Network(1, 12, 1024, f)(x, y)))
            new_loss = Loss(network_fn=new_network.apply, batch_size=1024)
            new_loss.set_index_action(1)
            a = new_loss.network_fn(model_pair2[0], model_pair2[1], s, 1)[0].policy_params
            return [a[i].item() / 10 for i in range(a.shape[0])]

        return n1, n2

    @staticmethod
    def plot_show(data_pair: Sequence[Sequence[list]], dataname: int = None):
        # where you draw plots you want
        # depends on what you need
        pass

    @staticmethod
    def calculate_velocity(state: jnp.array, px: list, py: list) -> tuple[list, list]:
        assert state.shape[0] == len(px) == len(py)
        pvx, pvy = [], []
        for i in range(len(px)):
            vx = state[i][3][0]
            vy = state[i][3][1]
            vx_tp1 = vx + 0.2 * px[i]
            vy_tp1 = vy + 0.2 * py[i]
            pvx.append(float(vx_tp1))
            pvy.append(float(vy_tp1))
        return pvx, pvy

    @staticmethod
    def calculate_rmse(t: list, p: list) -> float:
        assert len(t) == len(p)
        return math.sqrt(sum([(t[i] - p[i]) ** 2 for i in range(len(t))]) / 1024)

    @staticmethod
    def calculate_overspeed_rate(pvx: list, dataset_name: int) -> float:
        if dataset_name == 1:  # highd
            speedlimit = 130 / 3.6
        elif dataset_name == 2:  # NGSIM
            speedlimit = 65 * 1.609344 / 3.6
        elif dataset_name == 3:  # DJI
            speedlimit = 120 / 3.6
        else:
            assert False

        sum_over_speed = 0
        for i in pvx:
            if i > speedlimit:
                sum_over_speed += 1
                print('some dude is overspeeding!')
        return sum_over_speed / len(pvx)

    def run_batch(self,
                  dataset: Dataset,
                  batch_index: Union[int, Sequence[int], range],
                  sf: Optional[Sequence[list]] = None):
        batch_index = [batch_index] if isinstance(batch_index, int) else batch_index
        pax_all, pay_all, pvx_all, pvy_all = [], [], [], []
        tax_all, tay_all, tvx_all, tvy_all = [], [], [], []
        for idx in tqdm(batch_index):
            batch = dataset.getitem(idx)
            tax = [batch.action[i][0].item() for i in range(batch.action.shape[0])]
            tay = [batch.action[i][1].item() / 10 for i in range(batch.action.shape[0])]
            tvx, tvy = self.calculate_velocity(batch.state, tax, tay)
            tax_all.extend(tax)
            tay_all.extend(tay)
            tvx_all.extend(tvx)
            tvy_all.extend(tvy)

            if sf is not None:
                _network2_1, _network2_2 = self._init_model(self._model_pair1, self._model_pair2, sf)
                pax = _network2_1(batch.state)
                pay = _network2_2(batch.state)
                pvx, pvy = self.calculate_velocity(batch.state, pax, pay)
                pax_all.extend(pax)
                pay_all.extend(pay)
                pvx_all.extend(pvx)
                pvy_all.extend(pvy)
            else:
                pax = self._network_1(batch.state)  # [1024, 12]
                pay = self._network_2(batch.state)
                pvx, pvy = self.calculate_velocity(batch.state, pax, pay)
                pax_all.extend(pax)
                pay_all.extend(pay)
                pvx_all.extend(pvx)
                pvy_all.extend(pvy)
        # res = self.plot_show([[tax_all,tay_all,tvx_all,tvy_all],[pax_all,pay_all,pvx_all,pvy_all]], data_name)
        # scipy.io.savemat('./dji-highd-fulldata', {'ax': pax_all, 'ay': pay_all, 'vx': pvx_all, 'vy': pvy_all})

        return {}

    def run_single(self,
                   t:Transition, # must be state: [4, 12] action: [, 2]
                   sf: Optional[Sequence[list]] = None):
        state = jnp.tile(jnp.expand_dims(t.state, axis=0), (1024, 1, 1))
        if sf is not None:
            _network2_1, _network2_2 = self._init_model(self._model_pair1, self._model_pair2, sf)
            pax = _network2_1(state)
            pay = _network2_2(state)
        else:
            pax = self._network_1(state)  # [1024, 12]
            pay = self._network_2(state)

        return sum(pax)/1024, sum(pay)/1024 * 10

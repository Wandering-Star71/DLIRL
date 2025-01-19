import csv
import os
from datetime import datetime
from typing import Sequence

import haiku as hk
import pickle5 as pickle


def save_transition_to_disk(t, fname: str):
    with open(fname, 'wb') as f:
        pickle.dump(t, f, pickle.HIGHEST_PROTOCOL)


class Logger:
    def __init__(self, output_path: str) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        timestamp = datetime.now().strftime("%Y-%m%d-%H%M%S")
        # print(timestamp)
        self.log_path = output_path + timestamp + '/'
        self.csv_path = None

        self.csv_writer = None

        self.params_path = None
        self.states_path = None
        self.last_best_loss = None
        self.better_count = None

    def init_module_path(self, index_action: int = 0):
        self.csv_path = self.log_path + 'model{}/log.csv'.format(index_action)
        os.makedirs(self.log_path + 'model{}'.format(index_action), exist_ok=True)
        self._csv_init()
        self.params_path = self.log_path + 'model{}/param.pickle'.format(index_action)
        self.states_path = self.log_path + 'model{}/state.pickle'.format(index_action)
        # self.successors_path = self.log_path + 'model{}/successors.pickle'.format(index_action)
        self.last_best_loss = 1e5  # just set big
        self.better_count = 8

    def _csv_init(self) -> None:
        head_row = ['epoch', 'loss_all_mean', 'loss_all_min', 'loss_all_max']
        with open(self.csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(head_row)

    def write_csv(self, if_train: bool,
                  step: int,
                  loss_all: Sequence[float]) -> None:
        assert len(loss_all) > 0
        loss_all_mean = sum(loss_all) / len(loss_all)
        epoch = step if if_train else -step
        with open(self.csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch, loss_all_mean, min(loss_all), max(loss_all)])

    def if_better(self, loss: Sequence[float]) -> bool:
        loss_mean_now = sum(loss) / len(loss)
        if loss_mean_now < self.last_best_loss:
            self.last_best_loss = loss_mean_now
            return True
        else:
            self.better_count -= 1
            return False

    def write_hk_module(self, if_replace: bool,
                        hk_params: hk.Params,
                        hk_states: hk.Params) -> None:
        if if_replace:
            with open(self.params_path, 'wb') as f:
                pickle.dump(hk_params, f)
            with open(self.states_path, 'wb') as f:
                pickle.dump(hk_states, f)
            # with open(self.successors_path, 'wb') as f:
            #     pickle.dump(successor_features, f)
            print('its indeed better!')
        with open(self.params_path + '.last', 'wb') as f:
            pickle.dump(hk_params, f)
        with open(self.states_path + '.last', 'wb') as f:
            pickle.dump(hk_states, f)

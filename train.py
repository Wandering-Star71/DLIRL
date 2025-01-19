import random
import argparse
# import dingsound as d
from dlirl.agent import Agent
from dlirl.buffer import BufferLoop, ReplayBuffer, Dataset
from dlirl.util import Logger


def get_args():
    parser = argparse.ArgumentParser(description='xxx')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--out_saved_path', type=str, default='./save_param/',
                        help='path to save the trained model')
    parser.add_argument('--data_path', type=str, default='../HighDRawData/transition/',
                        help='transition data path when you finished collecting datd from dataset like HighD')
    parser.add_argument('--sf_pair', default=None,
                        help='set to None if you first train from a culture, set to a sf_pair like below example'
                             'when you want to do cross-cultural deployment to another culture')

# sf_pair example:
# sf_pair = [
#     [[1.0028020e-03], [8.0429208e-01], [-8.2532410e-05], [-6.2308460e-04], [1.1352326e-03], [-3.8277765e-03],
#      [-1.8484713e-04], [5.1234567e-03], [-2.3456789e-04], [7.8901234e-02], [4.5678901e-04], [-9.8765432e-05]],
#     [[-1.9813540e-04], [7.1619213e-01], [-2.2591578e-06], [6.8353097e-07], [-1.1687593e-04], [2.4629571e-07],
#      [-4.0282539e-06], [3.4567890e-03], [1.2345678e-04], [-5.6789012e-03], [8.9012345e-04], [6.5432109e-05]]
# ]

    return parser.parse_args()

def main(arg):
    random.seed(arg.seed)

    read_buffer = ReplayBuffer(arg.data_path,
                               batch_size=arg.batch_size,
                               random_key=arg.seed)
    read_buffer.get_batches()
    train_buffer = Dataset(arg.data_path, True)
    test_buffer = Dataset(arg.data_path, False)

    logger = Logger(arg.out_saved_path)

    agent = Agent()
    loop_buffer = BufferLoop(train_buffer, test_buffer, agent, logger, arg.batch_size)

    # num_episodes, eval_freq, num_action, train_ration, can all be set
    loop_buffer.run(seed=arg.seed, sf=arg.sf_pair)


if __name__ == '__main__':
    arg = get_args()
    main(arg)
    # ding when your train is finished :D
    # d.ding()

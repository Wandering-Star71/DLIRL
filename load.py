import argparse
from typing import Union, Sequence
from dlirl.buffer import Dataset, BufferLoad
# import dingsound as d


def get_args():
    parser = argparse.ArgumentParser(description='xxx')

    parser.add_argument('--batch_index', type=Union[int, Sequence[int], range], default=0,
                        help='specify index of which batch to load from test dataset')
    parser.add_argument('--model_path', type=str, default='./save_param/2025-0119-110932',
                        help='model path to be loaded')
    # parser.add_argument('--model_path2', type=str, default='save_param/DJI-sf-10%',
    #                     help='second model')
    parser.add_argument('--data_path', type=str, default='../HighDRawData/transition',
                        help='transition data path')
    parser.add_argument('--sf_pair', default=None,
                        help='see more discription from train.py')

    return parser.parse_args()


def main(arg):
    eval_buffer = BufferLoad(arg.model_path)
    # sf_pair = eval_buffer.get_the_successor_feature() # how to get sf_pair

    test_data = Dataset(arg.data_path, False)

    pred_res = eval_buffer.run_batch(test_data, arg.batch_index, sf=arg.sf_pair)
    # print(pred_res)


if __name__ == '__main__':
    args = get_args()
    main(args)
    # d.ding()

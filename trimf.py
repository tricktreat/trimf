import argparse

from args import train_argparser, eval_argparser
from config_reader import process_configs
from trimf import input_reader
from trimf.trimf_trainer import TriMFTrainer
# import os
import trimf.util as util

import warnings

warnings.filterwarnings("ignore")
def __train(run_args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = run_args.device_id
    
    trainer = TriMFTrainer(run_args)
    # util.set_seed(999)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)

 
def __eval(run_args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = run_args.device_id
    trainer = TriMFTrainer(run_args)
    # util.set_seed(999)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python trimf.py train ...'")

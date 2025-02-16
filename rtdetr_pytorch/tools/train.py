"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import yaml


def print_config(config):
    print(yaml.dump(config, sort_keys=False))


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed(backend='nccl')
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=args.amp)
    print_config(cfg.yaml_cfg)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)

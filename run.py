# coding = utf-8

import sys
import argparse

from util.experiment import Experiment
from util.conf import Configuration


def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters for Indexing Embedding experiments')
    
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    parser.add_argument('-E', '--embed', default=False, dest='to_embed', action='store_true', help='whether to embed database/query')

    args = parser.parse_args(argv[1: ])

    conf = Configuration(args.confpath, dump=True)

    if args.to_embed:
        conf.setHP('to_embed', True)

    experiment = Experiment(conf)
    experiment.run()


if __name__ == "__main__":
    main(sys.argv)
    
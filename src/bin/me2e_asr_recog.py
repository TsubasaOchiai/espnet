#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--gpu', '-g', default='-1', type=str,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    # task related
    # noisy
    parser.add_argument('--recog-feat-noisy', type=str, required=True,
                        help='Filename of recognition feature data (Kaldi scp) without enhancemet')
    parser.add_argument('--recog-label-noisy', type=str, required=True,
                        help='Filename of recognition label data (json) without enhancement')
    parser.add_argument('--result-label-noisy', type=str, required=True,
                        help='Filename of result label data (json) without enhancement')
    # enhan
    parser.add_argument('--recog-feat-enhan', type=str, nargs='*', required=True,
                        help='Filename of recognition feature data (Kaldi scp) with enhancemet')
    parser.add_argument('--recog-label-enhan', type=str, required=True,
                        help='Filename of recognition label data (json) with enhancement')
    parser.add_argument('--result-label-enhan', type=str, required=True,
                        help='Filename of result label data (json) with enhancement')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, required=True,
                        help='Model config file')
    # search related
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', default=0.0, type=float,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain max output length.'
                        + 'If maxlenratio=0.0 (default), it uses a end-detect function'
                        + 'to automatically find maximum hypothesis lengths')
    parser.add_argument('--minlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', default=0.0, type=float,
                        help='CTC weight in joint decoding')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--lm-weight', default=0.1, type=float,
                        help='RNNLM weight.')
    # me2e
    parser.add_argument('--melmat', type=str, required=True,
                        help='Filename of Mel-filterbank matrix data (Kaldi ark)')
    parser.add_argument('--cmvn', type=str, required=True,
                        help='Filename of cmvn statistics data (Kaldi ark)')
    parser.add_argument('--mode', default=None, type=str, required=True,
                        choices=['noisy', 'enhan'],
    args = parser.parse_args()

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # display PYTHONPATH
    logging.info('python path = ' + os.environ['PYTHONPATH'])

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # recog
    logging.info('backend = ' + args.backend)
    if args.backend == "chainer":
        from me2e_asr_chainer import recog
        recog(args)
    elif args.backend == "pytorch":
        raise NotImplementedError('currently, pytorch is not supported.')
        from me2e_asr_pytorch import recog
        recog(args)
    else:
        raise ValueError("chainer and pytorch are only supported.")


if __name__ == '__main__':
    main()

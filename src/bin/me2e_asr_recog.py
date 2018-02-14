#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import json
import logging
import os
import pickle
import random

# chainer related
import chainer
import numpy as np

# espnet related
from beamformer import NB_MVDR
from e2e_asr_attctc import E2E
from me2e_asr_attctc import ME2E
from e2e_asr_attctc import Loss

# for kaldi io
import kaldi_io_py
import lazy_io

# rnnlm
import lm_train


def converter_kaldi(name, readers, mode):
    # noisy mode
    if mode == 'noisy':
        bidim = readers[name].shape[1] / 2
        # separate real and imaginary part
        feat_real = readers[name][:,:bidim]
        feat_imag = readers[name][:,bidim:]
    # enhancement mode
    elif mode == 'enhan':
        bidim = readers[0][name].shape[1] / 2
        # separate real and imaginary part
        feat_real = [reader[name][:,:bidim] for reader in readers]
        feat_imag = [reader[name][:,bidim:] for reader in readers]
    else:
        logging.error(
            "Error: need to specify an appropriate decoding mode")
        sys.exit()

    feat = {}
    feat['real'] = feat_real
    feat['imag'] = feat_imag

    return feat
        

def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--gpu', '-g', default='-1', type=str,
                        help='GPU ID (negative value indicates CPU)')
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

    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    # seed setting (chainer seed may not need it)
    nseed = args.seed
    random.seed(nseed)
    np.random.seed(nseed)
    os.environ["CHAINER_SEED"] = str(nseed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        bidim, eidim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # get Mel-filterbank
    melmat = read_mat(args.melmat)

    # get cmvn statistics
    stats = read_mat(args.cmvn)
    dim = len(stats[0]) - 1
    count = stats[0][dim]
    cmvn_mean = stats[0][0:dim]/count
    cmvn_std = np.sqrt(stats[1][0:dim]/count - cmvn_mean*cmvn_mean)
    cmvn_stats = (cmvn_mean, cmvn_std)

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    enhan = NB_MVDR(bidim, args)
    asr = E2E(eidim, odim, args)
    me2e = ME2E(enhan, asr, melmat, cmvn_stats)
    model = Loss(me2e, args.mtlalpha)
    chainer.serializers.load_npz(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_train.ClassifierWithState(lm_train.RNNLM(len(train_args.char_list), 650))
        chainer.serializers.load_npz(args.rnnlm, rnnlm)

    # prepare Kaldi reader
    # noisy
    names_noisy = [key for key,mat in kaldi_io.read_mat_scp(args.recog_feat_noisy)]
    recog_reader_noisy = kaldi_io_py.read_dict_scp(args.recog_feat_noisy)
    # enhan
    names_enhan = [key for key,mat in kaldi_io.read_mat_scp(args.recog_feat_enhan)]
    recog_reader_enhan = [kaldi_io_py.read_dict_scp(feat) for feat in args.recog_feat_enhan]

    # read json data
    with open(args.recog_label_noisy, 'rb') as f:
        recog_json_noisy = json.load(f)['utts']
    with open(args.recog_label_enhan, 'rb') as f:
        recog_json_enhan = json.load(f)['utts']

    if args.mode == 'noisy':
        names = names_noisy
        recog_reader = recog_reader_noisy
        recog_json = recog_json_noisy
    elif args.mode == 'enhan':
        names = names_enhan
        recog_reader = recog_reader_enhan
        recog_json = recog_json_enhan

    new_json = {}
    for name in names:
        logging.info('decoding ' + name)

        feat = converter_kaldi(name, recog_reader, args.mode)

        y_hat = me2e.recognize(feat, args, train_args.char_list, rnnlm)
        y_true = map(int, recog_json[name]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = recog_json[name]

        # added recognition results to json
        new_json[name]['rec_tokenid'] = " ".join(
            [str(idx[0]) for idx in y_hat])
        new_json[name]['rec_token'] = " ".join(seq_hat)
        new_json[name]['rec_text'] = seq_hat_text

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4).encode('utf_8'))


if __name__ == '__main__':
    main()

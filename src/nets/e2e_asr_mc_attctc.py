#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging

import numpy as np

import chainer
import chainer.functions as F
from e2e_asr_attctc import _subsamplex


class LogMel(chainer.Chain):

    def __init__(self, melmat):
        super(LogMel, self).__init__()

        self.melmat = melmat

    def __call__(self, feats, xp):
        melmat = chainer.Variable(xp.array(self.melmat, dtype=np.float32))
        feats_real = feats['real']
        feats_imag = feats['imag']

        logmel_feats = []
        for feat_real, feat_imag in zip(feats_real, feats_imag):
            pow_feat = feat_real ** 2 + feat_imag ** 2
            mel_feat = F.matmul(pow_feat, melmat, False, True)
            logmel_feats.append(F.log(mel_feat + 10 ** (-20)))

        return logmel_feats


class CMVN(chainer.Chain):

    def __init__(self, cmvn_stats):
        super(CMVN, self).__init__()

        self.cmvn_mean = cmvn_stats[0]
        self.cmvn_std = cmvn_stats[1]

    def __call__(self, feats, xp):
        cmvn_mean = chainer.Variable(xp.array(self.cmvn_mean, dtype=np.float32))
        cmvn_std = chainer.Variable(xp.array(self.cmvn_std, dtype=np.float32))

        cmvn_feats = []
        for feat in feats:
            feat -= F.broadcast_to(cmvn_mean, shape=feat.shape)
            feat /= F.broadcast_to(cmvn_std, shape=feat.shape)
            cmvn_feats.append(feat)

        return cmvn_feats


class E2E_MC(chainer.Chain):
    def __init__(self, enhan, asr, melmat, cmvn_stats):
        super(E2E_MC, self).__init__()

        with self.init_scope():
            # enhan
            self.enhan = enhan
            # asr
            self.asr = asr

        self.logmel = LogMel(melmat)
        self.cmvn = CMVN(cmvn_stats)

    # x[i]: ('utt_id', {'ilen':'xxx',...}})
    def __call__(self, data):
        '''E2E_MC forward

        :param data:
        :return:
        '''
        return self.asr(data, self._pre_call)

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E greedy/beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        return self.asr.recognize(x, recog_args, char_list, rnnlm, self._pre_recognize)

    def _pre_call(self, data):
        utt_type = data[0][1]['utt_type']
        logging.info('utt_type: ' + utt_type)

        if utt_type == 'noisy':
            # utt list of frame x dim
            xs_real = [i[1]['feat']['real'] for i in data]
            xs_imag = [i[1]['feat']['imag'] for i in data]
            ilens = self.xp.array([utt.shape[0] for utt in xs_real], dtype=np.int32)
            hs_real = [chainer.Variable(self.xp.array(utt, dtype=np.float32)) for utt in xs_real]
            hs_imag = [chainer.Variable(self.xp.array(utt, dtype=np.float32)) for utt in xs_imag]

            hs = {}
            hs['real'] = hs_real
            hs['imag'] = hs_imag
        elif utt_type == 'enhan':
            # utt list of channel list of frame x dim
            xs_real = [i[1]['feat']['real'] for i in data]
            xs_imag = [i[1]['feat']['imag'] for i in data]
            ilens = self.xp.array([utt[0].shape[0] for utt in xs_real], dtype=np.int32)
            hs_real = [[chainer.Variable(self.xp.array(ch, dtype=np.float32))
                        for ch in utt] for utt in xs_real]
            hs_imag = [[chainer.Variable(self.xp.array(ch, dtype=np.float32))
                        for ch in utt] for utt in xs_imag]

            hs = {}
            hs['real'] = hs_real
            hs['imag'] = hs_imag

            # 1. beamformer
            hs = self.enhan(hs)

        # utt list of olen
        ys = [self.xp.array(
            list(map(int, i[1]['tokenid'].split())), dtype=np.int32) for i in data]
        ys = [chainer.Variable(y) for y in ys]

        # 2. feature extractor
        hs = self.logmel(hs, self.xp)
        hs = self.cmvn(hs, self.xp)

        # subsample frame
        hs, ilens = _subsamplex(hs, self.asr.subsample[0])

        return hs, ys, ilens

    def _pre_recognize(self, x):
        x_real = x['real']
        x_imag = x['imag']
        ilen = self.xp.array(x_real[0].shape[0], dtype=np.int32)
        h_real = [chainer.Variable(self.xp.array(ch, dtype=np.float32)) for ch in x_real]
        h_imag = [chainer.Variable(self.xp.array(ch, dtype=np.float32)) for ch in x_imag]

        # make a utt list (1) to use the same interface for encoder
        hs = {}
        hs['real'] = [h_real]
        hs['imag'] = [h_imag]

        # 1. beamformer
        hs = self.enhan(hs)

        # 2. feature extractor
        hs = self.logmel(hs, self.xp)
        hs = self.cmvn(hs, self.xp)

        # subsample frame
        hs, ilens = _subsamplex(hs, self.asr.subsample[0])

        return hs, ilens
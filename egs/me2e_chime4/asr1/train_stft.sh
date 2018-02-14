
#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=10
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# decoding parameter
beam_size=20
penalty=0
penalty=0
maxlenratio=0.8
minlenratio=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# me2e
cmvn=data-fbank/tr05_multi_noisy/cmvn.ark
melmat=exp/make_melmat/melmat.ark
mode="noisy+enhan" # training mode

# data
chime4_data=/home/xtochiai/corpora/CHiME4

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh 
. ./cmd.sh 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr05_multi_noisy

if [ -z ${tag} ]; then
    expdir=exp/${mode}_${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        echo "delta feature is not supported"
        exit
#        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

# switch backend
if [[ ${backend} == chainer ]]; then
    train_script=me2e_asr_train.py
    decode_script=me2e_asr_recog.py
else
    echo "torch backend is not supported"
    exit
#    train_script=me2e_asr_train_th.py
#    decode_script=me2e_asr_recog_th.py
fi

# Only for this script
dict=data/lang_1char/${train_set}_units.txt

# noisy
train_feat_noisy="scp:data-stft/tr05_multi_noisy/feats.scp"
valid_feat_noisy="scp:data-stft/dt05_multi_isolated_1ch_track/feats.scp"

# enhan
train_feat_enhan=""
valid_feat_enhan=""
stft_ch="1 3 4 5 6"
for ch in ${stft_ch}; do
    train_feat_enhan="${train_feat_enhan} scp:data-stft/tr05_multi_noisy_ch${ch}/feats.scp"
    valid_feat_enhan="${valid_feat_enhan} scp:data-stft/dt05_multi_noisy_ch${ch}/feats.scp"
done

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} ${expdir}/train.log \
        ${train_script} \
        --gpu ${gpu} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --train-feat-noisy ${train_feat_noisy} \
        --valid-feat-noisy ${valid_feat_noisy} \
        --train-label-noisy data-stft/tr05_multi_noisy/data.json \
        --valid-label-noisy data-stft/dt05_multi_isolated_1ch_track/data.json \
        --train-feat-enhan ${train_feat_enhan} \
        --valid-feat-enhan ${valid_feat_enhan} \
        --train-label-enhan data-stft/tr05_multi_noisy_ch1/data.json \
        --valid-label-enhan data-stft/dt05_multi_noisy_ch1/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --melmat ${melmat} \
        --cmvn ${cmvn} \
        --mode ${mode}
fi

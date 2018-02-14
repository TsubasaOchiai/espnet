
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

recog_set="\
dt05_real_noisy dt05_simu_noisy et05_real_noisy et05_simu_noisy
"

if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}

        # noisy
        # split data
        data=data-stft/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        # make json labels for recognition
        data2json.sh ${data} ${dict} > ${data}/data.json
        
        recog_feat_noisy="scp:data-stft/${rtask}/split${nj}utt/JOB/feats.scp"

        # enhan
        # split_data
        for ch in ${stft_ch}:
            data=data-stft/${rtask}_ch${ch}
            split_data.sh --per-utt ${data} ${nj};
        done
        # make json labels for recognition
        data=data-stft/${rtask}_ch1
        data2json.sh ${data} ${dict} > ${data}/data.json

        recog_feat_enhan=""
        stft_ch="1 3 4 5 6"
        for ch in ${stft_ch}:
            recog_feat_enhan="${recog_feat_enhan} scp:data-stft/${rtask}_ch${ch}/split${nj}utt/JOB/feats.scp"
        done

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            ${decode_script} \
            --gpu ${gpu} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-feat-noisy ${recog_feat_noisy} \
            --recog-label-noisy data-stft/${rtask}/data.json \
            --recog-feat-enhan ${recog_feat_enhan} \
            --recog-label-enhan data-stft/${rtask}_ch1/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --melmat ${melmat} \
            --cmvn ${cmvn} \
            --mode ${mode} \
            &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


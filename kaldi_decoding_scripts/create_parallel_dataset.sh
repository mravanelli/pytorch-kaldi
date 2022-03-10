#!/usr/bin/env bash

kaldi_dir=/home/<username>/repos/kaldi/egs/librispeech/s5

cd $kaldi_dir

dataset_directories=$1  # Ex. train-noisy
data=$2                 # ~/data

dataset_names=$(echo $dataset_directories | sed s/-/_/g)

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc

gmmdir=exp/tri4b

stage=1
nj=8

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  for part in $dataset_directories; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
fi

echo "Stage 1 complete"

if [ $stage -le 2 ]; then

  for part in $dataset_names; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

fi

echo "Stage 2 complete"

if [ $stage -le 3 ]; then

  for part in $dataset_names; do
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/$part data/lang exp/tri4b exp/tri4b_ali_$part
  done
  
fi

echo "Stage 3 complete"

wait

for chunk in $dataset_names; do
    dir=fmllr/$chunk
    steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
        --transform-dir exp/tri4b_ali_$chunk $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1

    compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
done

echo "Stage 4 complete"

for part in $dataset_names; do
  # aligments
  steps/align_fmllr.sh --nj $nj data/$part data/lang exp/tri4b exp/tri4b_ali_$part

done

wait

echo "Dataset created successfully"
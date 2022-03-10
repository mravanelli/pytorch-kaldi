#!/usr/bin/env bash

kaldi_dir=/home/<username>/repos/kaldi/egs/librispeech/s5

cd $kaldi_dir

dataset_name=$1        # Ex. train_noisy
nr_utterances=$2       # Ex. 10000
dataset_new_name=$3    # Ex. train_noisy_10k

mfccdir=mfcc


gmmdir=exp/tri4b


nj=12

stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e


if [ $stage -le 1 ]; then
  utils/subset_data_dir.sh --first data/$dataset_name $nr_utterances data/$dataset_new_name
fi

echo "Stage 1 complete"

if [ $stage -le 2 ]; then

  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$dataset_new_name exp/make_mfcc/$dataset_new_name $mfccdir
  steps/compute_cmvn_stats.sh data/$dataset_new_name exp/make_mfcc/$dataset_new_name $mfccdir

fi

echo "Stage 2 complete"

if [ $stage -le 3 ]; then

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/$dataset_new_name data/lang $gmmdir ${gmmdir}_ali_$dataset_new_name

fi

echo "Stage 3 complete"


for chunk in $dataset_new_name; do
    dir=fmllr/$chunk
    steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
        --transform-dir ${gmmdir}_ali_$chunk $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1

    compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
done


echo "Stage 4 complete"

for part in $dataset_new_name; do
  # aligments
  steps/align_fmllr.sh --nj $nj data/$part data/lang ${gmmdir} ${gmmdir}_ali_$part
done

echo "Subset created"
#!/bin/bash

#Mirco Ravanelli – Jan 2018 (mirco.ravanelli@gmail.com)

# This script first shuffles or sorts (based on the sentence length) a kaldi feature list and then split it into a certain number of chunks.
# Shuffling a list could be good for feed-forward DNNs, while a sorted list can be useful for RNNs. 
# 
# Example for mfcc:
# Ordered chunks: ./create_chunks.sh /home/mirco/kaldi-trunk/egs/timit/s5/data/train splits_fea 5 train 1
# Shuffled chunks: ./create_chunks.sh /home/mirco/kaldi-trunk/egs/timit/s5/data/train splits_fea 5 train 0

# Example for fmllr:
# Ordered chunks: ./create_chunks.sh /home/mirco/kaldi-trunk/egs/timit/s5/data-fmllr-tri3/train splits_fea 5 train 1
# Shuffled chunks: ./create_chunks.sh /home/mirco/kaldi-trunk/egs/timit/s5/data-fmllr-tri3/train splits_fea 5 train 0
#
# Note: this scripts assumes that sox in installed.
#

data_folder=$1
out_folder=$2
N_chunks=$3
name_out=$4
ord=$5 # 0- shuffle fea list, 1-ordered list (from shorter to longer sentence)

rm -rf $out_folder/$name_out\_split*
rm -rf $out_folder/$name_out*.ark

mkdir -p $out_folder


if [ "$ord" = "1" ]; then
 echo "Computing sentence lenghts..."
 
 cat $data_folder/wav.scp | \
 while read -r line ; do
    id="$(echo $line | awk '{print $1}')"
    file="$(echo $line | awk '{gsub(/\|/,"");print $NF}')"
    len="$(soxi -D $file)"
    echo $len
  done > $data_folder/file_len.scp

  paste $data_folder/file_len.scp $data_folder/feats.scp | sort -n | awk {'$1="";print $0'} | cut -c 2- > $data_folder/feats_ord.scp

  total_lines=$(wc -l <$data_folder/feats_ord.scp)
  ((lines_per_file = (total_lines + $N_chunks - 1) / $N_chunks))
  split -d -a 3 --lines=${lines_per_file} $data_folder/feats_ord.scp $out_folder"/"$name_out"_split."

  echo "features in $out_folder have been ordered and split into $N_chunks chunks."

else
 echo "Feature shuffling"
 cat $data_folder/feats.scp | sort -R > $data_folder/feats_shuf.scp

 total_lines=$(wc -l <$data_folder/feats_shuf.scp)
 ((lines_per_file = (total_lines + $N_chunks - 1) / $N_chunks))
 split -d -a 3 --lines=${lines_per_file} $data_folder/feats_shuf.scp $out_folder"/"$name_out"_split."

 echo "features in $out_folder have been shuffled and split into $N_chunks chunks."

fi


# Normalization options
compute-cmvn-stats --spk2utt=ark:$data_folder/spk2utt scp:$data_folder/feats.scp ark:$out_folder/$name_out"_cmvn_speaker.ark"
compute-cmvn-stats scp:$data_folder/feats.scp ark:$out_folder/$name_out"_cmvn_snt.ark"

echo 'done cmnvs.'

# do not forget to generate aligmnents (labels) for dev and test sets:
#steps/nnet/align.sh --nj 4 /home/mirco/kaldi-trunk/egs/timit/s5/data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev
#steps/nnet/align.sh --nj 4 /home/mirco/kaldi-trunk/egs/timit/s5/data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test



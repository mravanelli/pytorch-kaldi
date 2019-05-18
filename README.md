# The PyTorch-Kaldi Speech Recognition Toolkit
<img src="pytorch-kaldi_logo.png" width="220" img align="left">
PyTorch-Kaldi is an open-source repository for developing state-of-the-art DNN/HMM speech recognition systems. The DNN part is managed by PyTorch, while feature extraction, label computation, and decoding are performed with the Kaldi toolkit.

This repository contains the last version of the  PyTorch-Kaldi toolkit (PyTorch-Kaldi-v1.0). To take a look into the previous version (PyTorch-Kaldi-v0.1), [click here](https://bitbucket.org/mravanelli/pytorch-kaldi-v0.0/src/master/).

If you use this code or part of it, please cite the following paper:

*M. Ravanelli, T. Parcollet, Y. Bengio, "The PyTorch-Kaldi Speech Recognition Toolkit", [arXiv](https://arxiv.org/abs/1811.07453)*

```
@inproceedings{pytorch-kaldi,
title    = {The PyTorch-Kaldi Speech Recognition Toolkit},
author    = {M. Ravanelli and T. Parcollet and Y. Bengio},
booktitle    = {In Proc. of ICASSP},
year    = {2019}
}
```

The toolkit is released under a **Creative Commons Attribution 4.0 International license**. You can copy, distribute, modify the code for research, commercial and non-commercial purposes. We only ask to cite our paper referenced above.

To improve transparency and replicability of speech recognition results, we give users the possibility to release their PyTorch-Kaldi model within this repository. Feel free to contact us (or doing a pull request) for that. Moreover, if your paper uses PyTorch-Kaldi, it is also possible to advertise it in this repository.

[See a short introductory video on the PyTorch-Kaldi Toolkit](https://www.youtube.com/watch?v=VDQaf0SS4K0&t=2s)

## Next Version
We are actively working on the next version of PyTorch-Kaldi (v0.3). The architecture of the toolkit will be more modular and flexible. Beyond speech recognition, the new toolkit will be suitable for other applications such as speaker recognition, speech enhancement, speech separation, etc.
The toolkit will support labels and features in many different formats (not just the current kaldi one) and will be much easier feeding the system with the raw samples directly. The toolkit will support a number of data processing functions, including data augmentation and contamination. We are also working to support self-supervised training.  

## Table of Contents
* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [How to install](#how-to-install)
* [Recent Updates](#recent-updates)
* [Tutorials:](#timit-tutorial)
  * [TIMIT tutorial](#timit-tutorial)
  * [Librispeech tutorial](#librispeech-tutorial)
* [Toolkit Overview:](#overview-of-the-toolkit-architecture)
  * [Toolkit architecture](#overview-of-the-toolkit-architecture)
  * [Configuration files](#description-of-the-configuration-files)
* [FAQs:](#how-can-i-plug-in-my-model)
  * [How can I plug-in my model?](#how-can-i-plug-in-my-model)
  * [How can I tune the hyperparameters?](#how-can-i-tune-the-hyperparameters)
  * [How can I use my own dataset?](#how-can-i-use-my-own-dataset)
  * [How can I plug-in my own features?](#how-can-i-plug-in-my-own-features)
  * [How can I transcript my own audio files?](#how-can-i-transcript-my-own-audio-files)
  * [Batch size, learning rate, and droput scheduler](#Batch-size,-learning-rate,-and-dropout-scheduler)
  * [How can I contribute to the project?](#how-can-i-contribute-to-the-project)
* [EXTRA:](#speech-recognition-from-the-raw-waveform-with-sincnet)  
  * [Speech recognition from the raw waveform with SincNet](#speech-recognition-from-the-raw-waveform-with-sincnet)
  * [Joint training between speech enhancement and ASR](#joint-training-between-speech-enhancement-and-asr)
  * [Distant Speech Recognition with DIRHA](#distant-speech-recognition-with-dirha)
  * [Training an autoencoder](#training-an-autoencoder)
* [References](#references)


## Introduction
The PyTorch-Kaldi project aims to bridge the gap between the Kaldi and the PyTorch toolkits, trying to inherit the efficiency of Kaldi and the flexibility of PyTorch. PyTorch-Kaldi is not only a simple interface between these toolkits, but it embeds several useful features for developing modern speech recognizers. For instance, the code is specifically designed to naturally plug-in user-defined acoustic models. As an alternative, users can exploit several pre-implemented neural networks that can be customized using intuitive configuration files. PyTorch-Kaldi supports multiple feature and label streams as well as combinations of neural networks, enabling the use of complex neural architectures. The toolkit is publicly-released along with rich documentation and is designed to properly work locally or on HPC clusters.

Some features of the new version of the PyTorch-Kaldi toolkit:

- Easy interface with Kaldi.
- Easy plug-in of user-defined models.
- Several pre-implemented models (MLP, CNN, RNN, LSTM, GRU, Li-GRU, SincNet).
- Natural implementation of complex models based on multiple features, labels, and neural architectures.
- Easy and flexible configuration files.
- Automatic recovery from the last processed chunk.
- Automatic chunking and context expansions of the input features.
- Multi-GPU training.
- Designed to work locally or on HPC clusters.
- Tutorials on TIMIT and Librispeech Datasets.

## Prerequisites
1. If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of the Kaldi binaries into $HOME/.bashrc. For instance, make sure that .bashrc contains the following paths:
```
export KALDI_ROOT=/home/mirco/kaldi-trunk
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH
```
Remember to change the KALDI_ROOT variable using your path. As a first test to check the installation, open a bash shell, type "copy-feats" or "hmm-info" and make sure no errors appear.

2. If not already done, install PyTorch (http://pytorch.org/). We tested our codes on PyTorch 1.0 and PyTorch 0.4. An older version of PyTorch is likely to raise errors. To check your installation, type “python” and, once entered into the console, type “import torch”, and make sure no errors appear.

3. We recommend running the code on a GPU machine. Make sure that the CUDA libraries (https://developer.nvidia.com/cuda-downloads) are installed and correctly working. We tested our system on Cuda 9.0, 9.1 and 8.0. Make sure that python is installed (the code is tested with python 2.7 and python 3.7). Even though not mandatory, we suggest using Anaconda (https://anaconda.org/anaconda/python).

## Recent updates

**19 Feb. 2019: updates:**
- It is now possible to dynamically change batch size, learning rate, and dropout factors during training. We thus implemented a scheduler that supports the following formalism within the config files:
```
batch_size_train = 128*12 | 64*10 | 32*2
```
The line above means: do 12 epochs with 128 batches, 10 epochs with 64 batches, and 2 epochs with 32 batches. A similar formalism can be used for learning rate and dropout scheduling. [See this section for more information](#batch-size,-learning-rate,-and-dropout-scheduler).

**5 Feb. 2019: updates:**
1. Our toolkit now supports parallel data loading (i.e., the next chunk is stored in memory while processing the current chunk). This allows a significant speed up.
2. When performing monophone regularization users can now set “dnn_lay = N_lab_out_mono”. This way the number of monophones is automatically inferred by our toolkit.
3. We integrated the kaldi-io toolkit from the [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python) project into data_io-py.
4. We provided a better hyperparameter setting for SincNet ([see this section](#speech-recognition-from-the-raw-waveform-with-sincnet))
5. We released some baselines with the DIRHA dataset ([see this section](#distant-speech-recognition-with-dirha)). We also provide some configuration examples for a simple autoencoder ([see this section](#training-an-autoencoder)) and for a system that jointly trains a speech enhancement and a speech recognition module ([see this section](#joint-training-between-speech-enhancement-and-asr))
6. We fixed some minor bugs.

**Notes on the next version:**
In the next version, we plan to further extend the functionalities of our toolkit, supporting more models and features formats. The goal is to make our toolkit suitable for other speech-related tasks such as end-to-end speech recognition, speaker-identification, keyword spotting, speech separation, speech activity detection, speech enhancement, etc. If you would like to propose some novel functionalities, please give us your feedback by [filling this survey](https://docs.google.com/forms/d/12jd-QP5m8NAJVpiypvtVGy1n_d2iuWaLozXq5hsg4yA/edit?usp=sharing).



## How to install
To install PyTorch-Kaldi, do the following steps:

1. Make sure all the software recommended in the “Prerequisites” sections are installed and are correctly working
2. Clone the PyTorch-Kaldi repository:
```
git clone https://github.com/mravanelli/pytorch-kaldi
```
3.  Go into the project folder and Install the needed packages with:
```
pip install -r requirements.txt
```


## TIMIT tutorial
In the following, we provide a short tutorial of the PyTorch-Kaldi toolkit based on the popular TIMIT dataset.

1. Make sure you have the TIMIT dataset. If not, it can be downloaded from the LDC website (https://catalog.ldc.upenn.edu/LDC93S1).

2. Make sure Kaldi and PyTorch installations are fine. Make also sure that your KALDI paths are currently working (you should add the Kaldi paths into the .bashrc as reported in the section "Prerequisites"). For instance, type "copy-feats" and "hmm-info" and make sure no errors appear. 

3. Run the Kaldi s5 baseline of TIMIT. This step is necessary to compute features and labels later used to train the PyTorch neural network. We recommend running the full timit s5 recipe (including the DNN training): 

```
cd kaldi/egs/timit/s5
./run.sh
./local/nnet/run_dnn.sh
```

This way all the necessary files are created and the user can directly compare the results obtained by Kaldi with that achieved with our toolkit.

4. Compute the alignments (i.e, the phone-state labels) for test and dev data with the following commands (go into $KALDI_ROOT/egs/timit/s5). If you want to use tri3 alignments, type:
```
steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
```

If you want to use dnn alignments (as suggested), type:
```
steps/nnet/align.sh --nj 4 data-fmllr-tri3/train data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali

steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
```

5. We start this tutorial with a very simple MLP network trained on mfcc features.  Before launching the experiment, take a look at the configuration file  *cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg*. See the [Description of the configuration files](#description-of-the-configuration-files) for a detailed description of all its fields. 

6. Change the config file according to your paths. In particular:
- Set “fea_lst” with the path of your mfcc training list (that should be in $KALDI_ROOT/egs/timit/s5/data/train/feats.scp)
- Add your path (e.g., $KALDI_ROOT/egs/timit/s5/data/train/utt2spk) into “--utt2spk=ark:”
- Add your CMVN transformation e.g.,$KALDI_ROOT/egs/timit/s5/mfcc/cmvn_train.ark
- Add the folder where labels are stored (e.g.,$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali for training and ,$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev for dev data).

To avoid errors make sure that all the paths in the cfg file exist. **Please, avoid using paths containing bash variables since paths are read literally and are not automatically expanded** (e.g., use /home/mirco/kaldi-trunk/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali instead of $KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali)

7. Run the ASR experiment:
```
python run_exp.py cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg
```

This script starts a full ASR experiment and performs training, validation, forward, and decoding steps.  A progress bar shows the evolution of all the aforementioned phases. The script *run_exp.py* progressively creates the following files in the output directory:

- *res.res*: a file that summarizes training and validation performance across various validation epochs.
- *log.log*: a file that contains possible errors and warnings.
- *conf.cfg*: a copy of the configuration file.
- *model.svg* is a picture that shows the considered model and how the various neural networks are connected. This is really useful to debug models that are more complex than this one (e.g, models based on multiple neural networks).
- The folder *exp_files* contains several files that summarize the evolution of training and validation over the various epochs. For instance, files *.info report chunk-specific information such as the chunk_loss and error and the training time. The *.cfg files are the chunk-specific configuration files (see general architecture for more details), while files *.lst report the list of features used to train each specific chunk.
- At the end of training, a directory called *generated outputs* containing plots of loss and errors during the various training epochs is created.

**Note that you can stop the experiment at any time.** If you run again the script it will automatically start from the last chunk correctly processed. The training could take a couple of hours, depending on the available GPU. Note also that if you would like to change some parameters of the configuration file (e.g., n_chunks=,fea_lst=,batch_size_train=,..) you must specify a different output folder (output_folder=).

**Debug:** If you run into some errors, we suggest to do the following checks:
1.    Take a look into the standard output.
2.    If it is not helpful, take a look into the log.log file.
3.    Take a look into the function run_nn into the core.py library. Add some prints in the various part of the function to isolate the problem and figure out the issue.


8. At the end of training, the phone error rate (PER\%) is appended into the res.res file. To see more details on the decoding results, you can go into “decoding_test” in the output folder and take a look to the various files created.  For this specific example, we obtained the following *res.res* file:


```
ep=000 tr=['TIMIT_tr'] loss=3.398 err=0.721 valid=TIMIT_dev loss=2.268 err=0.591 lr_architecture1=0.080000 time(s)=86
ep=001 tr=['TIMIT_tr'] loss=2.137 err=0.570 valid=TIMIT_dev loss=1.990 err=0.541 lr_architecture1=0.080000 time(s)=87
ep=002 tr=['TIMIT_tr'] loss=1.896 err=0.524 valid=TIMIT_dev loss=1.874 err=0.516 lr_architecture1=0.080000 time(s)=87
ep=003 tr=['TIMIT_tr'] loss=1.751 err=0.494 valid=TIMIT_dev loss=1.819 err=0.504 lr_architecture1=0.080000 time(s)=88
ep=004 tr=['TIMIT_tr'] loss=1.645 err=0.472 valid=TIMIT_dev loss=1.775 err=0.494 lr_architecture1=0.080000 time(s)=89
ep=005 tr=['TIMIT_tr'] loss=1.560 err=0.453 valid=TIMIT_dev loss=1.773 err=0.493 lr_architecture1=0.080000 time(s)=88
.........
ep=020 tr=['TIMIT_tr'] loss=0.968 err=0.304 valid=TIMIT_dev loss=1.648 err=0.446 lr_architecture1=0.002500 time(s)=89
ep=021 tr=['TIMIT_tr'] loss=0.965 err=0.304 valid=TIMIT_dev loss=1.649 err=0.446 lr_architecture1=0.002500 time(s)=90
ep=022 tr=['TIMIT_tr'] loss=0.960 err=0.302 valid=TIMIT_dev loss=1.652 err=0.447 lr_architecture1=0.001250 time(s)=88
ep=023 tr=['TIMIT_tr'] loss=0.959 err=0.301 valid=TIMIT_dev loss=1.651 err=0.446 lr_architecture1=0.000625 time(s)=88
%WER 18.1 | 192 7215 | 84.0 11.9 4.2 2.1 18.1 99.5 | -0.583 | /home/mirco/pytorch-kaldi-new/exp/TIMIT_MLP_basic5/decode_TIMIT_test_out_dnn1/score_6/ctm_39phn.filt.sys
```

The achieved PER(%) is 18.1%. Note that there could be some variability in the results, due to different initializations on different machines. We believe that averaging the performance obtained with different initialization seeds (i.e., change the field *seed* in the config file) is crucial for TIMIT since the natural performance variability might completely hide the experimental evidence. We noticed a standard deviation of about 0.2% for the TIMIT experiments.

If you want to change the features, you have to first compute them with the Kaldi toolkit. To compute fbank features, you have to open *$KALDI_ROOT/egs/timit/s5/run.sh* and compute them with the following lines:
```
feadir=fbank

for x in train dev test; do
  steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_fbank/$x $feadir
  steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $feadir
done
```

Then, change the aforementioned configuration file with the new feature list.
If you already have run the full timit Kaldi recipe, you can directly find the fmllr features in *$KALDI_ROOT/egs/timit/s5/data-fmllr-tri3*.
If you feed the neural network with such features you should expect a substantial performance improvement, due to the adoption of the speaker adaptation.

In the TIMIT_baseline folder, we propose several other examples of possible TIMIT baselines. Similarly to the previous example, you can run them by simply typing:
```
python run_exp.py $cfg_file
```
There are some examples with recurrent (TIMIT_RNN*,TIMIT_LSTM*,TIMIT_GRU*,TIMIT_LiGRU*) and CNN architectures (TIMIT_CNN*). We also propose a more advanced model (TIMIT_DNN_liGRU_DNN_mfcc+fbank+fmllr.cfg) where we used a combination of feed-forward and recurrent neural networks fed by a concatenation of mfcc, fbank, and fmllr features. Note that the latter configuration files correspond to the best architecture described in the reference paper. As you might see from the above-mentioned configuration files, we improve the ASR performance by including some tricks such as the monophone regularization (i.e., we jointly estimate both context-dependent and context-independent targets). The following table reports the results obtained by running the latter systems (average PER\%):

| Model  | mfcc | fbank | fMLLR | 
| ------ | -----| ------| ------| 
|  Kaldi DNN Baseline | -----| ------| 18.5 |
|  MLP  | 18.2 | 18.7 | 16.7 | 
|  RNN  | 17.7 | 17.2 | 15.9 | 
|  SRU  | -----| 16.6 | -----|
|LSTM| 15.1  | 14.3  |14.5  | 
|GRU| 16.0 | 15.2|  14.9 | 
|li-GRU| **15.5**  | **14.9**|  **14.2** | 

Results show that, as expected, fMLLR features outperform MFCCs and FBANKs coefficients, thanks to the speaker adaptation process. Recurrent models significantly outperform the standard MLP one, especially when using LSTM, GRU, and Li-GRU architecture, that effectively address gradient vanishing through multiplicative gates. The best result *PER=$14.2$\%* is obtained with the [Li-GRU model](https://arxiv.org/pdf/1803.10225.pdf) [2,3], that is based on a single gate and thus saves 33% of the computations over a standard GRU. 

The best results are actually obtained with a more complex architecture that combines MFCC, FBANK, and fMLLR features (see *cfg/TIMI_baselines/TIMIT_mfcc_fbank_fmllr_liGRU_best.cfg*). To the best of our knowledge, the **PER=13.8\%** achieved by the latter system yields the best-published performance on the TIMIT test-set. 

The Simple Recurrent Units (SRU) is an efficient and highly parallelizable recurrent model. Its performance on ASR is worse than standard LSTM, GRU, and Li-GRU models, but it is significantly faster. SRU is implemented [here](https://github.com/taolei87/sru) and described in the following paper:

T. Lei, Y. Zhang, S. I. Wang, H. Dai, Y. Artzi, "Simple Recurrent Units for Highly Parallelizable Recurrence, Proc. of EMNLP 2018. [arXiv](https://arxiv.org/pdf/1709.02755.pdf)

To do experiments with this model, use the config file *cfg/TIMIT_baselines/TIMIT_SRU_fbank.cfg*. Before you should install the model using ```pip install sru``` and you should uncomment "import sru" in *neural_networks.py*.


You can directly compare your results with ours by going [here](https://bitbucket.org/mravanelli/pytorch-kaldi-exp-timit/src/master/). In this external repository, you can find all the folders containing the generated files.

## Librispeech tutorial
The steps to run PyTorch-Kaldi on the Librispeech dataset are similar to that reported above for TIMIT. The following tutorial is based on the *100h sub-set*, but it can be easily extended to the full dataset (960h).

1. Run the Kaldi recipe for librispeech at least until Stage 13 (included)
2. Copy exp/tri4b/trans.* files into exp/tri4b/decode_tgsmall_train_clean_100/
```
mkdir exp/tri4b/decode_tgsmall_train_clean_100 && cp exp/tri4b/trans.* exp/tri4b/decode_tgsmall_train_clean_100/
```
3. Compute the fmllr features by running the following script. 

```
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

gmmdir=exp/tri4b

for chunk in train_clean_100 dev_clean test_clean; do
    dir=fmllr/$chunk
    steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
        --transform-dir $gmmdir/decode_tgsmall_$chunk \
            $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1

    compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
done
```

4. compute aligmenents using:
```
# aligments on dev_clean and test_clean
steps/align_fmllr.sh --nj 30 data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100
steps/align_fmllr.sh --nj 10 data/dev_clean data/lang exp/tri4b exp/tri4b_ali_dev_clean_100
steps/align_fmllr.sh --nj 10 data/test_clean data/lang exp/tri4b exp/tri4b_ali_test_clean_100
```

5. run the experiments with the following command:
```
  python run_exp.py cfg/Librispeech_baselines/libri_MLP_fmllr.cfg
```

If you would like to use a recurrent model you can use *libri_RNN_fmllr.cfg*, *libri_LSTM_fmllr.cfg*, *libri_GRU_fmllr.cfg*, or *libri_liGRU_fmllr.cfg*. The training of recurrent models might take some days (depending on the adopted GPU).  The performance obtained with the tgsmall graph are reported in the following table:

| Model  | WER% | 
| ------ | -----| 
|  MLP  |  9.6 |
|LSTM   |  8.6  |
|GRU     | 8.6 | 
|li-GRU| 8.6 |

These results are obtained without adding a lattice rescoring (i.e., using only the *tgsmall* graph). You can improve the performance by adding lattice rescoring in this way (run it from the *kaldi_decoding_script* folder of Pytorch-Kaldi):
```
data_dir=/data/milatmp1/ravanelm/librispeech/s5/data/
dec_dir=/u/ravanelm/pytorch-Kaldi-new/exp/libri_fmllr/decode_test_clean_out_dnn1/
out_dir=/u/ravanelm/pytorch-kaldi-new/exp/libri_fmllr/

steps/lmrescore_const_arpa.sh  $data_dir/lang_test_{tgsmall,fglarge} \
          $data_dir/test_clean $dec_dir $out_dir/decode_test_clean_fglarge   || exit 1;
```
The final results obtaineed using rescoring (*fglarge*) are reported in the following table:

| Model  | WER% |  
| ------ | -----| 
|  MLP  |  6.5 |
|LSTM   |  6.4  |
|GRU     | 6.3 | 
|li-GRU| **6.2**  |


You can take a look into the results obtained [here](https://bitbucket.org/mravanelli/pytorch-kaldi-exp-librispeech/src/master/).




## Overview of the toolkit architecture
The main script to run an ASR experiment is **run_exp.py**. This python script performs training, validation, forward, and decoding steps.  Training is performed over several epochs, that progressively process all the training material with the considered neural network.
After each training epoch, a validation step is performed to monitor the system performance on *held-out* data. At the end of training, the forward phase is performed by computing the posterior probabilities of the specified test dataset. The posterior probabilities are normalized by their priors (using a count file) and stored into an ark file. A decoding step is then performed to retrieve the final sequence of words uttered by the speaker in the test sentences.

The *run_exp.py* script takes in input a global config file (e.g., *cfg/TIMIT_MLP_mfcc.cfg*) that specifies all the needed options to run a full experiment. The code *run_exp.py* calls another function  **run_nn** (see core.py library) that performs training, validation, and forward operations on each chunk of data.
The function *run_nn* takes in input a chunk-specific config file (e.g, exp/TIMIT_MLP_mfcc/exp_files/train_TIMIT_tr+TIMIT_dev_ep000_ck00.cfg*) that specifies all the needed parameters for running a single-chunk experiment. The run_nn function outputs some info filles (e.g., *exp/TIMIT_MLP_mfcc/exp_files/train_TIMIT_tr+TIMIT_dev_ep000_ck00.info*) that summarize losses and errors of the processed chunk.

The results are summarized into the *res.res* files, while errors and warnings are redirected into the *log.log* file.


## Description of the configuration files:
There are two types of config files (global and chunk-specific cfg files). They are both in *INI* format and are read, processed, and modified with the *configparser* library of python. 
The global file contains several sections, that specify all the main steps of a speech recognition experiments (training, validation, forward, and decoding).
The structure of the config file is described in a prototype file (see for instance *proto/global.proto*) that not only lists all the required sections and fields but also specifies the type of each possible field. For instance, *N_ep=int(1,inf)* means that the fields *N_ep* (i.e, number of training epochs) must be an integer ranging from 1 to inf. Similarly, *lr=float(0,inf)* means that the lr field (i.e., the learning rate) must be a float ranging from 0 to inf. Any attempt to write a config file not compliant with these specifications will raise an error.

Let's now try to open a config file (e.g., *cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg*) and let's describe the main sections:

```
[cfg_proto]
cfg_proto = proto/global.proto
cfg_proto_chunk = proto/global_chunk.proto
```
The current version of the config file first specifies the paths of the global and chunk-specific prototype files in the section *[cfg_proto]*.

```
[exp]
cmd = 
run_nn_script = run_nn
out_folder = exp/TIMIT_MLP_basic5
seed = 1234
use_cuda = True
multi_gpu = False
save_gpumem = False
n_epochs_tr = 24
```
The section [exp] contains some important fields, such as the output folder (*out_folder*) and the path of the chunk-specific processing script *run_nn* (by default this function should be implemented in the core.py library). The field *N_epochs_tr* specifies the selected number of training epochs. Other options about using_cuda, multi_gpu, and save_gpumem can be enabled by the user. The field *cmd* can be used to append a command to run the script on a HPC cluster.

```
[dataset1]
data_name = TIMIT_tr
fea = fea_name=mfcc
    fea_lst=quick_test/data/train/feats_mfcc.scp
    fea_opts=apply-cmvn --utt2spk=ark:quick_test/data/train/utt2spk  ark:quick_test/mfcc/train_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
    cw_left=5
    cw_right=5
    
lab = lab_name=lab_cd
    lab_folder=quick_test/dnn4_pretrain-dbn_dnn_ali
    lab_opts=ali-to-pdf
    lab_count_file=auto
    lab_data_folder=quick_test/data/train/
    lab_graph=quick_test/graph
    
n_chunks = 5

[dataset2]
data_name = TIMIT_dev
fea = fea_name=mfcc
    fea_lst=quick_test/data/dev/feats_mfcc.scp
    fea_opts=apply-cmvn --utt2spk=ark:quick_test/data/dev/utt2spk  ark:quick_test/mfcc/dev_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
    cw_left=5
    cw_right=5
    
lab = lab_name=lab_cd
    lab_folder=quick_test/dnn4_pretrain-dbn_dnn_ali_dev
    lab_opts=ali-to-pdf
    lab_count_file=auto
    lab_data_folder=quick_test/data/dev/
    lab_graph=quick_test/graph
n_chunks = 1

[dataset3]
data_name = TIMIT_test
fea = fea_name=mfcc
    fea_lst=quick_test/data/test/feats_mfcc.scp
    fea_opts=apply-cmvn --utt2spk=ark:quick_test/data/test/utt2spk  ark:quick_test/mfcc/test_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
    cw_left=5
    cw_right=5
    
lab = lab_name=lab_cd
    lab_folder=quick_test/dnn4_pretrain-dbn_dnn_ali_test
    lab_opts=ali-to-pdf
    lab_count_file=auto
    lab_data_folder=quick_test/data/test/
    lab_graph=quick_test/graph
    
n_chunks = 1
```

The config file contains a number of sections (*[dataset1]*, *[dataset2]*, *[dataset3]*,...) that describe all the corpora used for the ASR experiment.  The fields on the *[dataset\*]* section describe all the features and labels considered in the experiment.
The features, for instance, are specified in the field *fea:*, where *fea_name* contains the name given to the feature, *fea_lst* is the list of features (in the scp Kaldi format), *fea_opts* allows users to specify how to process the features (e.g., doing CMVN or adding the derivatives), while *cw_left* and *cw_right* set the characteristics of the context window (i.e., number of left and right frames to append). Note that the current version of the PyTorch-Kaldi toolkit supports the definition of multiple features streams. Indeed, as shown in *cfg/TIMIT_baselines/TIMIT_mfcc_fbank_fmllr_liGRU_best.cfg* multiple feature streams (e.g., mfcc, fbank, fmllr) are employed.

Similarly, the *lab* section contains some sub-fields.  For instance, *lab_name* refers to the name given to the label,  while *lab_folder* contains the folder where the alignments generated by the Kaldi recipe are stored.  *lab_opts* allows the user to specify some options on the considered alignments. For example  *lab_opts="ali-to-pdf"* extracts standard context-dependent phone-state labels, while *lab_opts=ali-to-phones --per-frame=true* can be used to extract monophone targets. *lab_count_file* is used to specify the file that contains the counts of the considered phone states. 
These counts are important in the forward phase, where the posterior probabilities computed by the neural network are divided by their priors. PyTorch-Kaldi allows users to both specify an external count file or to automatically retrieve it (using *lab_count_file=auto*). Users can also specify *lab_count_file=none* if the count file is not strictly needed, e.g., when the labels correspond to an output not used to generate the posterior probabilities used in the forward phase (see for instance the monophone targets in *cfg/TIMIT_baselines/TIMIT_MLP_mfcc.cfg*). *lab_data_folder*, instead, corresponds to the data folder created during the Kaldi data preparation. It contains several files, including the text file eventually used for the computation of the final WER. The last sub-field *lab_graph* is the path of the Kaldi graph used to generate the labels.  

The full dataset is usually large and cannot fit the GPU/RAM memory. It should thus be split into several chunks.  PyTorch-Kaldi automatically splits the dataset into the number of chunks specified in *N_chunks*. The number of chunks might depend on the specific dataset. In general, we suggest processing speech chunks of about 1 or 2 hours (depending on the available memory).

```
[data_use]
train_with = TIMIT_tr
valid_with = TIMIT_dev
forward_with = TIMIT_test
```

This section tells how the data listed into the sections *[datasets\*]* are used within the *run_exp.py* script. 
The first line means that we perform training with the data called *TIMIT_tr*. Note that this dataset name must appear in one of the dataset sections, otherwise the config parser will raise an error. Similarly,  the second and third lines specify the data used for validation and forward phases, respectively.

```
[batches]
batch_size_train = 128
max_seq_length_train = 1000
increase_seq_length_train = False
start_seq_len_train = 100
multply_factor_seq_len_train = 2
batch_size_valid = 128
max_seq_length_valid = 1000
```

*batch_size_train* is used to define the number of training examples in the mini-batch.  The fields *max_seq_length_train* truncates the sentences longer than the specified value. When training recurrent models on very long sentences, out-of-memory issues might arise. With this option, we allow users to mitigate such memory problems by truncating long sentences. Moreover, it is possible to progressively grow the maximum sentence length during training by setting *increase_seq_length_train=True*. If enabled, the training starts with a maximum sentence length specified in start_seq_len_train (e.g, *start_seq_len_train=100*). After each epoch the maximum sentence length is multiplied by the multply_factor_seq_len_train (e.g *multply_factor_seq_len_train=2*).
We have observed that this simple strategy generally improves the system performance since it encourages the model to first focus on short-term dependencies and learn longer-term ones only at a later stage.

Similarly,*batch_size_valid* and *max_seq_length_valid* specify the number of examples in the mini-batches and the maximum length for the dev dataset.

```
[architecture1]
arch_name = MLP_layers1
arch_proto = proto/MLP.proto
arch_library = neural_networks
arch_class = MLP
arch_pretrain_file = none
arch_freeze = False
arch_seq_model = False
dnn_lay = 1024,1024,1024,1024,N_out_lab_cd
dnn_drop = 0.15,0.15,0.15,0.15,0.0
dnn_use_laynorm_inp = False
dnn_use_batchnorm_inp = False
dnn_use_batchnorm = True,True,True,True,False
dnn_use_laynorm = False,False,False,False,False
dnn_act = relu,relu,relu,relu,softmax
arch_lr = 0.08
arch_halving_factor = 0.5
arch_improvement_threshold = 0.001
arch_opt = sgd
opt_momentum = 0.0
opt_weight_decay = 0.0
opt_dampening = 0.0
opt_nesterov = False
```

The sections *[architecture\*]* are used to specify the architectures of the neural networks involved in the ASR experiments. The field *arch_name* specifies the name of the architecture. Since different neural networks can depend on a different set of hyperparameters, the user has to add the path of a proto file that contains the list of hyperparameters into the field *proto*.  For example,  the prototype file for a standard MLP model contains the following fields:
```
[proto]
library=path
class=MLP
dnn_lay=str_list
dnn_drop=float_list(0.0,1.0)
dnn_use_laynorm_inp=bool
dnn_use_batchnorm_inp=bool
dnn_use_batchnorm=bool_list
dnn_use_laynorm=bool_list
dnn_act=str_list
```

Similarly to the other prototype files, each line defines a hyperparameter with the related value type. All the hyperparameters defined in the proto file must appear into the global configuration file under the corresponding *[architecture\*]* section.
The field *arch_library* specifies where the model is coded (e.g. *neural_nets.py*), while *arch_class* indicates the name of the class where the architecture is implemented (e.g. if we set *class=MLP* we will do *from neural_nets.py import MLP*).

The field *arch_pretrain_file* can be used to pre-train the neural network with a previously-trained architecture, while *arch_freeze* can be set to *False* if you want to train the parameters of the architecture during training and should be set to *True* do keep the parameters fixed (i.e., frozen) during training. The section *arch_seq_model* indicates if the architecture is sequential (e.g. RNNs) or non-sequential (e.g., a  feed-forward MLP or CNN). The way PyTorch-Kaldi processes the input batches is different in the two cases. For recurrent neural networks (*arch_seq_model=True*) the sequence of features is not randomized (to preserve the elements of the sequences), while for feedforward models  (*arch_seq_model=False*) we randomize the features (this usually helps to improve the performance). In the case of multiple architectures, sequential processing is used if at least one of the employed architectures is marked as sequential  (*arch_seq_model=True*).

Note that the hyperparameters starting with "arch_" and "opt_" are mandatory and must be present in all the architecture specified in the config file. The other hyperparameters (e.g., dnn_*, ) are specific of the considered architecture (they depend on how the class MLP is actually implemented by the user) and can define number and typology of hidden layers, batch and layer normalizations, and other parameters.
Other important parameters are related to the optimization of the considered architecture. For instance, *arch_lr* is the learning rate, while *arch_halving_factor* is used to implement learning rate annealing. In particular, when the relative performance improvement on the dev-set between two consecutive epochs is smaller than that specified in the *arch_improvement_threshold* (e.g, arch_improvement_threshold) we multiply the learning rate by the *arch_halving_factor* (e.g.,*arch_halving_factor=0.5*). The field arch_opt specifies the type of optimization algorithm. We currently support SGD, Adam, and Rmsprop. The other parameters are specific to the considered optimization algorithm (see the PyTorch documentation for exact meaning of all the optimization-specific hyperparameters).
Note that the different architectures defined in *[archictecture\*]* can have different optimization hyperparameters and they can even use a different optimization algorithm.

```
[model]
model_proto = proto/model.proto
model = out_dnn1=compute(MLP_layers1,mfcc)
    loss_final=cost_nll(out_dnn1,lab_cd)
    err_final=cost_err(out_dnn1,lab_cd)
```    

The way all the various features and architectures are combined is specified in this section with a very simple and intuitive meta-language.
The field *model:* describes how features and architectures are connected to generate as output a set of posterior probabilities. The line *out_dnn1=compute(MLP_layers,mfcc)* means "*feed the architecture called MLP_layers1 with the features called mfcc and store the output into the variable out_dnn1*”.
From the neural network output *out_dnn1* the error and the loss functions are computed using the labels called *lab_cd*, that have to be previously defined into the *[datasets\*]* sections. The *err_final* and *loss_final* fields are mandatory subfields that define the final output of the model.

A much more complex example (discussed here just to highlight the potentiality of the toolkit) is reported in *cfg/TIMIT_baselines/TIMIT_mfcc_fbank_fmllr_liGRU_best.cfg*:
```    
[model]
model_proto=proto/model.proto
model:conc1=concatenate(mfcc,fbank)
      conc2=concatenate(conc1,fmllr)
      out_dnn1=compute(MLP_layers_first,conc2)
      out_dnn2=compute(liGRU_layers,out_dnn1)
      out_dnn3=compute(MLP_layers_second,out_dnn2)
      out_dnn4=compute(MLP_layers_last,out_dnn3)
      out_dnn5=compute(MLP_layers_last2,out_dnn3)
      loss_mono=cost_nll(out_dnn5,lab_mono)
      loss_mono_w=mult_constant(loss_mono,1.0)
      loss_cd=cost_nll(out_dnn4,lab_cd)
      loss_final=sum(loss_cd,loss_mono_w)     
      err_final=cost_err(out_dnn4,lab_cd)
```    
In this case we first concatenate mfcc, fbank, and fmllr features and we then feed a MLP. The output of the MLP is fed into the a recurrent neural network (specifically a Li-GRU model). We then have another MLP layer (*MLP_layers_second*) followed by two softmax classifiers (i.e., *MLP_layers_last*, *MLP_layers_last2*). The first one estimates standard context-dependent states, while the second estimates monophone targets. The final cost function is a weighted sum between these two predictions. In this way we implement the monophone regularization, that turned out to be useful to improve the ASR performance.

The full model can be considered as a single big computational graph, where all the basic architectures used in the [model] section are jointly trained. For each mini-batch, the input features are propagated through the full model and the cost_final is computed using the specified labels. The gradient of the cost function with respect to all the learnable parameters of the architecture is then computed. All the parameters of the employed architectures are then updated together with the algorithm specified in the *[architecture\*]* sections.
```    
[forward]
forward_out = out_dnn1
normalize_posteriors = True
normalize_with_counts_from = lab_cd
save_out_file = True
require_decoding = True
```    

The section forward first defines which is the output to forward (it must be defined into the model section).  if *normalize_posteriors=True*, these posterior are normalized by their priors (using a count file).  If *save_out_file=True*, the posterior file (usually a very big ark file) is stored, while if *save_out_file=False* this file is deleted when not needed anymore.
The *require_decoding* is a boolean that specifies if we need to decode the specified output.  The field *normalize_with_counts_from* set which counts using to normalize the posterior probabilities.

```    
[decoding]
decoding_script_folder = kaldi_decoding_scripts/
decoding_script = decode_dnn.sh
decoding_proto = proto/decoding.proto
min_active = 200
max_active = 7000
max_mem = 50000000
beam = 13.0
latbeam = 8.0
acwt = 0.2
max_arcs = -1
skip_scoring = false
scoring_script = local/score.sh
scoring_opts = "--min-lmwt 1 --max-lmwt 10"
norm_vars = False
```    

The decoding section reports parameters about decoding, i.e. the steps that allows one to pass from a sequence of the context-dependent probabilities provided by the DNN into a sequence of words. The field *decoding_script_folder* specifies the folder where the decoding script is stored. The *decoding script* field is the script used for decoding (e.g., *decode_dnn.sh*) that should be in the *decoding_script_folder* specified before.  The field *decoding_proto* reports all the parameters needed for the considered decoding script. 

To make the code more flexible, the config parameters can also be specified within the command line. For example, you can run:
```    
 python run_exp.py quick_test/example_newcode.cfg --optimization,lr=0.01 --batches,batch_size=4
```    
The script will replace the learning rate in the specified cfg file with the specified lr value. The modified config file is then stored into *out_folder/config.cfg*.

The script *run_exp.py* automatically creates chunk-specific config files, that are used by the *run_nn* function to perform a single chunk training. The structure of chunk-specific cfg files is very similar to that of the global one. The main difference is a field *to_do={train, valid, forward}* that specifies the type of processing to on the features chunk specified in the field *fea*.

*Why proto files?*
Different neural networks, optimization algorithms, and HMM decoders might depend on a different set of hyperparameters. To address this issue, our current solution is based on the definition of some prototype files (for global, chunk, architecture config files). In general, this approach allows a more transparent check of the fields specified into the global config file. Moreover, it allows users to easily add new parameters without changing any line of the python code.
For instance, to add a user-defined model, a new proto file (e.g., *user-model.prot*o) that specifies the hyperparameter must be written. Then, the user should only write a class  (e.g., user-model in *neural_networks.py*) that implements the architecture).

## [FAQs]
## How can I plug-in my model
The toolkit is designed to allow users to easily plug-in their own acoustic models. To add a customized neural model do the following steps:
1. Go into the proto folder and create a new proto file (e.g., *proto/myDNN.proto*). The proto file is used to specify the list of the hyperparameters of your model that will be later set into the configuration file. To have an idea about the information to add to your proto file, you can take a look into the *MLP.proto* file: 

```
[proto]
dnn_lay=str_list
dnn_drop=float_list(0.0,1.0)
dnn_use_laynorm_inp=bool
dnn_use_batchnorm_inp=bool
dnn_use_batchnorm=bool_list
dnn_use_laynorm=bool_list
dnn_act=str_list
```
2. The parameter *dnn_lay* must be a list of string, *dnn_drop* (i.e., the dropout factors for each layer) is a list of float ranging from 0.0 and 1.0, *dnn_use_laynorm_inp* and *dnn_use_batchnorm_inp* are booleans that enable or disable batch or layer normalization of the input.  *dnn_use_batchnorm* and *dnn_use_laynorm* are a list of boolean that decide layer by layer if batch/layer normalization has to be used. 
The parameter *dnn_act* is again a list of string that sets the activation function of each layer. Since every model is based on its own set of hyperparameters,  different models have a different prototype file. For instance, you can take a look into *GRU.proto* and see that the hyperparameter list is different from that of a standard MLP.  Similarly to the previous examples, you should add here your list of hyperparameters and save the file.

3. Write a PyTorch class implementing your model.
 Open the library *neural_networks.py* and look at some of the models already implemented. For simplicity, you can start taking a look into the class MLP.  The classes have two mandatory methods: **init** and **forward**. The first one is used to initialize the architecture, the second specifies the list of computations to do. 
The method *init* takes in input two variables that are automatically computed within the *run_nn* function.  **inp_dim** is simply the dimensionality of the neural network input, while **options** is a dictionary containing all the parameters specified into the section *architecture* of the configuration file.  
For instance, you can access to the DNN activations of the various layers  in this way: 
```options['dnn_lay'].split(',')```. 
As you might see from the MLP class, the initialization method defines and initializes all the parameters of the neural network. The forward method takes in input a tensor **x** (i.e., the input data) and outputs another vector containing x. 
If your model is a sequence model (i.e., if there is at least one architecture with arch_seq_model=true in the cfg file), x is a tensor with (time_steps, batches, N_in), otherwise is a (batches, N_in) matrix. The class **forward** defines the list of computations to transform the input tensor into a corresponding output tensor. The output must have the sequential format (time_steps, batches, N_out) for recurrent models and the non-sequential format (batches, N_out) for feed-forward models.
Similarly to the already-implemented models the user should write a new class (e.g., myDNN) that implements the customized model:
```
class myDNN(nn.Module):
    
    def __init__(self, options,inp_dim):
        super(myDNN, self).__init__()
             // initialize the parameters

            def forward(self, x):
                 // do some computations out=f(x)
                  return out
```

4. Create a configuration file.
Now that you have defined your model and the list of its hyperparameters, you can create a configuration file. To create your own configuration file, you can take a look into an already existing config file (e.g., for simplicity you can consider *cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg*). After defining the adopted datasets with their related features and labels, the configuration file has some sections called *[architecture\*]*. Each architecture implements a different neural network. In *cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg* we only have *[architecture1]* since the acoustic model is composed of a single neural network. To add your own neural network, you have to write an architecture section (e.g., *[architecture1]*) in the following way:
```
[architecture1]
arch_name= mynetwork (this is a name you would like to use to refer to this architecture within the following model section)
arch_proto=proto/myDNN.proto (here is the name of the proto file defined before)
arch_library=neural_networks (this is the name of the library where myDNN is implemented)
arch_class=myDNN (This must be the name of the  class you have implemented)
arch_pretrain_file=none (With this you can specify if you want to pre-train your model)
arch_freeze=False (set False if you want to update the parameters of your model)
arch_seq_model=False (set False for feed-forward models, True for recurrent models)
```
Then, you have to specify proper values for all the hyperparameters specified in *proto/myDNN.proto*. For the *MLP.proto*, we have:
```
dnn_lay=1024,1024,1024,1024,1024,N_out_lab_cd
dnn_drop=0.15,0.15,0.15,0.15,0.15,0.0
dnn_use_laynorm_inp=False
dnn_use_batchnorm_inp=False
dnn_use_batchnorm=True,True,True,True,True,False
dnn_use_laynorm=False,False,False,False,False,False
dnn_act=relu,relu,relu,relu,relu,softmax
```
Then, add the following parameters related to the optimization of your own architecture. You can use here standard sdg, adam, or rmsprop (see cfg/TIMIT_baselines/TIMIT_LSTM_mfcc.cfg for an example with rmsprop):
```
arch_lr=0.08
arch_halving_factor=0.5
arch_improvement_threshold=0.001
arch_opt=sgd
opt_momentum=0.0
opt_weight_decay=0.0
opt_dampening=0.0
opt_nesterov=False
```

5. Save the configuration file into the cfg folder (e.g, *cfg/myDNN_exp.cfg*).

6. Run the experiment with: 
```
python run_exp.sh cfg/myDNN_exp.cfg
```

7. To debug the model you can first take a look at the standard output. The config file is automatically parsed by the *run_exp.sh* and it raises errors in case of possible problems. You can also take a look into the *log.log* file to see additional information on the possible errors. 


When implementing a new model, an important debug test consists of doing an overfitting experiment (to make sure that the model is able to overfit a tiny dataset). If the model is not able to overfit, it means that there is a major bug to solve.

8. Hyperparameter tuning.
In deep learning, it is often important to play with the hyperparameters to find the proper setting for your model. This activity is usually very computational and time-consuming but is often necessary when introducing new architectures. To help hyperparameter tuning, we developed a utility that implements a random search of the hyperparameters (see next section for more details).


## How can I tune the hyperparameters
A hyperparameter tuning is often needed in deep learning to search for proper neural architectures. To help tuning the hyperparameters within PyTorch-Kaldi, we have implemented a simple utility that implements a random search. In particular, the script  *tune_hyperparameters.py* generates a set of random configuration files and can be run in this way:
```
python tune_hyperparameters.py cfg/TIMIT_MLP_mfcc.cfg exp/TIMIT_MLP_mfcc_tuning 10 arch_lr=randfloat(0.001,0.01) batch_size_train=randint(32,256) dnn_act=choose_str{relu,relu,relu,relu,softmax|tanh,tanh,tanh,tanh,softmax}
```
The first parameter is the reference cfg file that we would like to modify, while the second one is the folder where the random configuration files are saved. The third parameter is the number of the random config file that we would like to generate. There is then the list of all the hyperparameters that we want to change. For instance, *arch_lr=randfloat(0.001,0.01)* will replace the field *arch_lr* with a random float ranging from 0.001 to 0.01. *batch_size_train=randint(32,256)* will replace *batch_size_train* with a random integer between 32 and 256 and so on.
Once the config files are created, they can be run sequentially or in parallel with:
```
python run_exp.py $cfg_file
```

## How can I use my own dataset
PyTorch-Kaldi can be used with any speech dataset. To use your own dataset, the steps to take are similar to those discussed in the TIMIT/Librispeech tutorials. In general, what you have to do is the following:
1. Run the Kaldi recipe with your dataset. Please, see the Kaldi website to have more information on how to perform data preparation.
2. Compute the alignments on training, validation, and test data.
3. Write a PyTorch-Kaldi config file *$cfg_file*.
4. Run the config file with ```python run_exp.sh $cfg_file```.

## How can I plug-in my own features
The current version of PyTorch-Kaldi supports input features stored with the Kaldi ark format. If the user wants to perform experiments with customized features, the latter must be converted into the ark format. Take a look into the Kaldi-io-for-python git repository (https://github.com/vesis84/kaldi-io-for-python) for a detailed description about converting numpy arrays into ark files. 
Moreover, you can take a look into our utility called save_raw_fea.py. This script generates Kaldi ark files containing raw features, that are later used to train neural networks fed by the raw waveform directly (see the section about processing audio with SincNet).

## How can I transcript my own audio files
The current version of Pytorch-Kaldi supports the standard production process of using a Pytorch-Kaldi pre-trained acoustic model to transcript one or multiples .wav files. It is important to understand that you must have a trained Pytorch-Kaldi model. While you don't need labels or alignments anymore, Pytorch-Kaldi still needs many files to transcript a new audio file:
1. The features and features list *feats.scp* (with .ark files, see #how-can-i-plug-my-own-features)
2. The decoding graph (usually created with mkgraph.sh during previous model training such as triphones models). This graph is not needed if you're not decoding.

Once you have all these files, you can start adding your dataset section to the global configuration file. The easiest way is to copy the *cfg* file used to train your acoustic model and just modify by adding a new *[dataset]*:
```
[dataset4]
data_name = myWavFile
fea = fea_name=fbank
  fea_lst=myWavFilePath/data/feats.scp
  fea_opts=apply-cmvn --utt2spk=ark:myWavFilePath/data//utt2spk  ark:myWavFilePath/cmvn_test.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
  cw_left=5
  cw_right=5

lab = lab_name=none
  lab_data_folder=myWavFilePath/data/
  lab_graph=myWavFilePath/exp/tri3/graph
n_chunks=1

[data_use]
train_with = TIMIT_tr
valid_with = TIMIT_dev
forward_with = myWavFile
```
The key string for your audio file transcription is *lab_name=none*. The *none* tag asks Pytorch-Kaldi to enter a *production* mode that only does the forward propagation and decoding without any labels. You don't need TIMIT_tr and TIMIT_dev to be on your production server since Pytorch-Kaldi will skip this information to directly go to the forward phase of the dataset given in the *forward_with* field. As you can see, the global *fea* field requires the exact same parameters than standard training or testing dataset, while the *lab* field only requires two parameters. Please, note that *lab_data_folder* is nothing more than the same path as *fea_lst*. Finally, you still need to specify the number of chunks you want to create to process this file (1 hour = 1 chunk).<br /> 
**WARNINGS** <br />
In your standard .cfg, you might have used keywords such as *N_out_lab_cd* that can not be used anymore. Indeed, in a production scenario, you don't want to have the training data on your machine. Therefore, all the *variables* that were on your .cfg file must be replaced by their true values. To replace all the *N_out_{mono,lab_cd}* you can take a look at the output of:
```
hmm-info /path/to/the/final.mdl/used/to/generate/the/training/ali
```
Then, if you normalize posteriors as (check in your .cfg Section forward):
```
normalize_posteriors = True
normalize_with_counts_from = lab_cd
```
You must replace *lab_cd* by:
```
normalize_posteriors = True
normalize_with_counts_from = /path/to/ali_train_pdf.counts
```
This normalization step is crucial for HMM-DNN speech recognition. DNNs, in fact, provide posterior probabilities, while HMMs are generative models that work with likelihoods. To derive the required likelihoods, one can simply divide the posteriors by the prior probabilities. To create this *ali_train_pdf.counts* file you can follow:
```
alidir=/path/to/the/exp/tri_ali (change it with your path to the exp with the ali)
num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')
labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" ali_train_pdf.counts
```
et voilà ! In a production scenario, you might need to transcript a huge number of audio files, and you don't want to create as much as needed .cfg file. In this extent, and after creating this initial production .cfg file (you can leave the path blank), you can call the run_exp.py script with specific arguments referring to your different.wav features:
```
python run_exp.py cfg/TIMIT_baselines/TIMIT_MLP_fbank_prod.cfg --dataset4,fea,0,fea_lst="myWavFilePath/data/feats.scp" --dataset4,lab,0,lab_data_folder="myWavFilePath/data/" --dataset4,lab,0,lab_graph="myWavFilePath/exp/tri3/graph/"
```

This command will internally alter the configuration file with your specified paths, and run and your defined features! Note that passing long arguments to the run_exp.py script requires a specific notation. *--dataset4* specifies the name of the created section, *fea* is the name of the higher level field, *fea_lst* or *lab_graph* are the name of the lowest level field you want to change. The *0* is here to indicate which lowest level field you want to alter, indeed some configuration files may contain multiple *lab_graph* per dataset! Therefore, *0* indicates the first occurrence, *1* the second ... Paths MUST be encapsulated by " " to be interpreted as full strings! Note that you need to alter the *data_name* and *forward_with* fields if you don't want different .wav files transcriptions to erase each other (decoding files are stored accordingly to the field*data_name*). ``` --dataset4,data_name=MyNewName --data_use,forward_with=MyNewName ```.

## Batch size, learning rate, and dropout scheduler
In order to give users more flexibility, the latest version of PyTorch-Kaldi supports scheduling of the batch size, max_seq_length_train, learning rate, and dropout factor.
This means that it is now possible to change these values during training. To support this feature, we implemented the following formalisms within the config files:
```
batch_size_train = 128*12 | 64*10 | 32*2
```
In this case, our batch size will be 128 for the first 12 epochs, 64 for the following 10 epochs, and 32 for the last two epochs. By default "*" means "for N times", while "|" is used to indicate a change of the batch size. Note that if the user simply sets ```batch_size_train = 128```, the batch size is kept fixed during all the training epochs by default.

A similar formalism can be used to perform learning rate scheduling:
```
arch_lr = 0.08*10|0.04*5|0.02*3|0.01*2|0.005*2|0.0025*2
```
In this case, if the user simply sets ```arch_lr = 0.08``` the learning rate is annealed with the *new-bob* procedure used in the previous version of the toolkit. In practice, we start from the specified learning rate and we multiply it by a halving factor every time that the improvement on the validation dataset is smaller than the threshold specified in the field *arch_improvement_threshold*. 

Also the dropout factor can now be changed during training with the following formalism:

```
dnn_drop = 0.15*12|0.20*12,0.15,0.15*10|0.20*14,0.15,0.0
```
With the line before we can set a different dropout rate for different layers and for different epochs. 
For instance, the first hidden layer will have a dropout rate of 0.15 for the first 12 epochs, and 0.20 for the other 12. The dropout factor of the second layer, instead, will remain constant to 0.15 over all the training.  The same formalism is used for all the layers. Note that "|" indicates a change in the dropout factor within the same layer, while "," indicates a different layer.

You can take a look here into a config file where batch sizes, learning rates, and dropout factors are changed here:
```
cfg/TIMIT_baselines/TIMIT_mfcc_basic_flex.cfg
```
or here:
```
cfg/TIMIT_baselines/TIMIT_liGRU_fmllr_lr_schedule.cfg
```




## How can I contribute to the project
The project is still in its initial phase and we invite all potential contributors to participate. We hope to build a community of developers larger enough to progressively maintain, improve, and expand the functionalities of our current toolkit.  For instance, it could be helpful to report any bug or any suggestion to improve the current version of the code. People can also contribute by adding additional neural models, that can eventually make richer the set of currently-implemented architectures.




## [EXTRA]
## Speech recognition from the raw waveform with SincNet

[Take a look into our video introduction to SincNet](https://www.youtube.com/watch?v=mXQBObRGUgk&feature=youtu.be)

SincNet is a convolutional neural network recently proposed to process raw audio waveforms. In particular, SincNet encourages the first layer to discover more meaningful filters by exploiting parametrized sinc functions. In contrast to standard CNNs, which learn all the elements of each filter, only low and high cutoff frequencies of band-pass filters are directly learned from data. This inductive bias offers a very compact way to derive a customized filter-bank front-end, that only depends on some parameters with a clear physical meaning.

For a more detailed description of the SincNet model, please refer to the following papers:

- *M. Ravanelli, Y. Bengio, "Speaker Recognition from raw waveform with SincNet", in Proc. of SLT 2018 [ArXiv](https://arxiv.org/abs/1808.00158)*

- *M. Ravanelli, Y.Bengio, "Interpretable Convolutional Filters with SincNet", in Proc. of NIPS@IRASL 2018 [ArXiv](https://arxiv.org/abs/1811.09725)*

To use this model for speech recognition on TIMIT, to the following steps:
1. Follows the steps described in the “TIMIT tutorial”.
2. Save the raw waveform into the Kaldi ark format. To do it, you can use the save_raw_fea.py utility in our repository. The script saves the input signals into a binary Kaldi archive, keeping the alignments with the pre-computed labels. You have to run it for all the data chunks (e.g., train, dev, test). It can also specify the length of the speech chunk (*sig_wlen=200 # ms*) composing each frame.
3. Open the *cfg/TIMIT_baselines/TIMIT_SincNet_raw.cfg*, change your paths, and run:
```    
python ./run_exp.sh cfg/TIMIT_baselines/TIMIT_SincNet_raw.cfg
```    

4. With this architecture, we have obtained a **PER(%)=17.1%**. A standard CNN fed the same features gives us a **PER(%)=18.%**. Please, see [here](https://bitbucket.org/mravanelli/pytorch-kaldi-exp-timit/src/master/) to take a look into our results. Our results on SincNet outperforms results obtained with MFCCs and FBANKs fed by standard feed-forward networks.

In the following table, we compare the result of SincNet with other feed-forward neural network:

| Model  | WER(\%) | 
| ------ | -----|
|  MLP -fbank  | 18.7 | 
|  MLP -mfcc  | 18.2 | 
|  CNN -raw  | 18.1 | 
|SincNet -raw | **17.2**  | 


## Joint training between speech enhancement and ASR
In this section, we show how to use PyTorch-Kaldi to jointly train a cascade between a speech enhancement and a speech recognition neural networks. The speech enhancement has the goal of improving the quality of the speech signal by minimizing the MSE between clean and noisy features. The enhanced features then feed another neural network that predicts context-dependent phone states.

In the following, we report a toy-task example based on a reverberated version of TIMIT, that is only intended to show how users should set the config file to train such a combination of neural networks. 
 Even though some implementation details (and the adopted datasets) are different, this tutorial is inspired by this paper:

- *M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Batch-normalized joint training for DNN-based distant speech recognition", in Proceedings of STL 2016 [arXiv](https://arxiv.org/abs/1703.08471)*


To run the system do the following steps:

1- Make sure you have the standard clean version of TIMIT available.

2- Run the *Kaldi s5* baseline of TIMIT. This step is necessary to compute the clean features (that will be the labels of the speech enhancement system) and the alignments (that will be the labels of the speech recognition system). We recommend running the full timit s5 recipe (including the DNN training).

3- The standard TIMIT recipe uses MFCCs features. In this tutorial, instead, we use FBANK features. To compute  FBANK features run the following script in *$KALDI_ROOT/egs/TIMIT/s5* :
```    
feadir=fbank

for x in train dev test; do
  steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_fbank/$x $feadir
  steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $feadir
done
```    
Note that we use 40 FBANKS here, while Kaldi uses by default 23 FBANKs. To compute 40-dimensional features go into "$KALDI_ROOT/egs/TIMIT/conf/fbank.conf" and change the number of considered output filters.


4- Go to [this external repository](https://github.com/mravanelli/pySpeechRev/blob/master/README.md) and follow the steps to generate a reverberated version of TIMIT starting from the clean one. Note that this is just a *toy task* that is only helpful to show how setting up a joint-training system.

5- Compute the FBANK features for the TIMIT_rev dataset. To do it, you can copy the scripts in *$KALDI_ROOT/egs/TIMIT/ into $KALDI_ROOT/egs/TIMIT_rev/*. Please, copy also the data folder. Note that the audio files in the TIMIT_rev folders are saved with the standard *WAV* format, while TIMIT is released with the *SPHERE* format. To bypass this issue, open the files *data/train/wav.scp*, *data/dev/wav.scp*, *data/test/wav.scp* and delete the part about *SPHERE* reading (e.g., */home/mirco/kaldi-trunk/tools/sph2pipe_v2.5/sph2pipe -f wav*). You also have to change the paths from the standard TIMIT to the reverberated one (e.g. replace /TIMIT/ with /TIMIT_rev/). Remind to remove the final pipeline symbol“ |”. Save the changes and run the computation of the fbank features in this way:

``` 
feadir=fbank

for x in train dev test; do
  steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_fbank/$x $feadir
  steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $feadir
done
```
Remember to change the $KALDI_ROOT/egs/TIMIT_rev/conf/fbank.conf file in order to compute 40 features rather than the 23 FBANKS of the default configuration.

6- Once features are computed, open the following config file: 

```
cfg/TIMIT_baselines/TIMIT_rev/TIMIT_joint_training_liGRU_fbank.cfg
``` 

Remember to change the paths according to where data are stored in your machine. As you can see, we consider two types of features. The *fbank_rev* features are computed from the TIMIT_rev dataset, while the *fbank_clean* features are derived from the standard TIMIT dataset and are used as targets for the speech enhancement neural network. 
As you can see in the *[model]* section of the config file, we have the cascade between networks doing speech enhancement and speech recognition. The speech recognition architecture jointly estimates both context-dependent and monophone targets (thus using the so-called monophone regularization). 
To run an experiment type the following command:
``` 
python run_exp.py  cfg/TIMIT_baselines/TIMIT_rev/TIMIT_joint_training_liGRU_fbank.cfg
``` 

7- Results
With this configuration file, you should obtain a **Phone Error Rate (PER)=28.1%**. Note that some oscillations around this performance are more than natural and are due to different initialization of the neural parameters.

You can take a closer look into our results [here](https://bitbucket.org/mravanelli/pytorch-kaldi-exp-timit/src/master/TIMIT_rev/TIMIT_joint_training_liGRU_fbank/)

## Distant Speech Recognition with DIRHA
In this tutorial, we use the DIRHA-English dataset to perform a distant speech recognition experiment. The DIRHA English Dataset is a **multi-microphone speech corpus** being developed under the EC project DIRHA. The corpus is composed of both real and simulated sequences recorded with 32 sample-synchronized microphones in a domestic environment. The database contains signals of different characteristics in terms of noise and reverberation making it suitable for various multi-microphone signal processing and distant speech recognition tasks. The part of the dataset currently released is composed of 6 native US speakers (3 Males, 3 Females) uttering 409 wall-street journal sentences.  The training data have been created using a realistic data contamination approach, that is based on contaminating the clean speech *wsj-5k* sentences with high-quality multi-microphone impulse responses measured in the targeted environment. For more details on this dataset, please refer to the following papers:

- *M. Ravanelli, L. Cristoforetti, R. Gretter, M. Pellin, A. Sosi, M. Omologo, "The DIRHA-English corpus and related tasks for distant-speech recognition in domestic environments", in Proceedings of ASRU 2015. [ArXiv](https://arxiv.org/abs/1710.02560)*

- *M. Ravanelli, P. Svaizer, M. Omologo, "Realistic Multi-Microphone Data Simulation for Distant Speech Recognition",  in Proceedings of Interspeech 2016. [ArXiv](https://arxiv.org/abs/1711.09470)*

In this tutorial, we use the aforementioned simulated data for training (using LA6 microphone), while test is performed using the real recordings (LA6).  This task is very realistic, but also very challenging. The speech signals are characterized by a reverberation time of about 0.7 seconds. Non-stationary domestic noises (such as vacuum cleaner, steps, phone rings, etc.) are also present in the real recordings.


Let’s start now with the practical tutorial.

1- If not available, [download the DIRHA dataset from the LDC website](https://catalog.ldc.upenn.edu/LDC2018S01). LDC releases the full dataset for a small fee.

2- [Go this external reposotory](https://github.com/SHINE-FBK/DIRHA_English_wsj). As reported in this repository, you have to generate the contaminated WSJ dataset with the provided MATLAB script. Then, you can run the proposed KALDI baseline to have features and labels ready for our pytorch-kaldi toolkit.

3- Open the following configuration file:
``` 
cfg/DIRHA_baselines/DIRHA_liGRU_fmllr.cfg
``` 
The latter configuration file implements a simple RNN model based on a Light Gated Recurrent Unit (Li-GRU). We used fMLLR as input features. Change the paths and run the following command:

``` 
python run_exp.py cfg/DIRHA_baselines/DIRHA_liGRU_fmllr.cfg
``` 

4- Results:
The aforementioned system should provide  **Word Error Rate (WER%)=23.2%**. 
You can find the results obtained by us [here](https://bitbucket.org/mravanelli/pytorch-kaldi-dirha-exp/). 

Using the other configuration files in the *cfg/DIRHA_baselines* folder you can perform experiments with different setups. With the provided configuration files you can obtain the following results:

| Model  | WER(\%) | 
| ------ | -----|
|  MLP  | 26.1 | 
|GRU| 25.3 | 
|Li-GRU| **23.8**  | 

## Training an autoencoder
The current version of the repository is mainly designed for speech recognition experiments. We are actively working a new version, which is much more flexible and can manage input/output different from Kaldi features/labels. Even with the current version, however, it is possible to implement other systems, such as an autoencoder.

An autoencoder is a neural network whose inputs and outputs are the same. The middle layer normally contains a bottleneck that forces our representations to compress the information of the input. In this tutorial, we provide a toy example based on the TIMIT dataset. For instance, see the following configuration file:
``` 
cfg/TIMIT_baselines/TIMIT_MLP_fbank_autoencoder.cfg
``` 
Our inputs are the standard 40-dimensional fbank coefficients that are gathered using a context windows of 11 frames (i.e., the total dimensionality of our input is 440). A feed-forward neural network (called MLP_encoder) encodes our features into a 100-dimensional representation. The decoder (called MLP_decoder) is fed by the learned representations and tries to reconstruct the output. The system is trained with **Mean Squared Error (MSE)** metric.
Note that in the [Model] section we added this line “err_final=cost_err(dec_out,lab_cd)” at the end. The current version of the model, in fact, by default needs that at least one label is specified (we will remove this limit in the next version). 

You can train the system running the following command:
``` 
python run_exp.py cfg/TIMIT_baselines/TIMIT_MLP_fbank_autoencoder.cfg
``` 
The results should look like this:

``` 
ep=000 tr=['TIMIT_tr'] loss=0.139 err=0.999 valid=TIMIT_dev loss=0.076 err=1.000 lr_architecture1=0.080000 lr_architecture2=0.080000 time(s)=41
ep=001 tr=['TIMIT_tr'] loss=0.098 err=0.999 valid=TIMIT_dev loss=0.062 err=1.000 lr_architecture1=0.080000 lr_architecture2=0.080000 time(s)=39
ep=002 tr=['TIMIT_tr'] loss=0.091 err=0.999 valid=TIMIT_dev loss=0.058 err=1.000 lr_architecture1=0.040000 lr_architecture2=0.040000 time(s)=39
ep=003 tr=['TIMIT_tr'] loss=0.088 err=0.999 valid=TIMIT_dev loss=0.056 err=1.000 lr_architecture1=0.020000 lr_architecture2=0.020000 time(s)=38
ep=004 tr=['TIMIT_tr'] loss=0.087 err=0.999 valid=TIMIT_dev loss=0.055 err=0.999 lr_architecture1=0.010000 lr_architecture2=0.010000 time(s)=39
ep=005 tr=['TIMIT_tr'] loss=0.086 err=0.999 valid=TIMIT_dev loss=0.054 err=1.000 lr_architecture1=0.005000 lr_architecture2=0.005000 time(s)=39
ep=006 tr=['TIMIT_tr'] loss=0.086 err=0.999 valid=TIMIT_dev loss=0.054 err=1.000 lr_architecture1=0.002500 lr_architecture2=0.002500 time(s)=39
ep=007 tr=['TIMIT_tr'] loss=0.086 err=0.999 valid=TIMIT_dev loss=0.054 err=1.000 lr_architecture1=0.001250 lr_architecture2=0.001250 time(s)=39
ep=008 tr=['TIMIT_tr'] loss=0.086 err=0.999 valid=TIMIT_dev loss=0.054 err=0.999 lr_architecture1=0.000625 lr_architecture2=0.000625 time(s)=41
ep=009 tr=['TIMIT_tr'] loss=0.086 err=0.999 valid=TIMIT_dev loss=0.054 err=0.999 lr_architecture1=0.000313 lr_architecture2=0.000313 time(s)=38
```
You should only consider the field "loss=". The filed "err=" only contains not useuful information in this case (for the aforementioned reason).
You can take a look into the generated features typing the following command:

``` 
copy-feats ark:exp/TIMIT_MLP_fbank_autoencoder/exp_files/forward_TIMIT_test_ep009_ck00_enc_out.ark  ark,t:- | more
``` 

## References
[1] M. Ravanelli, T. Parcollet, Y. Bengio, "The PyTorch-Kaldi Speech Recognition Toolkit", [ArxIv](https://arxiv.org/abs/1811.07453)

[2] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Improving speech recognition by revising gated recurrent units", in Proceedings of Interspeech 2017. [ArXiv](https://arxiv.org/abs/1710.00641)

[3] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Light Gated Recurrent Units for Speech Recognition", in IEEE Transactions on Emerging Topics in Computational Intelligence. [ArXiv](https://arxiv.org/abs/1803.10225)

[4] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017. [ArXiv](https://arxiv.org/abs/1712.06086)

[5] T. Parcollet, M. Ravanelli, M. Morchid, G. Linarès, C. Trabelsi, R. De Mori, Y. Bengio, "Quaternion Recurrent Neural Networks", in Proceedings of ICLR 2019 [ArXiv](https://arxiv.org/abs/1806.04418)

[6] T. Parcollet, M. Morchid, G. Linarès, R. De Mori, "Bidirectional Quaternion Long-Short Term Memory Recurrent Neural Networks for Speech Recognition", in Proceedings of ICASSP 2019 [ArXiv](https://arxiv.org/abs/1811.02566)



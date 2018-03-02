# pytorch-kaldi
pytorch-kaldi is a public repository for developing state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN part is managed by pytorch, while feature extraction, label computation, and decoding are performed with the kaldi toolkit.


## Introduction:
This project releases a collection of codes and utilities to develop state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN/RNN part is implemented in pytorch, while feature extraction, alignments, and decoding are performed with the Kaldi toolkit.  The current version of the provided system has the following features:
- Supports different types of NNs (e.g., *MLP*, *RNN*, *LSTM*, *GRU*, *Minimal GRU*, *Light GRU*) [1,2,3]
- Supports  recurrent dropout
- Supports  batch and layer normalization
- Supports unidirectional/bidirectional RNNs
- Supports  residual/skip connections
- Supports  twin regularization [4]
- python2/python3 compatibility
- multi-gpu training
- recovery/saving checkpoints
- easy interface with kaldi.

The provided solution is designed for large-scale speech recognition experiments on both standard machines and HPC clusters. 

## Prerequisites:
- Linux is required (we tested our release on Ubuntu 17.04 and various versions of Debian).

- We recommend to run the codes on a GPU machine. Make sure that the cuda libraries (https://developer.nvidia.com/cuda-downloads) are installed and correctly working. We tested our system on cuda 9.0, 9.1 and 8.0.
Make sure that python is installed (the code is tested with python 2.7 and python 3.6). Even though not mandatory, we suggest to use Anaconda (https://anaconda.org/anaconda/python).

- If not already done, install pytorch (http://pytorch.org/). We tested our codes on pytorch 0.3.0 and pytorch 0.3.1. Older version of pytorch are likely to rise errors. To check your installation, type “python” and, once entered into the console, type “import torch”. Make sure everything is fine.

- If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of the Kaldi binaries into $HOME/.bashrc. As a first test to check the installation, open a bash shell, type “copy-feats” and make sure no errors appear.

- Install *kaldi-io* package from the kaldi-io-for-python project (https://github.com/vesis84/kaldi-io-for-python). It provides a simple interface between kaldi and python. To install it:
``` 
run git clone https://github.com/vesis84/kaldi-io-for-python.git
add PYTHONPATH=${PYTHONPATH}: to $HOME/.bashrc
``` 
Type *import kaldi_io* into the python console and make sure the package is correctly imported. You can find more info (including some reading and writing tests) on https://github.com/vesis84/kaldi-io-for-python.

- The implementation of the RNN models sorts the training sentences according to their length. This allows the system to minimize the need of zero padding when forming minibatches. The duration of each sentence is extracted using *sox*. Please, make sure it is installed (it is only used when generating the feature lists in *create_chunk.sh*)


## How to run a TIMIT experiment:
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

1. Run the Kaldi s5 baseline of TIMIT.
This step is necessary to compute features and labels later used to train the pytorch MLP. In particular:
- go to *$KALDI_ROOT/egs/timit/s5* and run the script *run.sh*. 
- Make sure everything works fine. 
- Please, also run the Karel’s DNN baseline using *local/nnet/run_dnn.sh*. 
- Do not forget to compute the alignments for test and dev data with the following commands.
If you wanna use tri3 alignments, type:
``` 
steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
``` 

If you wanna use dnn alignments (as suggested), type:
``` 
steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
``` 

2. Split the feature lists into chunks.
- Go to the *pytorch-kaldi* folder. 
- The script *create_chunks.sh* first shuffles or sorts (according to the sentence length) a kaldi feature list and then split it into a certain number of chunks. Shuffling a list could be good for feed-forward DNNs, while a sorted list can be useful for RNNs (for minimizing the need of zero-padding when forming minibatches). The code also computes per-speaker and per-sentence CMVN.

For shuffled mfcc features run:
``` 
./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/train mfcc_shu 5 train 0
./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/dev mfcc_shu 1 dev 0
./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/test mfcc_shu 1 test 0
``` 

For ordered mfcc features run:
``` 
./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/train mfcc_ord 5 train 1
./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/dev mfcc_ord 1 dev 1
./create_chunks.sh $KALDI_ROOT/egs/timit/s5/data/test mfcc_ord 1 test 1
``` 



3. Setup the Config file.
- Go into the *cfg* folder
- open a config file (e.g,*TIMIT_MLP.cfg*,*TIMIT_GRU.cfg*) and modify it according to your paths:
- *tr_fea_scp* contains the list of features created with create_chunks.sh.
- *tr_fea_opts* allows users to easily add normalizations, derivatives and other types of feature processing.
- *tr_lab_folder* is the kaldi folder containing the alignments (labels).
- *tr_lab_opts* allows users to derive context-dependent phone targets (when set to *ali-to-pdf*) or monophone targets (when set to *ali-to-phones --per-frame*).
- Please, modify the paths for dev and test data as well.

- Feel free to modify the DNN architecture and the other optimization parameters according to your needs.
The required *count_file* (used to normalize the DNN posteriors before feeding the decoder and automatically created by kaldi when running s5 recipe) can be found here:

``` 
$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts
``` 

- Use the option *use_cuda=1* for running the code on a GPU (strongly suggested).
- Use the option *save_gpumem=0* to save gpu memory. The code will be a little bit slower (about 10-15%), but it saves gpu memory.
- See *config_description.cfg* for a more detailed description of all the possible options.

4. Run the experiment.
- Type the following command to run DNN training:
``` 
./run_exp.sh cfg/baselines/TIMIT_MLP.cfg > log.log 
``` 
or 
``` 
./run_exp.sh cfg/baselines/TIMIT_GRU.cfg > log.log 
``` 

Note that *run_exp.sh* is a bash script that performs a full ASR experiment (training, forward, and decoding steps).
If everything works fine, you should find the following files into the output folder:
- a file *res.res* summarizing the training and eval performance over the various epochs. 
Take a look to *exp/our_results* for taking a look to the results you should obtaine where running the code.

- a folder *decode_test* containing the speech recognition results. If you type *./RESULTS* you should be able to see the Phone Error Rate (PER%) for each experiment. 
- the model *.pkl* is the final model used for speech decoding.
- the files *.info* report loss and error performance for each training chunk.
- the file *log.log* contains possible errors occurred in the training procedure.



## TIMIT Results:

The results reported in each cell of the  table are the average *PER%* performance obtained  on the test set  after running five ASR experiments with different initialization seeds. We believe that averaging the performance obtained with different initialization seeds is crucial  for TIMIT, since the natural performance variability might completely hide the experimental evidence.  

The main hyperparameters of the models (i.e., learning rate, number of layers, number of hidden neurons, dropout factor) have been optimized through a grid search performed on the development set (see the config files in *cfg/baselines* for an overview on the hyperparameters adopted for each NN). 

The RNN models are bidirectional, use recurrent dropout, and batch normalization is applied to feedforward connections. 

| Model  | mfcc | fbank | fMLLR | 
| ------ | -----| ------| ------| 
|  Kaldi DNN Baseline | -----| ------| 18.5 |
|  MLP  | -----| ------| ------| 
|reluRNN| -----| ------| ------| 
|LSTM| -----| ------|--- | 
|GRU| 16.0 ± 0.13| ------|  15.3 ± 0.32| 
|M-GRU| 16.1  ± 0.28| ------|  15.2 ± 0.23| 
|li-GRU| 15.5  ± 0.33| ------|  **14.6** ± 0.32| 


The RNN architectures are significantly better than the MLP one. In particular, the Li-GRU model (see [1,2] for more details) performs slightly better that the other models. As expected fMLLR features lead to the best performance. The performance of  *14.6%* obtained with our best fMLLR system is, to the best of our knowledge, one of the best results so far achieved with the TIMIT dataset.

For comparison and reference purposes,  you can find the output results obtained by us in the folders  *exp/our_results/TIMIT_{MLP,RNN,LSTM,GRU,M_GRU,liGRU}*. 


## Brief Overview of the Architecture

The main script to run experiments is *run_exp.sh*.  The only parameter that it takes in input is the configuration file, which contains a full description of the data, architecture, optimization and decoding step. The user can use the variable *$cmd* for submitting jobs on HPC clusters.

Each training epoch is divided into many chunks.  The pytorch code *run_nn_single_ep.py* performs training over a single chunk and provides in output a model file in *.pkl* format and a *.info* file (that contains various information such as the current training loss and error). 

After each training epoch, the performance on the dev-set is monitored. If the relative performance improvement  is below a given threshold, the learning rate is decreased by an halving factor. The training loop is iterated for the specified number of training epochs. When training is finished, a forward step is carried on for generating the set of posterior probabilities that will be processed by the kaldi decoder. 

After decoding, the final transcriptions and scores are available in the output folder. If, for some reason, the training procedure is interrupted the process can be resumed starting from the last processed chunk.



## Adding customized DNN models
One can easily write its own customized DNN model and plugs it into neural_nets.py. Similarly to the models already implemented, the user has to write a *init* method for initializing the DNN parameters and a forward method. The forward method should take in input the current features *x* and the corresponding labels *lab*. It has to provide at the output the loss, the error and the posterior probabilities of the processed minibatch.  Once the customized DNN has been created, the new model should be imported into the *run_nn_single_ep.py* file in this way:

``` 
from neural_nets import mydnn as ann
``` 

It is also important to properly set the label *rnn=1* if the model is a RNN model and *rnn=0* if it is a feedforward DNNs. Note that RNN and feed-forward models are based on different feature processing (for RNN models  the features are ordered according to their length, for feed-forward DNNs the features are shuffled.)


## References

[1] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Improving speech recognition by revising gated recurrent units", in Proceedings of Interspeech 2017. [ArXiv](https://arxiv.org/abs/1710.00641)

[2] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Light Gated Recurrent Units for Speech Recognition", in IEEE Transactions on
Emerging Topics in Computational Intelligence (to appear).

[3] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017. [ArXiv](https://arxiv.org/abs/1712.06086)

[4] D. Serdyuk, R. Nan Ke, A. Sordoni, A. Trischler, C. Pal, Y. Bengio, "Twin Networks: Matching the Future for Sequence Generation", ICLR 2018 [ArXiv](https://arxiv.org/pdf/1708.06742.pdf)
  




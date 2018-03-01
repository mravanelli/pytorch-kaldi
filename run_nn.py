#!/usr/bin/env python

# pytorch_speechMLP 
# Mirco Ravanelli 
# Montreal Institute for Learning Algoritms (MILA)
# University of Montreal 

# January 2018

# Description: 
# This code implements with pytorch a basic MLP  for speech recognition. 
# It exploits an interface to  kaldi for feature computation and decoding. 
# How to run it:
# python MLP_speech.py --cfg TIMIT_MLP.cfg

# TO DO:
# - scrivi codice semplificato per epoca singola
# - fai test approfonditi per valutare consistenza risultati
# - verifica prestazioni (helios,azure)




import kaldi_io
import numpy as np
import torch
from torch.autograd import Variable
import timeit
import torch.optim as optim
import os
from data_io import load_chunk,load_counts,read_conf
import random
import torch.nn as nn
import sys


  
    
# Reading options in cfg file
options=read_conf()

# to do options
do_training=bool(int(options.do_training))
do_eval=bool(int(options.do_eval))
do_forward=bool(int(options.do_forward))

# Reading data options
fea_scp=options.fea_scp
fea_opts=options.fea_opts
lab_folder=options.lab_folder
lab_opts=options.lab_opts

out_file=options.out_file

# Reading count file from kaldi
count_file=options.count_file
pt_file=options.pt_file

# reading architectural options
left=int(options.cw_left)
right=int(options.cw_right)
seed=int(options.seed)
use_cuda=bool(int(options.use_cuda))
multi_gpu=bool(int(options.multi_gpu))
NN_type=options.NN_type

# reading optimization options
batch_size=int(options.batch_size)
lr=float(options.lr)
save_gpumem=int(options.save_gpumem)
opt=options.optimizer


if NN_type=='RNN':
   from neural_nets import RNN as ann
   rnn=1

if NN_type=='LSTM':
   from neural_nets import LSTM as ann
   rnn=1
   
if NN_type=='GRU':
  from neural_nets import GRU as ann
  rnn=1
if NN_type=='MLP':
   from neural_nets import MLP as ann
   rnn=0


start_time=timeit.default_timer()

# Setting torch seed
torch.manual_seed(seed)
random.seed(seed)

if rnn or do_eval or do_forward:
   seed=-1
   
[data_name,data_set,data_end_index]=load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,left,right,seed)

if not(save_gpumem):
   data_set=torch.from_numpy(data_set).float().cuda()
else:
   data_set=torch.from_numpy(data_set).float()  

# Model initialization
N_fea=data_set.shape[1]-1
options.input_dim=N_fea
N_out=int(data_set[:,N_fea].max()-data_set[:,N_fea].min()+1) 
options.num_classes=N_out
    
net = ann(options)

# multi gpu data parallelization
if multi_gpu:
 net = nn.DataParallel(net)
      
# Loading model into the cuda device
if use_cuda:
  net.cuda() 
       
# Optimizer initialization
if opt=='sgd':
  optimizer = optim.SGD(net.parameters(), lr=lr) 
else:
  optimizer = optim.RMSprop(net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
     
if pt_file!='none':
   checkpoint_load = torch.load(pt_file)
   net.load_state_dict(checkpoint_load['model_par'])
   optimizer.load_state_dict(checkpoint_load['optimizer_par'])
   optimizer.param_groups[0]['lr']=lr


N_snt=len(data_name)


if do_training:
  net.train()
  test_flag=0   
  N_batches=int(N_snt/batch_size)
  if rnn==0:
   N_ex_tr=data_set.shape[0]
   N_batches=int(N_ex_tr/batch_size)
   
if do_eval:
 N_batches=N_snt
 net.eval()
 test_flag=1
 batch_size=1
 
 if do_forward:
  post_file=kaldi_io.open_or_fd(out_file,'wb')
  counts = load_counts(count_file)
  

beg_batch=0
end_batch=batch_size   

snt_index=0
beg_snt=0 

loss_sum=0
err_sum=0


  
for i in range(N_batches):
   
   if do_training:
    
    if rnn==1:
     max_len=data_end_index[snt_index+batch_size-1]-data_end_index[snt_index+batch_size-2]
   
     inp= Variable(torch.zeros(max_len,batch_size,N_fea)).contiguous()
     lab= Variable(torch.zeros(max_len,batch_size)).contiguous().long()
   
   
     for k in range(batch_size):
      snt_len=data_end_index[snt_index]-beg_snt
      N_zeros=max_len-snt_len
      # Appending a random number of initial zeros, tge others are at the end. 
      N_zeros_left=random.randint(0,N_zeros)
      # randomizing could have a regularization effect
      inp[N_zeros_left:N_zeros_left+snt_len,k,:]=data_set[beg_snt:beg_snt+snt_len,0:N_fea] 
      lab[N_zeros_left:N_zeros_left+snt_len,k]=data_set[beg_snt:beg_snt+snt_len,-1]
      
      beg_snt=data_end_index[snt_index]
      snt_index=snt_index+1
   
    else: # MLP case
     # features and labels for batch i
     inp= Variable(data_set[beg_batch:end_batch,0:N_fea]).contiguous()
     lab= Variable(data_set[beg_batch:end_batch,N_fea]).contiguous().long()
    
    
   if do_eval:
      end_snt=data_end_index[i]
      inp= Variable(data_set[beg_snt:end_snt,0:N_fea],volatile=True).contiguous()
      lab= Variable(data_set[beg_snt:end_snt,N_fea],volatile=True).contiguous().long()
      if rnn==1:
        inp=inp.view(inp.shape[0],1,inp.shape[1])
        lab=lab.view(lab.shape[0],1)
      beg_snt=data_end_index[i]
    
   
   [loss,err,pout] = net(inp,lab,test_flag)

   if do_forward:
    if rnn==1:
       pout=pout.view(pout.shape[0]*pout.shape[1],pout.shape[2]) 
    kaldi_io.write_mat(post_file, pout.data.cpu().numpy()-np.log(counts/np.sum(counts)), data_name[i])
    
   if do_training:
       
    # free the gradient buffer
    optimizer.zero_grad()  
  
    # Gradient computation
    loss.backward()
    
    # Gradient clipping
    #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
  
    # updating parameters
    optimizer.step()
   
   # Loss accumulation 
   loss_sum=loss_sum+loss.data
   err_sum=err_sum+err.data
   
   # update it to the next batch 
   beg_batch=end_batch
   end_batch=beg_batch+batch_size

 
loss_tot=loss_sum/N_batches
err_tot=err_sum/N_batches
 
end_time=timeit.default_timer() 

 
 # check point saving
if do_training:
 checkpoint={'model_par': net.state_dict(),
            'optimizer_par' : optimizer.state_dict()}

 torch.save(checkpoint,options.out_file)

info_file=out_file.replace(".pkl",".info")

# Printing info file
with open(info_file, "w") as inf:
   inf.write("model_in=%s\n" %(pt_file))
   inf.write("fea_in=%s\n" %(fea_scp))
   inf.write("loss=%f\n" %(loss_tot))
   inf.write("err=%f\n" %(err_tot))
   inf.write("elapsed_time=%f\n" %(end_time-start_time))
   
inf.close()

if do_forward:
    post_file.close()

 

 


  
  
  
  
  
 

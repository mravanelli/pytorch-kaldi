##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import sys
import configparser
import os
from utils import is_sequential_dict,model_init,optimizer_init,forward_model,progress
from data_io import load_counts
import numpy as np
import random
import torch
from distutils.util import strtobool
import time
import threading

from data_io import read_lab_fea,open_or_fd,write_mat
from utils import shift

def run_nn(data_name,data_set,data_end_index,fea_dict,lab_dict,arch_dict,cfg_file,processed_first,next_config_file):
    
    # This function processes the current chunk using the information in cfg_file. In parallel, the next chunk is load into the CPU memory
    
    # Reading chunk-specific cfg file (first argument-mandatory file) 
    if not(os.path.exists(cfg_file)):
         sys.stderr.write('ERROR: The config file %s does not exist!\n'%(cfg_file))
         sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config.read(cfg_file)
    
    # Setting torch seed
    seed=int(config['exp']['seed'])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
    # Reading config parameters
    output_folder=config['exp']['out_folder']
    use_cuda=strtobool(config['exp']['use_cuda'])
    multi_gpu=strtobool(config['exp']['multi_gpu'])
    
    to_do=config['exp']['to_do']
    info_file=config['exp']['out_info']
    
    model=config['model']['model'].split('\n')
    
    forward_outs=config['forward']['forward_out'].split(',')
    forward_normalize_post=list(map(strtobool,config['forward']['normalize_posteriors'].split(',')))
    forward_count_files=config['forward']['normalize_with_counts_from'].split(',')
    require_decodings=list(map(strtobool,config['forward']['require_decoding'].split(',')))
    
    use_cuda=strtobool(config['exp']['use_cuda'])
    save_gpumem=strtobool(config['exp']['save_gpumem'])
    is_production=strtobool(config['exp']['production'])

    if to_do=='train':
        batch_size=int(config['batches']['batch_size_train'])
    
    if to_do=='valid':
        batch_size=int(config['batches']['batch_size_valid'])
    
    if to_do=='forward':
        batch_size=1
        
    
    # ***** Reading the Data********
    if processed_first:
        
        # Reading all the features and labels for this chunk
        shared_list=[]
        
        p=threading.Thread(target=read_lab_fea, args=(cfg_file,is_production,shared_list,output_folder,))
        p.start()
        p.join()
        
        data_name=shared_list[0]
        data_end_index=shared_list[1]
        fea_dict=shared_list[2]
        lab_dict=shared_list[3]
        arch_dict=shared_list[4]
        data_set=shared_list[5]


        
        # converting numpy tensors into pytorch tensors and put them on GPUs if specified
        if not(save_gpumem) and use_cuda:
           data_set=torch.from_numpy(data_set).float().cuda()
        else:
           data_set=torch.from_numpy(data_set).float()
                           
    # Reading all the features and labels for the next chunk
    shared_list=[]
    p=threading.Thread(target=read_lab_fea, args=(next_config_file,is_production,shared_list,output_folder,))
    p.start()
    
    # Reading model and initialize networks
    inp_out_dict=fea_dict
    
    [nns,costs]=model_init(inp_out_dict,model,config,arch_dict,use_cuda,multi_gpu,to_do)
       
    # optimizers initialization
    optimizers=optimizer_init(nns,config,arch_dict)
           
    
    # pre-training
    for net in nns.keys():
      pt_file_arch=config[arch_dict[net][0]]['arch_pretrain_file']
      
      if pt_file_arch!='none':        
          checkpoint_load = torch.load(pt_file_arch)
          nns[net].load_state_dict(checkpoint_load['model_par'])
          optimizers[net].load_state_dict(checkpoint_load['optimizer_par'])
          optimizers[net].param_groups[0]['lr']=float(config[arch_dict[net][0]]['arch_lr']) # loading lr of the cfg file for pt
          
    
    
    
    if to_do=='forward':
        
        post_file={}
        for out_id in range(len(forward_outs)):
            if require_decodings[out_id]:
                out_file=info_file.replace('.info','_'+forward_outs[out_id]+'_to_decode.ark')
            else:
                out_file=info_file.replace('.info','_'+forward_outs[out_id]+'.ark')
            post_file[forward_outs[out_id]]=open_or_fd(out_file,output_folder,'wb')


    # check automatically if the model is sequential
    seq_model=is_sequential_dict(config,arch_dict)
    
    # ***** Minibatch Processing loop********
    if seq_model or to_do=='forward':
        N_snt=len(data_name)
        N_batches=int(N_snt/batch_size)
    else:
        N_ex_tr=data_set.shape[0]
        N_batches=int(N_ex_tr/batch_size)
        
    
    beg_batch=0
    end_batch=batch_size 
    
    snt_index=0
    beg_snt=0 
    

    start_time = time.time()
    
    # array of sentence lengths
    arr_snt_len=shift(shift(data_end_index, -1,0)-data_end_index,1,0)
    arr_snt_len[0]=data_end_index[0]
    
    
    loss_sum=0
    err_sum=0
    
    inp_dim=data_set.shape[1]
    for i in range(N_batches):   
        
        max_len=0
    
        if seq_model:
         
         max_len=int(max(arr_snt_len[snt_index:snt_index+batch_size]))  
         inp= torch.zeros(max_len,batch_size,inp_dim).contiguous()
    
            
         for k in range(batch_size):
              
                  snt_len=data_end_index[snt_index]-beg_snt
                  N_zeros=max_len-snt_len
                  
                  # Appending a random number of initial zeros, tge others are at the end. 
                  N_zeros_left=random.randint(0,N_zeros)
                 
                  # randomizing could have a regularization effect
                  inp[N_zeros_left:N_zeros_left+snt_len,k,:]=data_set[beg_snt:beg_snt+snt_len,:]
                  
                  beg_snt=data_end_index[snt_index]
                  snt_index=snt_index+1
                
        else:
            # features and labels for batch i
            if to_do!='forward':
                inp= data_set[beg_batch:end_batch,:].contiguous()
            else:
                snt_len=data_end_index[snt_index]-beg_snt
                inp= data_set[beg_snt:beg_snt+snt_len,:].contiguous()
                beg_snt=data_end_index[snt_index]
                snt_index=snt_index+1
    
        # use cuda
        if use_cuda:
            inp=inp.cuda()
    
        if to_do=='train':
            # Forward input, with autograd graph active
            outs_dict=forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs)
            
            for opt in optimizers.keys():
                optimizers[opt].zero_grad()
                
    
            outs_dict['loss_final'].backward()
            
            # Gradient Clipping (th 0.1)
            #for net in nns.keys():
            #    torch.nn.utils.clip_grad_norm_(nns[net].parameters(), 0.1)
            
            
            for opt in optimizers.keys():
                if not(strtobool(config[arch_dict[opt][0]]['arch_freeze'])):
                    optimizers[opt].step()
        else:
            with torch.no_grad(): # Forward input without autograd graph (save memory)
                outs_dict=forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs)
    
                    
        if to_do=='forward':
            for out_id in range(len(forward_outs)):
                
                out_save=outs_dict[forward_outs[out_id]].data.cpu().numpy()
                
                if forward_normalize_post[out_id]:
                    # read the config file
                    counts = load_counts(forward_count_files[out_id])
                    out_save=out_save-np.log(counts/np.sum(counts))             
                    
                # save the output    
                write_mat(output_folder,post_file[forward_outs[out_id]], out_save, data_name[i])
        else:
            loss_sum=loss_sum+outs_dict['loss_final'].detach()
            err_sum=err_sum+outs_dict['err_final'].detach()
           
        # update it to the next batch 
        beg_batch=end_batch
        end_batch=beg_batch+batch_size
        
        # Progress bar
        if to_do == 'train':
          status_string="Training | (Batch "+str(i+1)+"/"+str(N_batches)+")"+" | L:" +str(round(loss_sum.cpu().item()/(i+1),3))
          if i==N_batches-1:
             status_string="Training | (Batch "+str(i+1)+"/"+str(N_batches)+")"

             
        if to_do == 'valid':
          status_string="Validating | (Batch "+str(i+1)+"/"+str(N_batches)+")"
        if to_do == 'forward':
          status_string="Forwarding | (Batch "+str(i+1)+"/"+str(N_batches)+")"
          
        progress(i, N_batches, status=status_string)
    
    elapsed_time_chunk=time.time() - start_time 
    
    loss_tot=loss_sum/N_batches
    err_tot=err_sum/N_batches
    
    # clearing memory
    del inp, outs_dict, data_set
    
    # save the model
    if to_do=='train':
     
    
         for net in nns.keys():
             checkpoint={}
             checkpoint['model_par']=nns[net].state_dict()
             checkpoint['optimizer_par']=optimizers[net].state_dict()
             
             out_file=info_file.replace('.info','_'+arch_dict[net][0]+'.pkl')
             torch.save(checkpoint, out_file)
         
    if to_do=='forward':
        for out_name in forward_outs:
            post_file[out_name].close()
         
    
         
    # Write info file
    with open(info_file, "w") as text_file:
        text_file.write("[results]\n")
        if to_do!='forward':
            text_file.write("loss=%s\n" % loss_tot.cpu().numpy())
            text_file.write("err=%s\n" % err_tot.cpu().numpy())
        text_file.write("elapsed_time_chunk=%f\n" % elapsed_time_chunk)
    
    text_file.close()
    
    
    # Getting the data for the next chunk (read in parallel)    
    p.join()
    data_name=shared_list[0]
    data_end_index=shared_list[1]
    fea_dict=shared_list[2]
    lab_dict=shared_list[3]
    arch_dict=shared_list[4]
    data_set=shared_list[5]
    
    
    # converting numpy tensors into pytorch tensors and put them on GPUs if specified
    if not(save_gpumem) and use_cuda:
       data_set=torch.from_numpy(data_set).float().cuda()
    else:
       data_set=torch.from_numpy(data_set).float()
       
       
    return [data_name,data_set,data_end_index,fea_dict,lab_dict,arch_dict]

##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
#
# Description: This script generates kaldi ark files containing raw features. 
# The file list must be a file containing "snt_id file.wav". 
# Note that only wav files are supported here (sphere or other format are not supported)
##########################################################


import scipy.io.wavfile
import math
import numpy as np
import os
from data_io import read_vec_int_ark,write_mat


# Run it for all the data chunks (e.g., train, dev, test) => uncomment

lab_folder='/users/parcollet/KALDI/kaldi-trunk/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test'
lab_opts='ali-to-pdf'
out_folder='/users/parcollet/KALDI/kaldi-trunk/egs/timit/s5/data/raw_TIMIT_200ms/test'
wav_lst='/users/parcollet/KALDI/kaldi-trunk/egs/timit/s5/data/test/wav.lst'
scp_file_out='/users/parcollet/KALDI/kaldi-trunk/egs/timit/s5/data/raw_TIMIT_200ms/test/feats_raw.scp'

#lab_folder='quick_test/dnn4_pretrain-dbn_dnn_ali_dev'
#lab_opts='ali-to-pdf'
#out_folder='raw_TIMIT_200ms/dev'
#wav_lst='/home/mirco/pytorch-kaldi-new/quick_test/data/dev/wav_lst.scp'
#scp_file_out='quick_test/data/dev/feats_raw.scp'

#lab_folder='quick_test/dnn4_pretrain-dbn_dnn_ali_test'
#lab_opts='ali-to-pdf'
#out_folder='raw_TIMIT_200ms/test'
#wav_lst='/home/mirco/pytorch-kaldi-new/quick_test/data/test/wav_lst.scp'
#scp_file_out='quick_test/data/test/feats_raw.scp'



sig_fs=16000 # Hz
sig_wlen=200 # ms

lab_fs=16000  #Hz
lab_wlen=25   #ms
lab_wshift=10 #ms

sig_wlen_samp=int((sig_fs*sig_wlen)/1000)
lab_wlen_samp=int((lab_fs*lab_wlen)/1000)
lab_wshift_samp=int((lab_fs*lab_wshift)/1000)


# Create the output folder
try:
    os.stat(out_folder)
except:
    os.makedirs(out_folder)       


# Creare the scp file
scp_file = open(scp_file_out,"w") 

# reading the labels
lab= { k:v for k,v in read_vec_int_ark('gunzip -c '+lab_folder+'/ali*.gz | '+lab_opts+' '+lab_folder+'/final.mdl ark:- ark:-|', out_folder)} 

# reading the list file
with open(wav_lst) as f:
    sig_lst = f.readlines()

sig_lst = [x.strip() for x in sig_lst] 

for sig_file in sig_lst:
    sig_id=sig_file.split(' ')[0]
    sig_path=sig_file.split(' ')[1]
    [fs,signal]=scipy.io.wavfile.read(sig_path)
    signal=signal.astype(float)/32768
    signal=signal/np.max(np.abs(signal))
    
    cnt_fr=0
    beg_samp=0
    frame_all=[]
    
    while beg_samp+lab_wlen_samp<signal.shape[0]:
        sample_fr=np.zeros(sig_wlen_samp)
        central_sample_lab=int(((beg_samp+lab_wlen_samp/2)-1))
        central_fr_index=int(((sig_wlen_samp/2)-1))
        
        beg_signal_fr=int(central_sample_lab-(sig_wlen_samp/2))
        end_signal_fr=int(central_sample_lab+(sig_wlen_samp/2))
        
        if beg_signal_fr>=0 and end_signal_fr<=signal.shape[0]:
            sample_fr=signal[beg_signal_fr:end_signal_fr]
        else:
            if beg_signal_fr<0:
                n_left_samples=central_sample_lab
                sample_fr[central_fr_index-n_left_samples+1:]=signal[0:end_signal_fr]
            if end_signal_fr>signal.shape[0]:
                n_right_samples=signal.shape[0]-central_sample_lab
                sample_fr[0:central_fr_index+n_right_samples+1]=signal[beg_signal_fr:]

        frame_all.append(sample_fr)
        cnt_fr=cnt_fr+1
        beg_samp=beg_samp+lab_wshift_samp
        
    frame_all=np.asarray(frame_all)
    
    # Save the matrix into a kaldi ark
    out_file=out_folder+'/'+sig_id+'.ark'
    write_mat(out_folder, out_file, frame_all, key=sig_id)
    print(sig_id)
    scp_file.write(sig_id+' '+out_folder+'/'+sig_id+'.ark:'+str(len(sig_id)+1)+'\n')

        
    N_fr_comp=1 + math.floor((signal.shape[0] - 400) / 160)    
    #print("%s %i %i "%(lab[sig_id].shape[0],N_fr_comp,cnt_fr))
    
scp_file.close() 
    

    

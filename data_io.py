##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import numpy as np
import sys
from utils import compute_cw_max,dict_fea_lab_arch,is_sequential_dict
import os
import configparser
import re, gzip, struct

def load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right, max_sequence_length, output_folder, fea_only=False):
    def _input_is_wav_file(fea_scp):
        with open(fea_scp, 'r') as f:
            first_line = f.readline()
        ark_file = first_line.split(' ')[1].split(':')[0]
        with open(ark_file, 'rb') as f:
            first_ark_line = f.readline()
        return b'RIFF' in first_ark_line
    def _input_is_feature_file(fea_scp):
        return not _input_is_wav_file(fea_scp)
    def _read_features_and_labels_with_kaldi(fea_scp, fea_opts, fea_only, lab_folder, lab_opts, output_folder):
        fea = dict()
        lab = dict()
        if _input_is_feature_file(fea_scp):
            kaldi_bin="copy-feats"
            read_function = read_mat_ark
        elif _input_is_wav_file(fea_scp):
            kaldi_bin="wav-copy"
            read_function = read_vec_flt_ark
        fea = { k:m for k,m in read_function('ark:'+kaldi_bin+' scp:'+fea_scp+' ark:- |'+fea_opts,output_folder) }
        if not fea_only:
            lab = { k:v for k,v in read_vec_int_ark('gunzip -c '+lab_folder+'/ali*.gz | '+lab_opts+' '+lab_folder+'/final.mdl ark:- ark:-|',output_folder)  if k in fea} # Note that I'm copying only the aligments of the loaded fea
            fea = {k: v for k, v in fea.items() if k in lab} # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")
        return fea, lab
    def _chunk_features_and_labels(max_sequence_length, fea, lab, fea_only, input_is_wav):
        def _append_to_concat_list(fea_chunked, lab_chunked, fea_conc, lab_conc, name):
            for j in range(0, len(fea_chunked)):
                fea_conc.append(fea_chunked[j])
                lab_conc.append(lab_chunked[j])
                if len(fea_chunked) > 1:
                    snt_name.append(name+'_split'+str(j))
                else:
                    snt_name.append(k)
            return fea_conc, lab_conc
        def _chunk(max_sequence_length, fea, lab, fea_only):
            def _chunk_by_input_and_output_chunk_config(chunk_config, fea, lab, fea_only):
                ''' 
                If the sequence length is above the threshold, we split it with a minimal length max/4
                If max length = 500, then the split will start at 500 + (500/4) = 625. 
                A seq of length 625 will be splitted in one of 500 and one of 125
                '''
                chunk_size_fea, chunk_step_fea, chunk_size_lab, chunk_step_lab = chunk_config['chunk_size_fea'], chunk_config['chunk_step_fea'], chunk_config['chunk_size_lab'], chunk_config['chunk_step_lab']
                fea_chunked = list()
                lab_chunked = list()
                split_threshold_fea = chunk_size_fea + (chunk_size_fea/4)
                if(len(fea) > chunk_size_fea) and chunk_size_fea>0:
                    nr_of_chunks = (len(fea) + chunk_size_fea - 1) // chunk_size_fea
                    for i in range(nr_of_chunks):
                        chunk_start_fea = i * chunk_step_fea
                        if(len(fea[chunk_start_fea:]) > split_threshold_fea):
                            chunk_end_fea = chunk_start_fea + chunk_size_fea
                            fea_chunk = fea[chunk_start_fea:chunk_end_fea]
                            if not fea_only:
                                chunk_start_lab = i * chunk_step_lab
                                chunk_end_lab = chunk_start_lab + chunk_size_lab
                                lab_chunk = lab[chunk_start_lab:chunk_end_lab]
                            else:
                                lab_chunk = np.zeros((fea_chunk.shape[0],))
                            fea_chunked.append(fea_chunk)
                            lab_chunked.append(lab_chunk)
                        else:
                            fea_chunk = fea[chunk_start_fea:]
                            if not fea_only:
                                chunk_start_lab = i * chunk_step_lab
                                lab_chunk = lab[chunk_start_lab:]
                            else:
                                lab_chunk = np.zeros((fea_chunk.shape[0],))
                            lab_chunked.append(lab_chunk)
                            fea_chunked.append(fea_chunk)
                            break
                else:
                    fea_chunked.append(fea)
                    if not fea_only:
                      lab_chunked.append(lab)
                    else:
                      lab_chunked.append(np.zeros((fea.shape[0],)))
                return fea_chunked, lab_chunked
           
            chunk_config = dict()
            if type(max_sequence_length) == dict:
                chunk_config['chunk_size_fea'] = max_sequence_length['chunk_size_fea']
                chunk_config['chunk_step_fea'] = max_sequence_length['chunk_step_fea']
                chunk_config['chunk_size_lab'] = max_sequence_length['chunk_size_lab']
                chunk_config['chunk_step_lab'] = max_sequence_length['chunk_step_lab']
            elif type(max_sequence_length) == int:
                chunk_config['chunk_size_fea'] = max_sequence_length
                chunk_config['chunk_step_fea'] = max_sequence_length
                chunk_config['chunk_size_lab'] = max_sequence_length
                chunk_config['chunk_step_lab'] = max_sequence_length
            else:
                raise ValueError('Unknown type of max_sequence_length')
            return _chunk_by_input_and_output_chunk_config(chunk_config, fea, lab, fea_only)

        snt_name = list()
        fea_conc = list()
        lab_conc = list()
        feature_keys_soted_by_sequence_length = sorted(sorted(fea.keys()), key=lambda k: len(fea[k]))
        for k in feature_keys_soted_by_sequence_length:
            fea_el = fea[k]
            lab_el = None
            if not fea_only:
                lab_el = lab[k]
            fea_chunked, lab_chunked = _chunk(max_sequence_length, fea_el, lab_el, fea_only)
            fea_conc, lab_conc = _append_to_concat_list(fea_chunked, lab_chunked, fea_conc, lab_conc, k)
        return fea_conc, lab_conc, snt_name
    def _concatenate_features_and_labels(fea_conc, lab_conc):
        def _sort_chunks_by_length(fea_conc, lab_conc):
            fea_zipped = zip(fea_conc,lab_conc)
            fea_sorted = sorted(fea_zipped, key=lambda x: x[0].shape[0])
            fea_conc,lab_conc = zip(*fea_sorted)
            return fea_conc, lab_conc
        def _get_end_index_from_list(conc):
            end_snt=0
            end_index=list()
            for entry in conc:
                end_snt=end_snt+entry.shape[0]
                end_index.append(end_snt)
            return end_index

        fea_conc, lab_conc = _sort_chunks_by_length(fea_conc, lab_conc)
        end_index_fea = _get_end_index_from_list(fea_conc)
        end_index_lab = _get_end_index_from_list(lab_conc)
        fea_conc=np.concatenate(fea_conc)
        lab_conc=np.concatenate(lab_conc)
        return fea_conc, lab_conc, end_index_fea, end_index_lab
    def _match_feature_and_label_sequence_lengths(fea, lab, max_sequence_length):
        ALLOW_FRAME_DIFF_LARGER_ONE = False
        def _adjust_feature_sequence_length(fea, nr_of_fea_for_lab):
            nr_of_fea = fea.shape[0]
            if nr_of_fea > nr_of_fea_for_lab:
                fea_adj = np.take(fea, range(nr_of_fea_for_lab), axis=0)
            elif nr_of_fea < nr_of_fea_for_lab:
                padding = np.zeros(shape=(nr_of_fea_for_lab-nr_of_fea,) + fea.shape[1:])
                fea_adj = np.concatenate([fea, padding], axis=0)
            else:
                fea_adj = fea
            return fea_adj
        chunk_size_fea = max_sequence_length['chunk_size_fea']
        chunk_step_fea = max_sequence_length['chunk_step_fea']
        chunk_size_lab = max_sequence_length['chunk_size_lab']
        chunk_step_lab = max_sequence_length['chunk_step_lab']
        window_shift = max_sequence_length['window_shift']
        window_size = max_sequence_length['window_size']
        for k in fea.keys():
            nr_of_fea = fea[k].shape[0]
            nr_of_lab = lab[k].shape[0]
            nr_of_fea_for_lab = (nr_of_lab - 1) * window_shift + window_size
            if abs(nr_of_fea - nr_of_fea_for_lab) > window_shift and not ALLOW_FRAME_DIFF_LARGER_ONE:
               raise ValueError('Nr. of features: ' + str(nr_of_fea) + ' does not match nr. of labels: ' + str(nr_of_lab) + ' with expected nr. of features: ' + str(nr_of_fea_for_lab))
            fea[k] = _adjust_feature_sequence_length(fea[k], nr_of_fea_for_lab)
        return fea, lab

    fea, lab = _read_features_and_labels_with_kaldi(fea_scp, fea_opts, fea_only, lab_folder, lab_opts, output_folder)
    if _input_is_wav_file(fea_scp) and (not fea_only):
        fea, lab = _match_feature_and_label_sequence_lengths(fea, lab, max_sequence_length)
    fea_chunks, lab_chunks, chunk_names = _chunk_features_and_labels(max_sequence_length, fea, lab, fea_only, _input_is_wav_file(fea_scp))
    fea_conc, lab_conc, end_index_fea, end_index_lab = _concatenate_features_and_labels(fea_chunks, lab_chunks)
    return [chunk_names,fea_conc,lab_conc,np.asarray(end_index_fea),np.asarray(end_index_lab)] 


def context_window_old(fea,left,right):
 
 N_row=fea.shape[0]
 N_fea=fea.shape[1]
 frames = np.empty((N_row-left-right, N_fea*(left+right+1)))
 
 for frame_index in range(left,N_row-right):
  right_context=fea[frame_index+1:frame_index+right+1].flatten() # right context
  left_context=fea[frame_index-left:frame_index].flatten() # left context
  current_frame=np.concatenate([left_context,fea[frame_index],right_context])
  frames[frame_index-left]=current_frame

 return frames

def context_window(fea,left,right):
 
    N_elem=fea.shape[0]
    N_fea=fea.shape[1]
    
    fea_conc=np.empty([N_elem,N_fea*(left+right+1)])
    
    index_fea=0
    for lag in range(-left,right+1):
        fea_conc[:,index_fea:index_fea+fea.shape[1]]=np.roll(fea,-lag,axis=0)
        index_fea=index_fea+fea.shape[1]
        
    fea_conc=fea_conc[left:fea_conc.shape[0]-right]
    return fea_conc


def load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,left,right,max_sequence_length, output_folder,fea_only=False):
  
  # open the file
  [data_name,data_set,data_lab,end_index_fea,end_index_lab]=load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right, max_sequence_length, output_folder, fea_only)

  # TODO: currently end_index_lab is ignored

  # Context window
  if left!=0 or right!=0:
      data_set=context_window(data_set,left,right)

  end_index_fea=end_index_fea-left
  end_index_fea[-1]=end_index_fea[-1]-right

  # mean and variance normalization
  data_set=(data_set-np.mean(data_set,axis=0))/np.std(data_set,axis=0)

  # Label processing
  data_lab=data_lab-data_lab.min()
  if right>0:
    data_lab=data_lab[left:-right]
  else:
    data_lab=data_lab[left:]   
  
  data_set=np.column_stack((data_set, data_lab))

  return [data_name,data_set,end_index_fea]

def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts 

def read_lab_fea_refac01(cfg_file, fea_only, shared_list, output_folder):
    def _read_chunk_specific_config(cfg_file):
        if not(os.path.exists(cfg_file)):
            sys.stderr.write('ERROR: The config file %s does not exist!\n'%(cfg_file))
            sys.exit(0)
        else:
            config = configparser.ConfigParser()
            config.read(cfg_file)
        return config
    def _read_from_config(config, fea_only):
        def _get_max_seq_length_from_config_str(config_str):
            max_seq_length=[int(e) for e in config_str.split(',')]
            if len(max_seq_length) == 1:
                max_seq_length = max_seq_length[0]
            else:
                assert len(max_seq_length) == 6
                max_seq_length_list = max_seq_length
                max_seq_length = dict()
                max_seq_length['chunk_size_fea'] = max_seq_length_list[0]
                max_seq_length['chunk_step_fea'] = max_seq_length_list[1]
                max_seq_length['chunk_size_lab'] = max_seq_length_list[2]
                max_seq_length['chunk_step_lab'] = max_seq_length_list[3]
                max_seq_length['window_shift'] = max_seq_length_list[4]
                max_seq_length['window_size'] = max_seq_length_list[5]
            return max_seq_length
        
        to_do=config['exp']['to_do']
        if to_do=='train':
            max_seq_length=_get_max_seq_length_from_config_str(config['batches']['max_seq_length_train'])
        if to_do=='valid':
            max_seq_length=_get_max_seq_length_from_config_str(config['batches']['max_seq_length_valid'])
        if to_do=='forward':
            max_seq_length=-1 # do to break forward sentences
            fea_only=True
        fea_dict, lab_dict, arch_dict = dict_fea_lab_arch(config, fea_only)
        seq_model = is_sequential_dict(config, arch_dict)
        return to_do, max_seq_length, fea_dict, lab_dict, arch_dict, seq_model
    def _read_features_and_labels(fea_dict, lab_dict, max_seq_length, fea_only, output_folder):
        def _get_fea_config_from_dict(fea_dict_entr):
            fea_scp = fea_dict_entr[1]
            fea_opts = fea_dict_entr[2]
            cw_left = int(fea_dict_entr[3])
            cw_right = int(fea_dict_entr[4])
            return fea_scp, fea_opts, cw_left, cw_right
        def _get_lab_config_from_dict(lab_dict_entr, fea_only):
            if fea_only:
                lab_folder = None 
                lab_opts = None
            else:
                lab_folder = lab_dict_entr[1]
                lab_opts = lab_dict_entr[2]
            return lab_folder, lab_opts
        def _compensate_for_different_context_windows(data_set_fea, data_set_lab, cw_left_max, cw_left, cw_right_max, cw_right, data_end_index_fea, data_end_index_lab):
            data_set_lab = np.take(data_set_lab, range(cw_left_max-cw_left,data_set_lab.shape[0]-(cw_right_max-cw_right)), axis=0, mode='clip')
            data_set_fea = np.take(data_set_fea, range(cw_left_max-cw_left,data_set_fea.shape[0]-(cw_right_max-cw_right)), axis=0, mode='clip')
            data_end_index_fea = data_end_index_fea - (cw_left_max - cw_left)
            data_end_index_lab = data_end_index_lab - (cw_left_max - cw_left)
            data_end_index_fea[-1] = data_end_index_fea[-1] - (cw_right_max - cw_right)
            data_end_index_lab[-1] = data_end_index_lab[-1] - (cw_right_max - cw_right)
            return data_set_lab, data_set_fea, data_end_index_fea, data_end_index_lab
        def _update_data(data_set, labs, fea_dict, fea, fea_index, data_set_fea, labs_fea, cnt_fea, cnt_lab):
            if cnt_fea==0 and cnt_lab==0:
                data_set=data_set_fea
                labs=labs_fea
                fea_dict[fea].append(fea_index)
                fea_index=fea_index+data_set_fea.shape[1]
                fea_dict[fea].append(fea_index)
                fea_dict[fea].append(fea_dict[fea][6]-fea_dict[fea][5])
            elif cnt_fea==0 and (not cnt_lab==0):
                labs=np.column_stack((labs,labs_fea))
            elif (not cnt_fea==0) and cnt_lab==0:
                data_set=np.column_stack((data_set,data_set_fea))
                fea_dict[fea].append(fea_index)
                fea_index=fea_index+data_set_fea.shape[1]
                fea_dict[fea].append(fea_index)
                fea_dict[fea].append(fea_dict[fea][6]-fea_dict[fea][5])
            return data_set, labs, fea_dict, fea_index
        def _check_consistency(data_name, data_name_fea, data_end_index_fea_ini, data_end_index_fea, data_end_index_lab_ini, data_end_index_lab):
            if not (data_name == data_name_fea):
                sys.stderr.write('ERROR: different sentence ids are detected for the different features. Plase check again input feature lists"\n')
                sys.exit(0)
            if not (data_end_index_fea_ini == data_end_index_fea).all():
                sys.stderr.write('ERROR end_index must be the same for all the sentences"\n')
                sys.exit(0)
            if not (data_end_index_lab_ini == data_end_index_lab).all():
                sys.stderr.write('ERROR end_index must be the same for all the sentences"\n')
                sys.exit(0)
        def _update_lab_dict(lab_dict, data_set):
            cnt_lab=0
            for lab in lab_dict.keys():
                lab_dict[lab].append(data_set.shape[1]+cnt_lab)
                cnt_lab=cnt_lab+1
            return lab_dict
        def _load_chunk_refac01(fea_scp,fea_opts,lab_folder,lab_opts,left,right,max_sequence_length, output_folder,fea_only=False):
            [data_name,data_set,data_lab,end_index_fea,end_index_lab]=load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right, max_sequence_length, output_folder, fea_only)
            # TODO: this function will currently only work well if no context window is given or fea and lab have the same time dimensionality
            # Context window
            if left!=0 or right!=0:
                data_set=context_window(data_set,left,right)
            end_index_fea = end_index_fea - left
            end_index_lab = end_index_lab - left
            end_index_fea[-1] = end_index_fea[-1] - right
            end_index_lab[-1] = end_index_lab[-1] - right
            # mean and variance normalization
            data_set=(data_set-np.mean(data_set,axis=0))/np.std(data_set,axis=0)
            # Label processing
            data_lab=data_lab-data_lab.min()
            if right>0:
                data_lab=data_lab[left:-right]
            else:
                data_lab=data_lab[left:]   
            if len(data_set.shape) == 1:
                data_set = np.expand_dims(data_set, -1)
            return [data_name, data_set, data_lab, end_index_fea, end_index_lab]
        
        cw_left_max, cw_right_max = compute_cw_max(fea_dict)
        fea_index=0
        cnt_fea=0
        data_name = None 
        data_end_index_fea_ini = None 
        data_end_index_lab_ini = None 
        data_set = None
        labs = None
        for fea in fea_dict.keys():
            fea_scp, fea_opts, cw_left, cw_right = _get_fea_config_from_dict(fea_dict[fea])
            cnt_lab=0
            if fea_only:
                lab_dict.update({'lab_name':'none'})
            for lab in lab_dict.keys():
                lab_folder, lab_opts = _get_lab_config_from_dict(lab_dict[lab], fea_only)
                data_name_fea, data_set_fea, data_set_lab, data_end_index_fea, data_end_index_lab = _load_chunk_refac01(fea_scp, fea_opts, lab_folder, lab_opts, cw_left, cw_right, max_seq_length, output_folder, fea_only)
                if sum([abs(e) for e in [cw_left_max, cw_right_max, cw_left, cw_right]]) != 0: 
                    data_set_lab, data_set_fea, data_end_index_fea, data_end_index_lab = _compensate_for_different_context_windows(data_set_fea, data_set_lab, cw_left_max, cw_left, cw_right_max, cw_right, data_end_index_fea, data_end_index_lab)
                if cnt_fea == 0 and cnt_lab == 0:
                    data_end_index_fea_ini = data_end_index_fea
                    data_end_index_lab_ini = data_end_index_lab
                    data_name = data_name_fea
                data_set, labs, fea_dict, fea_index = _update_data(data_set, labs, fea_dict, fea, fea_index, data_set_fea, data_set_lab, cnt_fea, cnt_lab)
                _check_consistency(data_name, data_name_fea, data_end_index_fea_ini, data_end_index_fea, data_end_index_lab_ini, data_end_index_lab)
                cnt_lab=cnt_lab+1
            cnt_fea=cnt_fea+1
        if not fea_only:
            lab_dict = _update_lab_dict(lab_dict, data_set)
        return data_name, data_end_index_fea_ini, data_end_index_lab_ini, fea_dict, lab_dict, data_set, labs
    def _reorder_data_set(data_set, labs, seq_model, to_do):
        if not(seq_model) and to_do != 'forward' and (data_set.shape[0] == labs.shape[0]):
            data_set_shape = data_set.shape[1]
            data_set_joint = np.column_stack((data_set,labs))
            np.random.shuffle(data_set)
            data_set = data_set_joint[:, :data_set_shape]
            labs = np.squeeze(data_set_joint[:, data_set_shape:], axis=-1)
        return data_set, labs
    def _append_to_shared_list(shared_list, data_name, data_end_index_fea, data_end_index_lab, fea_dict, lab_dict, arch_dict, data_set):
        shared_list.append(data_name)
        shared_list.append(data_end_index_fea)
        shared_list.append(data_end_index_lab)
        shared_list.append(fea_dict)
        shared_list.append(lab_dict)
        shared_list.append(arch_dict)
        shared_list.append(data_set)
        return shared_list

    config = _read_chunk_specific_config(cfg_file)
    to_do, max_seq_length, fea_dict, lab_dict, arch_dict, seq_model = _read_from_config(config, fea_only)
    data_name, data_end_index_fea, data_end_index_lab, fea_dict, lab_dict, data_set, labs = _read_features_and_labels(fea_dict, lab_dict, max_seq_length, fea_only, output_folder)
    data_set, labs = _reorder_data_set(data_set, labs, seq_model, to_do)
    data_set = {'input': data_set, 'ref': labs}
    shared_list = _append_to_shared_list(shared_list, data_name, data_end_index_fea, data_end_index_lab, fea_dict, lab_dict, arch_dict, data_set)

def read_lab_fea(cfg_file,fea_only,shared_list,output_folder):
    
    # Reading chunk-specific cfg file (first argument-mandatory file) 
    if not(os.path.exists(cfg_file)):
         sys.stderr.write('ERROR: The config file %s does not exist!\n'%(cfg_file))
         sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config.read(cfg_file)
        
    
    # Reading some cfg parameters
    to_do=config['exp']['to_do']
    
    if to_do=='train':
        max_seq_length=int(config['batches']['max_seq_length_train']) #*(int(info_file[-13:-10])+1) # increasing over the epochs

    if to_do=='valid':
        max_seq_length=int(config['batches']['max_seq_length_valid'])

    if to_do=='forward':
        max_seq_length=-1 # do to break forward sentences
    
    [fea_dict,lab_dict,arch_dict]=dict_fea_lab_arch(config,fea_only)
    [cw_left_max,cw_right_max]=compute_cw_max(fea_dict)
    
    fea_index=0
    cnt_fea=0
    for fea in fea_dict.keys():
        
        # reading the features
        fea_scp=fea_dict[fea][1]
        fea_opts=fea_dict[fea][2]
        cw_left=int(fea_dict[fea][3])
        cw_right=int(fea_dict[fea][4])
        
        cnt_lab=0

        # Production case, we don't have labels (lab_name = none)
        if fea_only:
          lab_dict.update({'lab_name':'none'})
        for lab in lab_dict.keys():
            # Production case, we don't have labels (lab_name = none)
            if fea_only:
              lab_folder=None 
              lab_opts=None
            else:
              lab_folder=lab_dict[lab][1]
              lab_opts=lab_dict[lab][2]
    
            [data_name_fea,data_set_fea,data_end_index_fea]=load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,cw_left,cw_right,max_seq_length, output_folder, fea_only)
    
            
            # making the same dimenion for all the features (compensating for different context windows)
            labs_fea=data_set_fea[cw_left_max-cw_left:data_set_fea.shape[0]-(cw_right_max-cw_right),-1]
            data_set_fea=data_set_fea[cw_left_max-cw_left:data_set_fea.shape[0]-(cw_right_max-cw_right),0:-1]
            data_end_index_fea=data_end_index_fea-(cw_left_max-cw_left)
            data_end_index_fea[-1]=data_end_index_fea[-1]-(cw_right_max-cw_right)
    
            
            
            if cnt_fea==0 and cnt_lab==0:
                data_set=data_set_fea
                labs=labs_fea
                data_end_index=data_end_index_fea
                data_end_index=data_end_index_fea
                data_name=data_name_fea
                
                fea_dict[fea].append(fea_index)
                fea_index=fea_index+data_set_fea.shape[1]
                fea_dict[fea].append(fea_index)
                fea_dict[fea].append(fea_dict[fea][6]-fea_dict[fea][5])
                
                
            else:
                if cnt_fea==0:
                    labs=np.column_stack((labs,labs_fea))
                
                if cnt_lab==0:
                    data_set=np.column_stack((data_set,data_set_fea))
                    fea_dict[fea].append(fea_index)
                    fea_index=fea_index+data_set_fea.shape[1]
                    fea_dict[fea].append(fea_index)
                    fea_dict[fea].append(fea_dict[fea][6]-fea_dict[fea][5])
                
                
                # Checks if lab_names are the same for all the features
                if not(data_name==data_name_fea):
                    sys.stderr.write('ERROR: different sentence ids are detected for the different features. Plase check again input feature lists"\n')
                    sys.exit(0)
                
                # Checks if end indexes are the same for all the features
                if not(data_end_index==data_end_index_fea).all():
                    sys.stderr.write('ERROR end_index must be the same for all the sentences"\n')
                    sys.exit(0)
                    
            cnt_lab=cnt_lab+1
    
    
        cnt_fea=cnt_fea+1
        
    cnt_lab=0
    if not fea_only:   
      for lab in lab_dict.keys():
          lab_dict[lab].append(data_set.shape[1]+cnt_lab)
          cnt_lab=cnt_lab+1
           
    data_set=np.column_stack((data_set,labs))
    
    # check automatically if the model is sequential
    seq_model=is_sequential_dict(config,arch_dict)
    
    # Randomize if the model is not sequential
    if not(seq_model) and to_do!='forward':
        np.random.shuffle(data_set)
     
    # Split dataset in many part. If the dataset is too big, we can have issues to copy it into the shared memory (due to pickle limits)
    #N_split=10
    #data_set=np.array_split(data_set, N_split)
    
    # Adding all the elements in the shared list    
    shared_list.append(data_name)
    shared_list.append(data_end_index)
    shared_list.append(fea_dict)
    shared_list.append(lab_dict)
    shared_list.append(arch_dict)
    shared_list.append(data_set)
    



# The following libraries are copied from kaldi-io-for-python project (https://github.com/vesis84/kaldi-io-for-python)
    
# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")
    
#################################################
# Define all custom exceptions,
class UnsupportedDataType(Exception): pass
class UnknownVectorHeader(Exception): pass
class UnknownMatrixHeader(Exception): pass

class BadSampleSize(Exception): pass
class BadInputFormat(Exception): pass

class SubprocessFailed(Exception): pass

#################################################
# Data-type independent helper functions,

def open_or_fd(file, output_folder,mode='rb'):
  """ fd = open_or_fd(file)
   Open file, gzipped file, pipe, or forward the file-descriptor.
   Eventually seeks in the 'file' argument contains ':offset' suffix.
  """
  offset = None

  try:
    # strip 'ark:' prefix from r{x,w}filename (optional),
    if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
      (prefix,file) = file.split(':',1)
    # separate offset from filename (optional),
    if re.search(':[0-9]+$', file):
      (file,offset) = file.rsplit(':',1)
    # input pipe?
    if file[-1] == '|':
      fd = popen(file[:-1], output_folder,'rb') # custom,
    # output pipe?
    elif file[0] == '|':
      fd = popen(file[1:], output_folder,'wb') # custom,
    # is it gzipped?
    elif file.split('.')[-1] == 'gz':
      fd = gzip.open(file, mode)
    # a normal file...
    else:
      fd = open(file, mode)
  except TypeError:
    # 'file' is opened file descriptor,
    fd = file
  # Eventually seek to offset,
  if offset != None: fd.seek(int(offset))
  
  return fd

# based on '/usr/local/lib/python3.4/os.py'
def popen(cmd, output_folder,mode="rb"):
  if not isinstance(cmd, str):
    raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

  import subprocess, io, threading

  # cleanup function for subprocesses,
  def cleanup(proc, cmd):
    ret = proc.wait()
    if ret > 0:
      raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
    return

  # text-mode,
  if mode == "r":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdout)
  elif mode == "w":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdin)
  # binary,
  elif mode == "rb":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdout
  elif mode == "wb":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdin
  # sanity,
  else:
    raise ValueError("invalid mode %s" % mode)


def read_key(fd):
  """ [key] = read_key(fd)
   Read the utterance-key from the opened ark/stream descriptor 'fd'.
  """
  key = ''
  while 1:
    char = fd.read(1).decode("latin1")
    if char == '' : break
    if char == ' ' : break
    key += char
  key = key.strip()
  if key == '': return None # end of file,
  assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
  return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd,output_folder):
  """ Alias to 'read_vec_int_ark()' """
  return read_vec_int_ark(file_or_fd,output_folder)

def read_vec_int_ark(file_or_fd,output_folder):
  """ generator(key,vec) = read_vec_int_ark(file_or_fd)
   Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Read ark to a 'dictionary':
   d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      ali = read_vec_int(fd,output_folder)
      yield key, ali
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_vec_int(file_or_fd,output_folder):
  """ [int-vec] = read_vec_int(file_or_fd)
   Read kaldi integer vector, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd,output_folder)
  binary = fd.read(2).decode()
  if binary == '\0B': # binary flag
    assert(fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
    if vec_size == 0:
      return np.array([], dtype='int32')
    # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
    vec = np.frombuffer(fd.read(vec_size*5), dtype=[('size','int8'),('value','int32')], count=vec_size)
    assert(vec[0]['size'] == 4) # int32 size,
    ans = vec[:]['value'] # values are in 2nd column,
  else: # ascii,
    arr = (binary + fd.readline().decode()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=int)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans

# Writing,
def write_vec_int(file_or_fd, output_folder, v, key=''):
  """ write_vec_int(f, v, key='')
   Write a binary kaldi integer vector to filename or stream.
   Arguments:
   file_or_fd : filename or opened file descriptor for writing,
   v : the vector to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

   Example of writing single vector:
   kaldi_io.write_vec_int(filename, vec)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
  """
  fd = open_or_fd(file_or_fd, output_folder, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')
  try:
    if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
    fd.write('\0B'.encode()) # we write binary!
    # dim,
    fd.write('\4'.encode()) # int32 type,
    fd.write(struct.pack(np.dtype('int32').char, v.shape[0]))
    # data,
    for i in range(len(v)):
      fd.write('\4'.encode()) # int32 type,
      fd.write(struct.pack(np.dtype('int32').char, v[i])) # binary,
  finally:
    if fd is not file_or_fd : fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

# Reading,
def read_vec_flt_scp(file_or_fd,output_folder):
  """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
   Returns generator of (key,vector) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,vec in kaldi_io.read_vec_flt_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    for line in fd:
      (key,rxfile) = line.decode().split(' ')
      vec = read_vec_flt(rxfile,output_folder)
      yield key, vec
  finally:
    if fd is not file_or_fd : fd.close()

def read_vec_flt_ark(file_or_fd,output_folder):
  """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
   Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Read ark to a 'dictionary':
   d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      ali = read_vec_flt(fd,output_folder)
      yield key, ali
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_vec_flt(file_or_fd,output_folder):
  """ [flt-vec] = read_vec_flt(file_or_fd)
   Read kaldi float vector, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd,output_folder)
  binary = fd.read(2).decode()
  if binary == '\0B': # binary flag
    return _read_vec_flt_binary(fd)
  elif binary == 'RI':
    return _read_vec_flt_riff(fd)
  else:  # ascii,
    arr = (binary + fd.readline().decode()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=float)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans

def _read_vec_flt_riff(fd):
    RIFF_CHUNK_DESCR_HEADER_SIZE = 12
    ALREADY_READ_HEADER_BYTES = 2
    SUB_CHUNK_HEADER_SIZE = 8
    DATA_CHUNK_HEADER_SIZE = 8
    def pcm2float(signal, dtype='float32'):
        signal = np.asarray(signal)
        dtype = np.dtype(dtype)
        return signal.astype(dtype) / dtype.type(-np.iinfo(signal.dtype).min)

    import struct
    header = fd.read(RIFF_CHUNK_DESCR_HEADER_SIZE - ALREADY_READ_HEADER_BYTES)
    assert header[:2] == b'FF'
    chunk_header = fd.read(SUB_CHUNK_HEADER_SIZE)
    subchunk_id, subchunk_size = struct.unpack('<4sI', chunk_header)
    aformat, channels, samplerate, byterate, block_align, bps = struct.unpack('HHIIHH', fd.read(subchunk_size))
    subchunk2_id, subchunk2_size = struct.unpack('<4sI', fd.read(DATA_CHUNK_HEADER_SIZE))
    pcm_data = np.frombuffer(fd.read(subchunk2_size), dtype='int' + str(bps))
    return pcm2float(pcm_data)

def _read_vec_flt_binary(fd):
  header = fd.read(3).decode()
  if header == 'FV ' : sample_size = 4 # floats
  elif header == 'DV ' : sample_size = 8 # doubles
  else : raise UnknownVectorHeader("The header contained '%s'" % header)
  assert (sample_size > 0)
  # Dimension,
  assert (fd.read(1).decode() == '\4'); # int-size
  vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
  if vec_size == 0:
    return np.array([], dtype='float32')
  # Read whole vector,
  buf = fd.read(vec_size * sample_size)
  if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32')
  elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64')
  else : raise BadSampleSize
  return ans

# Writing,
def write_vec_flt(file_or_fd, output_folder, v, key=''):
  """ write_vec_flt(f, v, key='')
   Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
   Arguments:
   file_or_fd : filename or opened file descriptor for writing,
   v : the vector to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

   Example of writing single vector:
   kaldi_io.write_vec_flt(filename, vec)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
  """
  fd = open_or_fd(file_or_fd,output_folder, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')
  try:
    if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
    fd.write('\0B'.encode()) # we write binary!
    # Data-type,
    if v.dtype == 'float32': fd.write('FV '.encode())
    elif v.dtype == 'float64': fd.write('DV '.encode())
    else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % v.dtype)
    # Dim,
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, v.shape[0])) # dim
    # Data,
    fd.write(v.tobytes())
  finally:
    if fd is not file_or_fd : fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd,output_folder):
  """ generator(key,mat) = read_mat_scp(file_or_fd)
   Returns generator of (key,matrix) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,mat in kaldi_io.read_mat_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    for line in fd:
      (key,rxfile) = line.decode().split(' ')
      mat = read_mat(rxfile,output_folder)
      yield key, mat
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat_ark(file_or_fd,output_folder):
  """ generator(key,mat) = read_mat_ark(file_or_fd)
   Returns generator of (key,matrix) tuples, read from ark file/stream.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the ark:
   for key,mat in kaldi_io.read_mat_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
  """

  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      mat = read_mat(fd,output_folder)
      yield key, mat
      key = read_key(fd)   
  finally:
    if fd is not file_or_fd : fd.close()
  


def read_mat(file_or_fd,output_folder):
  """ [mat] = read_mat(file_or_fd)
   Reads single kaldi matrix, supports ascii and binary.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    binary = fd.read(2).decode()
    if binary == '\0B' :
      mat = _read_mat_binary(fd)
    else:
      assert(binary == ' [')
      mat = _read_mat_ascii(fd)
  finally:
    if fd is not file_or_fd: fd.close()
  return mat

def _read_mat_binary(fd):
  # Data type
  header = fd.read(3).decode()
  # 'CM', 'CM2', 'CM3' are possible values,
  if header.startswith('CM'): return _read_compressed_mat(fd, header)
  elif header == 'FM ': sample_size = 4 # floats
  elif header == 'DM ': sample_size = 8 # doubles
  else: raise UnknownMatrixHeader("The header contained '%s'" % header)
  assert(sample_size > 0)
  # Dimensions
  s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
  # Read whole matrix
  buf = fd.read(rows * cols * sample_size)
  if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32')
  elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64')
  else : raise BadSampleSize
  mat = np.reshape(vec,(rows,cols))
  return mat

def _read_mat_ascii(fd):
  rows = []
  while 1:
    line = fd.readline().decode()
    if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
    if len(line.strip()) == 0 : continue # skip empty line
    arr = line.strip().split()
    if arr[-1] != ']':
      rows.append(np.array(arr,dtype='float32')) # not last line
    else:
      rows.append(np.array(arr[:-1],dtype='float32')) # last line
      mat = np.vstack(rows)
      return mat


def _read_compressed_mat(fd, format):
  """ Read a compressed matrix,
      see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
      methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
  """
  assert(format == 'CM ') # The formats CM2, CM3 are not supported...

  # Format of header 'struct',
  global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
  per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

  # Read global header,
  globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

  # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
  #                         {           cols           }{     size         }
  col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
  col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)
  data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

  mat = np.zeros((cols,rows), dtype='float32')
  p0 = col_headers[:, 0].reshape(-1, 1)
  p25 = col_headers[:, 1].reshape(-1, 1)
  p75 = col_headers[:, 2].reshape(-1, 1)
  p100 = col_headers[:, 3].reshape(-1, 1)
  mask_0_64 = (data <= 64)
  mask_193_255 = (data > 192)
  mask_65_192 = (~(mask_0_64 | mask_193_255))

  mat += (p0  + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
  mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
  mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

  return mat.T # transpose! col-major -> row-major,


# Writing,
def write_mat(output_folder,file_or_fd, m, key=''):
  """ write_mat(f, m, key='')
  Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
  Arguments:
   file_or_fd : filename of opened file descriptor for writing,
   m : the matrix to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

   Example of writing single matrix:
   kaldi_io.write_mat(filename, mat)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,mat in dict.iteritems():
       kaldi_io.write_mat(f, mat, key=key)
  """
  fd = open_or_fd(file_or_fd, output_folder, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')
  try:
    if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
    fd.write('\0B'.encode()) # we write binary!
    # Data-type,
    if m.dtype == 'float32': fd.write('FM '.encode())
    elif m.dtype == 'float64': fd.write('DM '.encode())
    else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % m.dtype)
    # Dims,
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, m.shape[0])) # rows
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, m.shape[1])) # cols
    # Data,
    fd.write(m.tobytes())
  finally:
    if fd is not file_or_fd : fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#

def read_cnet_ark(file_or_fd,output_folder):
  """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
  return read_post_ark(file_or_fd,output_folder)

def read_post_rxspec(file_):
  """ adaptor to read both 'ark:...' and 'scp:...' inputs of posteriors,
  """
  if file_.startswith("ark:"):
      return read_post_ark(file_)
  elif file_.startswith("scp:"):
      return read_post_scp(file_)
  else:
      print("unsupported intput type: %s" % file_)
      print("it should begint with 'ark:' or 'scp:'")
      sys.exit(1)

def read_post_scp(file_or_fd,output_folder):
  """ generator(key,post) = read_post_scp(file_or_fd)
   Returns generator of (key,post) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,post in kaldi_io.read_post_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:post for key,post in kaldi_io.read_post_scp(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    for line in fd:
      (key,rxfile) = line.decode().split(' ')
      post = read_post(rxfile)
      yield key, post
  finally:
    if fd is not file_or_fd : fd.close()

def read_post_ark(file_or_fd,output_folder):
  """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
   Returns generator of (key,posterior) tuples, read from ark file.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Iterate the ark:
   for key,post in kaldi_io.read_post_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:post for key,post in kaldi_io.read_post_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      post = read_post(fd)
      yield key, post
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_post(file_or_fd,output_folder):
  """ [post] = read_post(file_or_fd)
   Reads single kaldi 'Posterior' in binary format.

   The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
   the outer-vector is usually time axis, inner-vector are the records
   at given time,  and the tuple is composed of an 'index' (integer)
   and a 'float-value'. The 'float-value' can represent a probability
   or any other numeric value.

   Returns vector of vectors of tuples.
  """
  fd = open_or_fd(file_or_fd,output_folder)
  ans=[]
  binary = fd.read(2).decode(); assert(binary == '\0B'); # binary flag
  assert(fd.read(1).decode() == '\4'); # int-size
  outer_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

  # Loop over 'outer-vector',
  for i in range(outer_vec_size):
    assert(fd.read(1).decode() == '\4'); # int-size
    inner_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of records for frame (or bin)
    data = np.frombuffer(fd.read(inner_vec_size*10), dtype=[('size_idx','int8'),('idx','int32'),('size_post','int8'),('post','float32')], count=inner_vec_size)
    assert(data[0]['size_idx'] == 4)
    assert(data[0]['size_post'] == 4)
    ans.append(data[['idx','post']].tolist())

  if fd is not file_or_fd: fd.close()
  return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd,output_folder):
  """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
   Returns generator of (key,cntime) tuples, read from ark file.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.

   Iterate the ark:
   for key,time in kaldi_io.read_cntime_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:time for key,time in kaldi_io.read_post_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      cntime = read_cntime(fd)
      yield key, cntime
      key = read_key(fd)
  finally:
    if fd is not file_or_fd : fd.close()

def read_cntime(file_or_fd,output_folder):
  """ [cntime] = read_cntime(file_or_fd)
   Reads single kaldi 'Confusion Network time info', in binary format:
   C++ type: vector<tuple<float,float> >.
   (begin/end times of bins at the confusion network).

   Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

   file_or_fd : file, gzipped file, pipe or opened file descriptor.

   Returns vector of tuples.
  """
  fd = open_or_fd(file_or_fd,output_folder)
  binary = fd.read(2).decode(); assert(binary == '\0B'); # assuming it's binary

  assert(fd.read(1).decode() == '\4'); # int-size
  vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

  data = np.frombuffer(fd.read(vec_size*10), dtype=[('size_beg','int8'),('t_beg','float32'),('size_end','int8'),('t_end','float32')], count=vec_size)
  assert(data[0]['size_beg'] == 4)
  assert(data[0]['size_end'] == 4)
  ans = data[['t_beg','t_end']].tolist() # Return vector of tuples (t_beg,t_end),

  if fd is not file_or_fd : fd.close()
  return ans


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file):
  """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
   using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
   - t-beg, t-end is in seconds,
   - assumed 100 frames/second,
  """
  segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
  # Sanity checks,
  assert(len(segs) > 0) # empty segmentation is an error,
  assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
  # Convert time to frame-indexes,
  start = np.rint([100 * rec[2] for rec in segs]).astype(int)
  end = np.rint([100 * rec[3] for rec in segs]).astype(int)
  # Taken from 'read_lab_to_bool_vec', htk.py,
  frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
                   np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
  assert np.sum(end-start) == np.sum(frms)
  return frms

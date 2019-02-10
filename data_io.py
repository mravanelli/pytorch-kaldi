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

    
    fea = { k:m for k,m in read_mat_ark('ark:copy-feats scp:'+fea_scp+' ark:- |'+fea_opts,output_folder) }

    if not fea_only:
      lab = { k:v for k,v in read_vec_int_ark('gunzip -c '+lab_folder+'/ali*.gz | '+lab_opts+' '+lab_folder+'/final.mdl ark:- ark:-|',output_folder)  if k in fea} # Note that I'm copying only the aligments of the loaded fea
      fea = {k: v for k, v in fea.items() if k in lab} # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")

    end_snt=0
    end_index=[]
    snt_name=[]
    fea_conc=[]
    lab_conc=[]
    
    tmp=0
    for k in sorted(sorted(fea.keys()), key=lambda k: len(fea[k])):

        #####
        # If the sequence length is above the threshold, we split it with a minimal length max/4
        # If max length = 500, then the split will start at 500 + (500/4) = 625. 
        # A seq of length 625 will be splitted in one of 500 and one of 125

        
        if(len(fea[k]) > max_sequence_length) and max_sequence_length>0:

          fea_chunked = []
          lab_chunked = []

          for i in range((len(fea[k]) + max_sequence_length - 1) // max_sequence_length):
            if(len(fea[k][i * max_sequence_length:]) > max_sequence_length + (max_sequence_length/4)):
              fea_chunked.append(fea[k][i * max_sequence_length:(i + 1) * max_sequence_length])
              if not fea_only:
                lab_chunked.append(lab[k][i * max_sequence_length:(i + 1) * max_sequence_length])
              else:
                lab_chunked.append(np.zeros((fea[k][i * max_sequence_length:(i + 1) * max_sequence_length].shape[0],)))
            else:
              fea_chunked.append(fea[k][i * max_sequence_length:])
              if not fea_only:
                lab_chunked.append(lab[k][i * max_sequence_length:])
              else:
                lab_chunked.append(np.zeros((fea[k][i * max_sequence_length:].shape[0],)))
              break

          for j in range(0, len(fea_chunked)):
            fea_conc.append(fea_chunked[j])
            lab_conc.append(lab_chunked[j])
            snt_name.append(k+'_split'+str(j))
            
        else:
          fea_conc.append(fea[k])
          if not fea_only:
            lab_conc.append(lab[k])
          else:
            lab_conc.append(np.zeros((fea[k].shape[0],)))
          snt_name.append(k)

        tmp+=1

    fea_zipped = zip(fea_conc,lab_conc)
    fea_sorted = sorted(fea_zipped, key=lambda x: x[0].shape[0])
    fea_conc,lab_conc = zip(*fea_sorted)
      
    for entry in fea_conc:
      end_snt=end_snt+entry.shape[0]
      end_index.append(end_snt)

    fea_conc=np.concatenate(fea_conc)
    lab_conc=np.concatenate(lab_conc)

    return [snt_name,fea_conc,lab_conc,np.asarray(end_index)] 


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
        fea_conc[:,index_fea:index_fea+fea.shape[1]]=np.roll(fea,lag,axis=0)
        index_fea=index_fea+fea.shape[1]
        
    fea_conc=fea_conc[left:fea_conc.shape[0]-right]
    
    return fea_conc


def load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,left,right,max_sequence_length, output_folder,fea_only=False):
  
  # open the file
  [data_name,data_set,data_lab,end_index]=load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right, max_sequence_length, output_folder, fea_only)

  # Context window
  if left!=0 or right!=0:
      data_set=context_window(data_set,left,right)

  end_index=end_index-left
  end_index[-1]=end_index[-1]-right

  # mean and variance normalization
  data_set=(data_set-np.mean(data_set,axis=0))/np.std(data_set,axis=0)

  # Label processing
  data_lab=data_lab-data_lab.min()
  if right>0:
    data_lab=data_lab[left:-right]
  else:
    data_lab=data_lab[left:]   
  
  data_set=np.column_stack((data_set, data_lab))

  return [data_name,data_set,end_index]

def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts 

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
    
    
    [fea_dict,lab_dict,arch_dict]=dict_fea_lab_arch(config)
    
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
      vec = read_vec_flt(rxfile)
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
      ali = read_vec_flt(fd)
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
  else:  # ascii,
    arr = (binary + fd.readline().decode()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=float)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans

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

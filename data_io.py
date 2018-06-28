import kaldi_io
import numpy as np
from optparse import OptionParser
from six.moves.configparser import ConfigParser


def load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right):
    
 fea= { k:m for k,m in kaldi_io.read_mat_ark('ark:copy-feats scp:'+fea_scp+' ark:- |'+fea_opts) }
 lab= { k:v for k,v in kaldi_io.read_vec_int_ark('gunzip -c '+lab_folder+'/ali*.gz | '+lab_opts+' '+lab_folder+'/final.mdl ark:- ark:-|')  if k in fea} # Note that I'm copying only the aligments of the loaded fea
 fea={k: v for k, v in fea.items() if k in lab} # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")

 count=0
 end_snt=0
 end_index=[]
 snt_name=[]
 for k in sorted(fea.keys(), key=lambda k: len(fea[k])):
     if count==0:
         count=1
         fea_conc=fea[k]
         lab_conc=lab[k]
         end_snt=end_snt+fea[k].shape[0]-left
     else:
         fea_conc=np.concatenate([fea_conc,fea[k]],axis=0)
         lab_conc=np.concatenate([lab_conc,lab[k]],axis=0)
         end_snt=end_snt+fea[k].shape[0]
 
        
     end_index.append(end_snt) 
     snt_name.append(k)
     
 end_index[-1]=end_index[-1]-right
    
 return [snt_name,fea_conc,lab_conc,end_index] 


def context_window(fea,left,right):
 
 N_row=fea.shape[0]
 N_fea=fea.shape[1]
 frames = np.empty((N_row-left-right, N_fea*(left+right+1)))
 
 for frame_index in range(left,N_row-right):
  right_context=fea[frame_index+1:frame_index+right+1].flatten() # right context
  left_context=fea[frame_index-left:frame_index].flatten() # left context
  current_frame=np.concatenate([left_context,fea[frame_index],right_context])
  frames[frame_index-left]=current_frame

 return frames


def load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,left,right,shuffle_seed):
  # open the file
  [data_name,data_set,data_lab,end_index]=load_dataset(fea_scp,fea_opts,lab_folder,lab_opts,left,right)

  # Context window
  data_set=context_window(data_set,left,right)

  # mean and variance normalization
  data_set=(data_set-np.mean(data_set,axis=0))/np.std(data_set,axis=0)

  # Label processing
  data_lab=data_lab-data_lab.min()
  if right>0:
   data_lab=data_lab[left:-right]
  else:
   data_lab=data_lab[left:]   
    
  data_set=np.column_stack((data_set, data_lab))
  
  # shuffle (only for test data)
  if shuffle_seed>0:
   np.random.seed(shuffle_seed)
   np.random.shuffle(data_set)
   
  
  return [data_name,data_set,end_index]
  
def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts    
        
 
def read_opts():
 
 parser=OptionParser()
 parser.add_option("--cfg") # Mandatory
 (options,args)=parser.parse_args()

 cfg_file=options.cfg
 Config = ConfigParser()
 Config.read(cfg_file)

 # DATA
 options.out_folder=Config.get('data', 'out_folder')
 options.tr_fea_scp=Config.get('data', 'tr_fea_scp')
 options.tr_fea_opts=Config.get('data', 'tr_fea_opts')
 options.tr_lab_folder=Config.get('data', 'tr_lab_folder')
 options.tr_lab_opts=Config.get('data', 'tr_lab_opts')

 options.dev_fea_scp=Config.get('data', 'dev_fea_scp')
 options.dev_fea_opts=Config.get('data', 'dev_fea_opts')
 options.dev_lab_folder=Config.get('data', 'dev_lab_folder')
 options.dev_lab_opts=Config.get('data', 'dev_lab_opts')

 options.te_fea_scp=Config.get('data', 'te_fea_scp')
 options.te_fea_opts=Config.get('data', 'te_fea_opts')
 options.te_lab_folder=Config.get('data', 'te_lab_folder')
 options.te_lab_opts=Config.get('data', 'te_lab_opts')

 options.count_file=Config.get('data', 'count_file')
 options.pt_file=Config.get('data', 'pt_file')

 # ARCHITECTURE
 options.hidden_dim=Config.get('architecture', 'hidden_dim')
 options.N_hid=Config.get('architecture', 'N_hid')
 options.drop_rate=Config.get('architecture', 'drop_rate')
 options.use_batchnorm=Config.get('architecture', 'use_batchnorm')
 options.cw_left=Config.get('architecture', 'cw_left')
 options.cw_right=Config.get('architecture', 'cw_right')
 options.seed=Config.get('architecture', 'seed')
 options.use_cuda=Config.get('architecture', 'use_cuda')
 options.multi_gpu=Config.get('architecture', 'multi_gpu')
 
 if Config.has_option('architecture', 'bidir'):
    options.bidir=Config.get('architecture', 'bidir')
    
 if Config.has_option('architecture', 'resnet'):
    options.resnet=Config.get('architecture', 'resnet')
    
 if Config.has_option('architecture', 'act'):
    options.act=Config.get('architecture', 'act')
    
 if Config.has_option('architecture', 'resgate'):
    options.resgate=Config.get('architecture', 'resgate')
    
 if Config.has_option('architecture', 'minimal_gru'):
    options.minimal_gru=Config.get('architecture', 'minimal_gru')
    
 if Config.has_option('architecture', 'skip_conn'):
    options.skip_conn=Config.get('architecture', 'skip_conn')

 if Config.has_option('architecture', 'act_gate'):
    options.act_gate=Config.get('architecture', 'act_gate')
    
 if Config.has_option('architecture', 'use_laynorm'):
    options.use_laynorm=Config.get('architecture', 'use_laynorm')
    
 if Config.has_option('architecture', 'cost'):
    options.cost=Config.get('architecture', 'cost')
    
 if Config.has_option('architecture', 'NN_type'):
    options.NN_type=Config.get('architecture', 'NN_type')
    
 if Config.has_option('architecture', 'twin_reg'):
    options.twin_reg=Config.get('architecture', 'twin_reg')
    
 if Config.has_option('architecture', 'twin_w'):
    options.twin_w=Config.get('architecture', 'twin_w')
    
 if Config.has_option('architecture', 'cnn_pre'):
    options.cnn_pre=Config.get('architecture', 'cnn_pre')
    
    
 options.N_ep=Config.get('optimization', 'N_ep')
 options.lr=Config.get('optimization', 'lr')
 options.halving_factor=Config.get('optimization', 'halving_factor')
 options.improvement_threshold=Config.get('optimization', 'improvement_threshold')
 options.batch_size=Config.get('optimization', 'batch_size')
 options.save_gpumem=Config.get('optimization', 'save_gpumem')
 options.optimizer=Config.get('optimization', 'optimizer')

 return options



def read_conf():
 
 parser=OptionParser()
 parser.add_option("--cfg") # Mandatory
 (options,args)=parser.parse_args()

 cfg_file=options.cfg
 Config = ConfigParser()
 Config.read(cfg_file)

 # DATA
 if Config.has_option('data', 'out_file'):
  options.out_file=Config.get('data', 'out_file')

 if Config.has_option('data', 'fea_scp'):
  options.fea_scp=Config.get('data', 'fea_scp')
  
 if Config.has_option('data', 'fea_opts'):
  options.fea_opts=Config.get('data', 'fea_opts')

 if Config.has_option('data', 'lab_folder'):
  options.lab_folder=Config.get('data', 'lab_folder')
  
 if Config.has_option('data', 'lab_opts'):
  options.lab_opts=Config.get('data', 'lab_opts')
  
 if Config.has_option('data', 'pt_file'):
  options.pt_file=Config.get('data', 'pt_file')
  
 if Config.has_option('data', 'count_file'):
  options.count_file=Config.get('data', 'count_file')


 # TO DO

 if Config.has_option('todo', 'do_training'):
     options.do_training=Config.get('todo', 'do_training')

 if Config.has_option('todo', 'do_eval'):
     options.do_eval=Config.get('todo', 'do_eval')
     
 if Config.has_option('todo', 'do_forward'):
     options.do_forward=Config.get('todo', 'do_forward')
        
   
 # ARCHITECTURE
 
 if Config.has_option('architecture', 'hidden_dim'):
  options.hidden_dim=Config.get('architecture', 'hidden_dim')
 
 if Config.has_option('architecture', 'N_hid'):
  options.N_hid=Config.get('architecture', 'N_hid')
 
 if Config.has_option('architecture', 'drop_rate'):
  options.drop_rate=Config.get('architecture', 'drop_rate')
 
 if Config.has_option('architecture', 'use_batchnorm'):
  options.use_batchnorm=Config.get('architecture', 'use_batchnorm')
 
 if Config.has_option('architecture', 'cw_left'):
  options.cw_left=Config.get('architecture', 'cw_left')
 
 if Config.has_option('architecture', 'cw_right'):
  options.cw_right=Config.get('architecture', 'cw_right')
 
 if Config.has_option('architecture', 'use_seed'):
  options.seed=Config.get('architecture', 'seed')
 
 if Config.has_option('architecture', 'use_cuda'):
  options.use_cuda=Config.get('architecture', 'use_cuda')
 
 if Config.has_option('architecture', 'multi_gpu'):
  options.multi_gpu=Config.get('architecture', 'multi_gpu')
 
 if Config.has_option('architecture', 'bidir'):
    options.bidir=Config.get('architecture', 'bidir')
    
 if Config.has_option('architecture', 'resnet'):
    options.resnet=Config.get('architecture', 'resnet')
    
 if Config.has_option('architecture', 'act'):
    options.act=Config.get('architecture', 'act')
    
 if Config.has_option('architecture', 'resgate'):
    options.resgate=Config.get('architecture', 'resgate')
    
 if Config.has_option('architecture', 'minimal_gru'):
    options.minimal_gru=Config.get('architecture', 'minimal_gru')
    
 if Config.has_option('architecture', 'skip_conn'):
    options.skip_conn=Config.get('architecture', 'skip_conn')

 if Config.has_option('architecture', 'act_gate'):
    options.act_gate=Config.get('architecture', 'act_gate')
    
 if Config.has_option('architecture', 'use_laynorm'):
    options.use_laynorm=Config.get('architecture', 'use_laynorm')
    
 if Config.has_option('architecture', 'cost'):
    options.cost=Config.get('architecture', 'cost')
    
 if Config.has_option('architecture', 'NN_type'):
    options.NN_type=Config.get('architecture', 'NN_type')
    
 if Config.has_option('architecture', 'twin_reg'):
    options.twin_reg=Config.get('architecture', 'twin_reg')
    
 if Config.has_option('architecture', 'twin_w'):
    options.twin_w=Config.get('architecture', 'twin_w')
    
 if Config.has_option('architecture', 'cnn_pre'):
    options.cnn_pre=Config.get('architecture', 'cnn_pre')
    
 if Config.has_option('architecture', 'seed'):
    options.seed=Config.get('architecture', 'seed')
    
 # Optimization
 if Config.has_option('optimization', 'lr'):
  options.lr=Config.get('optimization', 'lr')
 
 if Config.has_option('optimization', 'batch_size'):
  options.batch_size=Config.get('optimization', 'batch_size')
  
 if Config.has_option('optimization', 'save_gpumem'):
    options.save_gpumem=Config.get('optimization', 'save_gpumem')
 
 if Config.has_option('optimization', 'optimizer'):
  options.optimizer=Config.get('optimization', 'optimizer')

 return options
       
        

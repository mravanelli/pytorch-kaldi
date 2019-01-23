##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import configparser
import sys
import os.path
import random
import subprocess
import numpy as np
import re
import glob
from distutils.util import strtobool
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def run_command(cmd):
    """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        print(line.decode("utf-8"))
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)

def run_shell_display(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    while True:
        out = p.stdout.read(1).decode('utf-8')
        if out == '' and p.poll() != None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()
    return
      
def run_shell(cmd,log_file):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    
    (output, err) = p.communicate()
    p.wait()
    with open(log_file, 'a+') as logfile:
        logfile.write(output.decode("utf-8")+'\n')
        logfile.write(err.decode("utf-8")+'\n')
    
    #print(output.decode("utf-8"))    
    return output


def read_args_command_line(args,config):
    
    sections=[]
    fields=[]
    values=[]

    for i in range(2,len(args)):

        # check if the option is valid for second level
        r2=re.compile('--.*,.*=.*')

        # check if the option is valid for 4 level
        r4=re.compile('--.*,.*,.*,.*=".*"')
        if r2.match(args[i]) is None and r4.match(args[i]) is None:
            sys.stderr.write('ERROR: option \"%s\" from command line is not valid! (the format must be \"--section,field=value\")\n' %(args[i]))
            sys.exit(0)
        
        sections.append(re.search('--(.*),', args[i]).group(1))
        fields.append(re.search(',(.*)', args[i].split('=')[0]).group(1))
        values.append(re.search('=(.*)', args[i]).group(1))

    # parsing command line arguments
    for i in range(len(sections)):

        # Remove multi level is level >= 2
        sections[i] = sections[i].split(',')[0]

        if sections[i] in config.sections():

            # Case of args level > than 2 like --sec,fields,0,field="value"
            if len(fields[i].split(',')) >= 2:

                splitted = fields[i].split(',')

                #Get the actual fields
                field  = splitted[0]
                number = int(splitted[1])
                f_name = splitted[2]
                if field in list(config[sections[i]]):

                    # Get the current string of the corresponding field
                    current_config_field = config[sections[i]][field]

                    # Count the number of occurence of the required field
                    matching = re.findall(f_name+'.', current_config_field)
                    if number >= len(matching):
                        sys.stderr.write('ERROR: the field number \"%s\" provided from command line is not valid, we found \"%s\" \"%s\" field(s) in section \"%s\"!\n' %(number, len(matching), f_name, field ))
                        sys.exit(0)
                    else:
                        
                        # Now replace
                        str_to_be_replaced         = re.findall(f_name+'.*', current_config_field)[number]
                        new_str                    = str(f_name+'='+values[i])
                        replaced                   = nth_replace_string(current_config_field, str_to_be_replaced, new_str, number+1)
                        config[sections[i]][field] = replaced

                else:
                    sys.stderr.write('ERROR: field \"%s\" of section \"%s\" from command line is not valid!")\n' %(field,sections[i]))
                    sys.exit(0)
            else:
                if fields[i] in list(config[sections[i]]):
                    config[sections[i]][fields[i]]=values[i]
                else:
                    sys.stderr.write('ERROR: field \"%s\" of section \"%s\" from command line is not valid!")\n' %(fields[i],sections[i])) 
                    sys.exit(0)
        else:
            sys.stderr.write('ERROR: section \"%s\" from command line is not valid!")\n' %(sections[i]))
            sys.exit(0)
        
    return [sections,fields,values]


def compute_avg_performance(info_lst):
    
    losses=[]
    errors=[]
    times=[]

    for tr_info_file in info_lst:
        config_res = configparser.ConfigParser()
        config_res.read(tr_info_file)
        losses.append(float(config_res['results']['loss']))
        errors.append(float(config_res['results']['err']))
        times.append(float(config_res['results']['elapsed_time']))
        
    loss=np.mean(losses)
    error=np.mean(errors)
    time=np.sum(times)
    
    return [loss,error,time]


def check_field(inp,type_inp,field):

    valid_field=True
    
    if inp=='' and field!='cmd':
       sys.stderr.write("ERROR: The the field  \"%s\" of the config file is empty! \n" % (field))
       valid_field=False
       sys.exit(0)
       
    if type_inp=='path':
        if not(os.path.isfile(inp)) and not(os.path.isdir(inp)) and inp!='none':
            sys.stderr.write("ERROR: The path \"%s\" specified in the field  \"%s\" of the config file does not exists! \n" % (inp,field)) 
            valid_field=False
            sys.exit(0)
      
    if '{' and '}' in type_inp :
        arg_list=type_inp[1:-1].split(',')
        if inp not in arg_list:
            sys.stderr.write("ERROR: The field \"%s\" can only contain %s  arguments \n" % (field,arg_list))
            valid_field=False
            sys.exit(0)
       
    if 'int(' in type_inp:
        try: 
            int(inp)
        except ValueError:
            sys.stderr.write("ERROR: The field \"%s\" can only contain an integer (got \"%s\") \n" % (field,inp))
            valid_field=False
            sys.exit(0)
            
        # Check if the value if within the expected range
        lower_bound=type_inp.split(',')[0][4:]
        upper_bound=type_inp.split(',')[1][:-1]
        
        if lower_bound!="-inf":
            if int(inp)<int(lower_bound):
                sys.stderr.write("ERROR: The field \"%s\" can only contain an integer greater than %s (got \"%s\") \n" % (field,lower_bound,inp))
                valid_field=False
                sys.exit(0)
                
        if upper_bound!="inf":
            if int(inp)>int(upper_bound):
                sys.stderr.write("ERROR: The field \"%s\" can only contain an integer smaller than %s (got \"%s\") \n" % (field,upper_bound,inp))        
                valid_field=False
                sys.exit(0)
    
    if 'float(' in type_inp:
        try: 
            float(inp)
        except ValueError:
            sys.stderr.write("ERROR: The field \"%s\" can only contain a float (got \"%s\") \n" % (field,inp))
            valid_field=False
            sys.exit(0)

        # Check if the value if within the expected range
        lower_bound=type_inp.split(',')[0][6:]
        upper_bound=type_inp.split(',')[1][:-1]

        
        if lower_bound!="-inf":
            if float(inp)<float(lower_bound):
                sys.stderr.write("ERROR: The field \"%s\" can only contain a float greater than %s (got \"%s\") \n" % (field,lower_bound,inp))
                valid_field=False
                sys.exit(0)
                
        if upper_bound!="inf":
            if float(inp)>float(upper_bound):
                sys.stderr.write("ERROR: The field \"%s\" can only contain a float smaller than %s (got \"%s\") \n" % (field,upper_bound,inp))
                valid_field=False
                sys.exit(0)
        
    if type_inp=='bool':
        lst={'True','true','1','False','false','0'}
        if not(inp in lst):
            sys.stderr.write("ERROR: The field \"%s\" can only contain a boolean (got \"%s\") \n" % (field,inp))
            valid_field=False
            sys.exit(0)
            
    if 'int_list(' in type_inp:
        lst=inp.split(',')
        try: 
            list(map(int,lst))
        except ValueError:
            sys.stderr.write("ERROR: The field \"%s\" can only contain a list of integer (got \"%s\").  Make also sure there aren't white spaces between commas.\n" % (field,inp))
            valid_field=False
            sys.exit(0)

        # Check if the value if within the expected range
        lower_bound=type_inp.split(',')[0][9:]
        upper_bound=type_inp.split(',')[1][:-1]

        for elem in lst:
     
            if lower_bound!="-inf":
                if int(elem)<int(lower_bound):
                    sys.stderr.write("ERROR: The field \"%s\" can only contain an integer greater than %s (got \"%s\") \n" % (field,lower_bound,elem))
                    valid_field=False
                    sys.exit(0)
            if upper_bound!="inf":
                if int(elem)>int(upper_bound):
                    sys.stderr.write("ERROR: The field \"%s\" can only contain an integer smaller than %s (got \"%s\") \n" % (field,upper_bound,elem))
                    valid_field=False
                    sys.exit(0)            
            
    if 'float_list(' in type_inp:
        lst=inp.split(',')
        try: 
            list(map(float,lst))
        except ValueError:
            sys.stderr.write("ERROR: The field \"%s\" can only contain a list of floats (got \"%s\"). Make also sure there aren't white spaces between commas. \n" % (field,inp))
            valid_field=False
            sys.exit(0)
        # Check if the value if within the expected range
        lower_bound=type_inp.split(',')[0][11:]
        upper_bound=type_inp.split(',')[1][:-1]

        for elem in lst:
     
            if lower_bound!="-inf":
                if float(elem)<float(lower_bound):
                    sys.stderr.write("ERROR: The field \"%s\" can only contain a float greater than %s (got \"%s\") \n" % (field,lower_bound,elem))
                    valid_field=False
                    sys.exit(0)
                    
            if upper_bound!="inf":
                if float(elem)>float(upper_bound):
                    sys.stderr.write("ERROR: The field \"%s\" can only contain a float smaller than %s (got \"%s\") \n" % (field,upper_bound,elem))
                    valid_field=False
                    sys.exit(0)                  

    if type_inp=='bool_list':
        lst={'True','true','1','False','false','0'}
        inps=inp.split(',')
        for elem in inps:
            if not(elem in lst):
                sys.stderr.write("ERROR: The field \"%s\" can only contain a list of boolean (got \"%s\"). Make also sure there aren't white spaces between commas.\n" % (field,inp))
                valid_field=False
                sys.exit(0)
                    
            
    return  valid_field  



def get_all_archs(config):
    
    arch_lst=[]
    for sec in config.sections():
        if 'architecture' in sec:
            arch_lst.append(sec)
    return arch_lst
            
    
def expand_section(config_proto,config):
    
    
    # expands config_proto with fields in prototype files  
    name_data=[]
    name_arch=[]
    for sec in config.sections():
        if 'dataset' in sec:

            config_proto.add_section(sec)
            config_proto[sec]=config_proto['dataset']
            name_data.append(config[sec]['data_name'])
            
        if 'architecture' in sec:
            name_arch.append(config[sec]['arch_name'])
            config_proto.add_section(sec)
            config_proto[sec]=config_proto['architecture']
            proto_file=config[sec]['arch_proto']
    
            # Reading proto file (architecture)
            config_arch = configparser.ConfigParser()  
            config_arch.read(proto_file)
            
            # Reading proto options
            fields_arch=list(dict(config_arch.items('proto')).keys())
            fields_arch_type=list(dict(config_arch.items('proto')).values())
    
            for i in range(len(fields_arch)):
                config_proto.set(sec,fields_arch[i],fields_arch_type[i])


            # Reading proto file (architecture_optimizer)
            opt_type=config[sec]['arch_opt']
            if opt_type=='sgd':
                proto_file='proto/sgd.proto'
                
            if opt_type=='rmsprop':
                proto_file='proto/rmsprop.proto'
                
            if opt_type=='adam':
                proto_file='proto/adam.proto'
                
            config_arch = configparser.ConfigParser()  
            config_arch.read(proto_file)
            
            # Reading proto options
            fields_arch=list(dict(config_arch.items('proto')).keys())
            fields_arch_type=list(dict(config_arch.items('proto')).values())
    
            for i in range(len(fields_arch)):
                config_proto.set(sec,fields_arch[i],fields_arch_type[i])
                
                
    config_proto.remove_section('dataset')
    config_proto.remove_section('architecture')
    
    return [config_proto,name_data,name_arch]
    
    
def expand_section_proto(config_proto,config):
     
    # Read config proto file
    config_proto_optim_file=config['optimization']['opt_proto']
    config_proto_optim = configparser.ConfigParser()
    config_proto_optim.read(config_proto_optim_file)
    for optim_par in list(config_proto_optim['proto']):
        config_proto.set('optimization',optim_par,config_proto_optim['proto'][optim_par])

    


    
def check_cfg_fields(config_proto,config,cfg_file):
    
    # Check mandatory sections and fields            
    sec_parse=True
       
    for sec in config_proto.sections():
        
        if any(sec in s for s in config.sections()):
            
            # Check fields
            for field in list(dict(config_proto.items(sec)).keys()):
    
                if not(field in config[sec]):
                  sys.stderr.write("ERROR: The confg file %s does not contain the field \"%s=\" in section  \"[%s]\" (mandatory)!\n" % (cfg_file,field,sec))               
                  sec_parse=False
                else:
                    field_type=config_proto[sec][field]
                    if not(check_field(config[sec][field],field_type,field)):           
                        sec_parse=False
                       
                       
    
        # If a mandatory section doesn't exist...
        else:
            sys.stderr.write("ERROR: The confg file %s does not contain \"[%s]\" section (mandatory)!\n" % (cfg_file,sec))
            sec_parse=False
    
    if sec_parse==False:
        sys.stderr.write("ERROR: Revise the confg file %s \n" % (cfg_file))
        sys.exit(0)
    return sec_parse


def check_consistency_with_proto(cfg_file,cfg_file_proto):

    sec_parse=True
    
    # Check if cfg file exists
    try:
      open(cfg_file, 'r')
    except IOError:
       sys.stderr.write("ERROR: The confg file %s does not exist!\n" % (cfg_file))
       sys.exit(0)
       
    # Check if cfg proto  file exists
    try:
      open(cfg_file_proto, 'r')
    except IOError:
       sys.stderr.write("ERROR: The confg file %s does not exist!\n" % (cfg_file_proto))
       sys.exit(0)     
      
    # Parser Initialization    
    config = configparser.ConfigParser()
    
    # Reading the cfg file
    config.read(cfg_file)
    
    # Reading proto cfg file    
    config_proto = configparser.ConfigParser()
    config_proto.read(cfg_file_proto)


    
    # Adding the multiple entries in data and architecture sections 
    [config_proto,name_data,name_arch]=expand_section(config_proto,config)
        
    # Check mandatory sections and fields            
    sec_parse=check_cfg_fields(config_proto,config,cfg_file)
    
    if sec_parse==False:
        sys.exit(0)
        
    return [config_proto,name_data,name_arch]
    
       
    
def check_cfg(cfg_file,config,cfg_file_proto):
 
    # Check consistency between cfg_file and cfg_file_proto    
    [config_proto,name_data,name_arch]=check_consistency_with_proto(cfg_file,cfg_file_proto)

    # Reload data_name because they might be altered by arguments
    name_data=[]
    for sec in config.sections():
        if 'dataset' in sec:
            name_data.append(config[sec]['data_name'])
            
    # check consistency between [data_use] vs [data*]
    sec_parse=True
    data_use_with=[]
    for data in list(dict(config.items('data_use')).values()):
        data_use_with.append(data.split(','))
        
    data_use_with=sum(data_use_with, [])

    if not(set(data_use_with).issubset(name_data)):
        sys.stderr.write("ERROR: in [data_use] you are using a dataset not specified in [dataset*] %s \n" % (cfg_file))
        sec_parse=False
        
    # Parse fea and lab  fields in datasets*
    cnt=0
    fea_names_lst=[]
    lab_names_lst=[]
    for data in name_data:

        # Check for production case 'none' lab name
        [lab_names,_,_]=parse_lab_field(config[cfg_item2sec(config,'data_name',data)]['lab'])
        config['exp']['production']=str('False')
        if lab_names== ["none"] and data == config['data_use']['forward_with']:
            config['exp']['production']=str('True')
            continue
        elif lab_names == ["none"] and data != config['data_use']['forward_with']:
            continue

        [fea_names,fea_lsts,fea_opts,cws_left,cws_right]=parse_fea_field(config[cfg_item2sec(config,'data_name',data)]['fea'])
        [lab_names,lab_folders,lab_opts]=parse_lab_field(config[cfg_item2sec(config,'data_name',data)]['lab'])
        
        fea_names_lst.append(sorted(fea_names))
        lab_names_lst.append(sorted(lab_names))
        
        # Check that fea_name doesn't contain special characters
        for name_features in fea_names_lst[cnt]:
            if not(re.match("^[a-zA-Z0-9]*$", name_features)):
                    sys.stderr.write("ERROR: features names (fea_name=) must contain only letters or numbers (no special characters as \"_,$,..\") \n" )
                    sec_parse=False
                    sys.exit(0) 
            
        if cnt>0:
            if fea_names_lst[cnt-1]!=fea_names_lst[cnt]:
                sys.stderr.write("ERROR: features name (fea_name) must be the same of all the datasets! \n" )
                sec_parse=False
                sys.exit(0) 
            if lab_names_lst[cnt-1]!=lab_names_lst[cnt]:
                sys.stderr.write("ERROR: labels name (lab_name) must be the same of all the datasets! \n" )
                sec_parse=False
                sys.exit(0) 
            
        cnt=cnt+1

    # Create the output folder 
    out_folder=config['exp']['out_folder']

    if not os.path.exists(out_folder) or not(os.path.exists(out_folder+'/exp_files')) :
        os.makedirs(out_folder+'/exp_files')
        
    # Parsing forward field
    model=config['model']['model']
    possible_outs=list(re.findall('(.*)=',model.replace(' ','')))
    forward_out_lst=config['forward']['forward_out'].split(',')
    forward_norm_lst=config['forward']['normalize_with_counts_from'].split(',')
    forward_norm_bool_lst=config['forward']['normalize_posteriors'].split(',')

    lab_lst=list(re.findall('lab_name=(.*)\n',config['dataset1']['lab'].replace(' ','')))
    lab_folders=list(re.findall('lab_folder=(.*)\n',config['dataset1']['lab'].replace(' ','')))
    N_out_lab=['none'] * len(lab_lst)

        
    for i in range(len(forward_out_lst)):

        if forward_out_lst[i] not in possible_outs:
            sys.stderr.write('ERROR: the output \"%s\" in the section \"forward_out\" is not defined in section model)\n' %(forward_out_lst[i]))
            sys.exit(0)

        if strtobool(forward_norm_bool_lst[i]):

            if forward_norm_lst[i] not in lab_lst:
                if not os.path.exists(forward_norm_lst[i]):
                    sys.stderr.write('ERROR: the count_file \"%s\" in the section \"forward_out\" is does not exist)\n' %(forward_norm_lst[i]))
                    sys.exit(0)
                else:
                    # Check if the specified file is in the right format
                    f = open(forward_norm_lst[i],"r")
                    cnts = f.read()
                    if not(bool(re.match("(.*)\[(.*)\]", cnts))):
                        sys.stderr.write('ERROR: the count_file \"%s\" in the section \"forward_out\" is not in the right format)\n' %(forward_norm_lst[i]))
                        
                    
            else:
                # Try to automatically retrieve the config file
                if "ali-to-pdf" in lab_opts[lab_lst.index(forward_norm_lst[i])]:
                    log_file=config['exp']['out_folder']+'/log.log'
                    folder_lab_count=lab_folders[lab_lst.index(forward_norm_lst[i])]
                    cmd="hmm-info "+folder_lab_count+"/final.mdl | awk '/pdfs/{print $4}'"
                    output=run_shell(cmd,log_file)
                    if output.decode().rstrip()=='':
                        sys.stderr.write("ERROR: hmm-info command doesn't exist. Make sure your .bashrc contains the Kaldi paths and correctly exports it.\n")
                        sys.exit(0)

                    N_out=int(output.decode().rstrip())
                    N_out_lab[lab_lst.index(forward_norm_lst[i])]=N_out
                    count_file_path=out_folder+'/exp_files/forward_'+forward_out_lst[i]+'_'+forward_norm_lst[i]+'.count'
                    cmd="analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim="+str(N_out)+" \"ark:ali-to-pdf "+folder_lab_count+"/final.mdl \\\"ark:gunzip -c "+folder_lab_count+"/ali.*.gz |\\\" ark:- |\" "+ count_file_path
                    run_shell(cmd,log_file)
                    forward_norm_lst[i]=count_file_path

                else:
                    sys.stderr.write('ERROR: Not able to automatically retrieve count file for the label \"%s\". Please add a valid count file path in \"normalize_with_counts_from\" or set normalize_posteriors=False \n' %(forward_norm_lst[i]))
                    sys.exit(0)
                    
    # Update the config file with the count_file paths
    config['forward']['normalize_with_counts_from']=",".join(forward_norm_lst)

    
    # When possible replace the pattern "N_out_lab*" with the detected number of output

    for sec in config.sections():
        for field in list(config[sec]):
            for i in range(len(lab_lst)):
                pattern='N_out_'+lab_lst[i]
            
                if pattern in config[sec][field]:
                    if N_out_lab[i]!='none':
                        config[sec][field]=config[sec][field].replace(pattern,str(N_out_lab[i]))
                    else:
                       sys.stderr.write('ERROR: Cannot automatically retrieve the number of output in %s. Please, add manually the number of outputs \n' %(pattern))
                       sys.exit(0)
                       
                       
    # Check the model field
    parse_model_field(cfg_file)

    
    # Create block diagram picture of the model
    create_block_diagram(cfg_file)
    


    if sec_parse==False:
        sys.exit(0)
 
        
    return  [config,name_data,name_arch] 


def cfg_item2sec(config,field,value):
    
    for sec in config.sections():
        if field in list(dict(config.items(sec)).keys()):
            if value in list(dict(config.items(sec)).values()):
                return sec
            
    sys.stderr.write("ERROR: %s=%s not found in config file \n" % (field,value))
    sys.exit(0)
    return -1
        
        
        
def split_chunks(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
    
def create_chunks(config):
    
    # splitting data into chunks (see out_folder/additional_files)
    out_folder=config['exp']['out_folder']
    seed=int(config['exp']['seed'])
    N_ep=int(config['exp']['N_epochs_tr'])    
    
    # Setting the random seed
    random.seed(seed)
    
    # training chunk lists creation    
    tr_data_name=config['data_use']['train_with'].split(',')
    
    # Reading validation feature lists
    for dataset in tr_data_name:
        sec_data=cfg_item2sec(config,'data_name',dataset)
        [fea_names,list_fea,fea_opts,cws_left,cws_right]=parse_fea_field(config[cfg_item2sec(config,'data_name',dataset)]['fea'])

        N_chunks= int(config[sec_data]['N_chunks'])
        
        full_list=[]
        
        for i in range(len(fea_names)):
            full_list.append([line.rstrip('\n')+',' for line in open(list_fea[i])])
            full_list[i]=sorted(full_list[i])
          

        # concatenating all the featues in a single file (useful for shuffling consistently)
        full_list_fea_conc=full_list[0]
        for i in range(1,len(full_list)):  
            full_list_fea_conc=list(map(str.__add__,full_list_fea_conc,full_list[i]))
         
        
        for ep in range(N_ep):
           #  randomize the list
           random.shuffle(full_list_fea_conc)
           tr_chunks_fea=list(split_chunks(full_list_fea_conc,N_chunks))
           tr_chunks_fea.reverse()
    
           for ck in range(N_chunks):
                for i in range(len(fea_names)):
                    
                    tr_chunks_fea_split=[];
                    for snt in tr_chunks_fea[ck]:
                        #print(snt.split(',')[i])
                        tr_chunks_fea_split.append(snt.split(',')[i])
                        
                    output_lst_file=out_folder+'/exp_files/train_'+dataset+'_ep'+format(ep, "03d")+'_ck'+format(ck, "02d")+'_'+fea_names[i]+'.lst'
                    f=open(output_lst_file,'w')
                    tr_chunks_fea_wr=map(lambda x:x+'\n', tr_chunks_fea_split)
                    f.writelines(tr_chunks_fea_wr)
                    f.close()
    
            
    # Validation chunk lists creation    
    valid_data_name=config['data_use']['valid_with'].split(',')
    
    # Reading validation feature lists
    for dataset in valid_data_name:
        sec_data=cfg_item2sec(config,'data_name',dataset)
        [fea_names,list_fea,fea_opts,cws_left,cws_right]=parse_fea_field(config[cfg_item2sec(config,'data_name',dataset)]['fea'])

        N_chunks= int(config[sec_data]['N_chunks'])
        
        full_list=[]
        
        for i in range(len(fea_names)):
            full_list.append([line.rstrip('\n')+',' for line in open(list_fea[i])])
            full_list[i]=sorted(full_list[i])
          

        # concatenating all the featues in a single file (useful for shuffling consistently)
        full_list_fea_conc=full_list[0]
        for i in range(1,len(full_list)):  
            full_list_fea_conc=list(map(str.__add__,full_list_fea_conc,full_list[i]))
         
        # randomize the list
        random.shuffle(full_list_fea_conc)
        valid_chunks_fea=list(split_chunks(full_list_fea_conc,N_chunks))

        
        for ep in range(N_ep):
            for ck in range(N_chunks):
                for i in range(len(fea_names)):
                    
                    valid_chunks_fea_split=[];
                    for snt in valid_chunks_fea[ck]:
                        #print(snt.split(',')[i])
                        valid_chunks_fea_split.append(snt.split(',')[i])
                        
                    output_lst_file=out_folder+'/exp_files/valid_'+dataset+'_ep'+format(ep, "03d")+'_ck'+format(ck, "02d")+'_'+fea_names[i]+'.lst'
                    f=open(output_lst_file,'w')
                    valid_chunks_fea_wr=map(lambda x:x+'\n', valid_chunks_fea_split)
                    f.writelines(valid_chunks_fea_wr)
                    f.close()
                    
                    
    # forward chunk lists creation    
    forward_data_name=config['data_use']['forward_with'].split(',')
    
    # Reading validation feature lists
    for dataset in forward_data_name:
        sec_data=cfg_item2sec(config,'data_name',dataset)
        [fea_names,list_fea,fea_opts,cws_left,cws_right]=parse_fea_field(config[cfg_item2sec(config,'data_name',dataset)]['fea'])

        N_chunks= int(config[sec_data]['N_chunks'])
        
        full_list=[]
        
        for i in range(len(fea_names)):
            full_list.append([line.rstrip('\n')+',' for line in open(list_fea[i])])
            full_list[i]=sorted(full_list[i])
          

        # concatenating all the featues in a single file (useful for shuffling consistently)
        full_list_fea_conc=full_list[0]
        for i in range(1,len(full_list)):  
            full_list_fea_conc=list(map(str.__add__,full_list_fea_conc,full_list[i]))
            
         
        # randomize the list
        random.shuffle(full_list_fea_conc)
        forward_chunks_fea=list(split_chunks(full_list_fea_conc,N_chunks))

        
        for ck in range(N_chunks):
            for i in range(len(fea_names)):
                
                forward_chunks_fea_split=[];
                for snt in forward_chunks_fea[ck]:
                    #print(snt.split(',')[i])
                    forward_chunks_fea_split.append(snt.split(',')[i])
                    
                output_lst_file=out_folder+'/exp_files/forward_'+dataset+'_ep'+format(ep, "03d")+'_ck'+format(ck, "02d")+'_'+fea_names[i]+'.lst'
                f=open(output_lst_file,'w')
                forward_chunks_fea_wr=map(lambda x:x+'\n', forward_chunks_fea_split)
                f.writelines(forward_chunks_fea_wr)
                f.close()  

    
    
def write_cfg_chunk(cfg_file,config_chunk_file,cfg_file_proto_chunk,pt_files,lst_file,info_file,to_do,data_set_name,lr,max_seq_length_train_curr,name_data,ep,ck):
                    
    # writing the chunk-specific cfg file
    config = configparser.ConfigParser()
    config.read(cfg_file)
    
    config_chunk = configparser.ConfigParser()
    config_chunk.read(cfg_file)
    
    # Exp section
    config_chunk['exp']['to_do']=to_do
    config_chunk['exp']['out_info']=info_file
    
    # change seed for randomness
    config_chunk['exp']['seed']=str(int(config_chunk['exp']['seed'])+ep+ck)
    
    for arch in pt_files.keys():
        config_chunk[arch]['arch_pretrain_file']=pt_files[arch]
    
    # writing the current learning rate
    for lr_arch in lr.keys():
        config_chunk[lr_arch ]['arch_lr']=str(lr[lr_arch])
    
    # Data_chunk section
    config_chunk.add_section('data_chunk')

    config_chunk['data_chunk']=config[cfg_item2sec(config,'data_name',data_set_name)]
   

    lst_files=sorted(glob.glob(lst_file))

    current_fea=config_chunk['data_chunk']['fea']
    
    list_current_fea=re.findall('fea_name=(.*)\nfea_lst=(.*)\n', current_fea)
    
    for (fea, path) in list_current_fea:
        for path_cand in lst_files:
            fea_type_cand=re.findall('_(.*).lst', path_cand)[0].split('_')[-1]
            if fea_type_cand==fea:
                config_chunk['data_chunk']['fea']=config_chunk['data_chunk']['fea'].replace(path,path_cand)
                
    
    config_chunk.remove_option('data_chunk','data_name')
    config_chunk.remove_option('data_chunk','N_chunks')
    
    config_chunk.remove_section('decoding')
    config_chunk.remove_section('data_use')
    
    
    for dataset in name_data:
        config_chunk.remove_section(cfg_item2sec(config_chunk,'data_name',dataset))
    
    
    # Create batche section
    config_chunk.remove_option('batches','increase_seq_length_train')
    config_chunk.remove_option('batches','start_seq_len_train')
    config_chunk.remove_option('batches','multply_factor_seq_len_train')
    
    config_chunk['batches']['max_seq_length_train']=str(max_seq_length_train_curr)
    
    # Write cfg_file_chunk
    with open(config_chunk_file, 'w') as configfile:
        config_chunk.write(configfile)
                    
    # Check cfg_file_chunk
    [config_proto_chunk,name_data_ck,name_arch_ck]=check_consistency_with_proto(config_chunk_file,cfg_file_proto_chunk)
    
    
def parse_fea_field(fea):
    
    # Adding the required fields into a list
    fea_names=[]
    fea_lsts=[]
    fea_opts=[]
    cws_left=[]
    cws_right=[]
    
    for line in fea.split('\n'):
        
        line=re.sub(' +',' ',line)
    
        if 'fea_name=' in line:
           fea_names.append(line.split('=')[1]) 
           
        if 'fea_lst=' in line:
            fea_lsts.append(line.split('=')[1]) 
            
        if 'fea_opts=' in line:
            fea_opts.append(line.split('fea_opts=')[1]) 
            
        if 'cw_left=' in line:
            cws_left.append(line.split('=')[1])
            if not(check_field(line.split('=')[1],'int(0,inf)','cw_left')):
                sys.exit(0)
            
        if 'cw_right=' in line:
            cws_right.append(line.split('=')[1])         
            if not(check_field(line.split('=')[1],'int(0,inf)','cw_right')):
                sys.exit(0)  
                
                
    # Check features names
    if not(sorted(fea_names)==sorted(list(set(fea_names)))):
      sys.stderr.write('ERROR fea_names must be different! (got %s)' %(fea_names)) 
      sys.exit(0)
    
    snt_lst=[]
    cnt=0
    
    # Check consistency of feature lists
    for fea_lst in fea_lsts:
         if not(os.path.isfile(fea_lst)):
             sys.stderr.write("ERROR: The path \"%s\" specified in the field  \"fea_lst\" of the config file does not exists! \n" % (fea_lst))
             sys.exit(0)
         else:
             snts = sorted([line.rstrip('\n').split(' ')[0] for line in open(fea_lst)])
             snt_lst.append(snts)
             # Check if all the sentences are present in all the list files 
             if cnt>0:
                 if snt_lst[cnt-1]!=snt_lst[cnt]:
                     sys.stderr.write("ERROR: the files %s in fea_lst contain a different set of sentences! \n" % (fea_lst))
                     sys.exit(0)
             cnt=cnt+1
    return [fea_names,fea_lsts,fea_opts,cws_left,cws_right]


def parse_lab_field(lab):
    
    # Adding the required fields into a list
    lab_names=[]
    lab_folders=[]
    lab_opts=[]
    
    
    for line in lab.split('\n'):
        
        line=re.sub(' +',' ',line)
        
    
    
        if 'lab_name=' in line:
           lab_names.append(line.split('=')[1]) 
           
        if 'lab_folder=' in line:
            lab_folders.append(line.split('=')[1]) 
            
        if 'lab_opts=' in line:
            lab_opts.append(line.split('lab_opts=')[1]) 
            
                
    
               
    # Check features names
    if not(sorted(lab_names)==sorted(list(set(lab_names)))):
      sys.stderr.write('ERROR lab_names must be different! (got %s)' %(lab_names)) 
      sys.exit(0)
    
    
    # Check consistency of feature lists
    for lab_fold in lab_folders:
         if not(os.path.isdir(lab_fold)):
             sys.stderr.write("ERROR: The path \"%s\" specified in the field  \"lab_folder\" of the config file does not exists! \n" % (lab_fold))
             sys.exit(0)
             
    return [lab_names,lab_folders,lab_opts]


def compute_n_chunks(out_folder,data_list,ep,step):
    list_ck=sorted(glob.glob(out_folder+'/exp_files/'+step+'_'+data_list+'_ep'+format(ep, "03d")+'*.lst'))
    last_ck=list_ck[-1]
    N_ck=int(re.findall('_ck(.+)_', last_ck)[-1].split('_')[0])+1
    return N_ck


def parse_model_field(cfg_file):
        
    # Reading the config file
    config = configparser.ConfigParser()
    config.read(cfg_file)
    
    # reading the proto file
    model_proto_file=config['model']['model_proto']
    f = open(model_proto_file,"r")
    proto_model = f.read()
    
    # readiing the model string
    model=config['model']['model']
    
    # Reading fea,lab arch architectures from the cfg file   
    fea_lst=list(re.findall('fea_name=(.*)\n',config['dataset1']['fea'].replace(' ','')))
    lab_lst=list(re.findall('lab_name=(.*)\n',config['dataset1']['lab'].replace(' ','')))
    arch_lst=list(re.findall('arch_name=(.*)\n',open(cfg_file, 'r').read().replace(' ','')))
    possible_operations=re.findall('(.*)\((.*),(.*)\)\n',proto_model)

       
    
    possible_inputs=fea_lst
    model_arch=list(filter(None, model.replace(' ','').split('\n')))
    
    # Reading the model field line by line
    for line in model_arch:
        
        pattern='(.*)=(.*)\((.*),(.*)\)'
        
        if not re.match(pattern,line):
            sys.stderr.write('ERROR: all the entries must be of the following type: output=operation(str,str), got (%s)\n'%(line))
            sys.exit(0)
        else:
            
             # Analyze line and chech if it is compliant with proto_model
             [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])
             inps=[inp1,inp2]
             
             found=False
             for i in range(len(possible_operations)):
                 if operation==possible_operations[i][0]:
                     found=True
    
                     for k in range(1,3):
                         if possible_operations[i][k]=='architecture':
                             if inps[k-1] not in arch_lst:
                                 sys.stderr.write('ERROR: the architecture \"%s\" is not in the architecture lists of the config file (possible architectures are %s)\n' %(inps[k-1],arch_lst))
                                 sys.exit(0)
                                 
                         if possible_operations[i][k]=='label':
                             if inps[k-1] not in lab_lst:
                                 sys.stderr.write('ERROR: the label \"%s\" is not in the label lists of the config file (possible labels are %s)\n' %(inps[k-1],lab_lst))
                                 sys.exit(0)
     
                         if possible_operations[i][k]=='input':
                             if inps[k-1] not in possible_inputs:
                                 sys.stderr.write('ERROR: the input \"%s\" is not defined before (possible inputs are %s)\n' %(inps[k-1],possible_inputs))
                                 sys.exit(0) 
                                 
                         if possible_operations[i][k]=='float':
                             
                             try:
                                 float(inps[k-1])
                             except ValueError:
                                     sys.stderr.write('ERROR: the input \"%s\" must be a float, got %s\n' %(inps[k-1],line))
                                     sys.exit(0) 
                        
                     # Update the list of possible inpus
                     possible_inputs.append(out_name)
                     break
                         
                         
                     
             if found==False:
                 sys.stderr.write(('ERROR: operation \"%s\" does not exists (not defined into the model proto file)\n'%(operation)))
                 sys.exit(0)
                 
    # Check for the mandatory fiels
    if 'loss_final' not in "".join(model_arch):
        sys.stderr.write('ERROR: the variable loss_final should be defined in model\n')
        sys.exit(0)
    
    if 'err_final' not in "".join(model_arch):
        sys.stderr.write('ERROR: the variable err_final should be defined in model\n')
        sys.exit(0)
        
def terminal_node_detection(model_arch,node):
    
    terminal=True
    pattern='(.*)=(.*)\((.*),(.*)\)'
    
    for line in model_arch:
        [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])
        if inp1==node or inp2==node:
            terminal=False
            
    return terminal
    
    
def create_block_connection(lst_inp,model_arch,diag_lines,cnt_names,arch_dict):
  
  if lst_inp==[]:
      return [[],[],diag_lines]
  
  pattern='(.*)=(.*)\((.*),(.*)\)'

  arch_current=[]
  output_conn=[]
  current_inp=[]

   
  for input_element in lst_inp:
           
      
      for l in range(len(model_arch)):
          [out_name,operation,inp1,inp2]=list(re.findall(pattern,model_arch[l])[0])
          
          if inp1==input_element or inp2==input_element:
              if operation=='compute':
                  arch_current.append(inp1)
                  output_conn.append(out_name)
                  current_inp.append(inp2)
                  model_arch[l]='processed'+'='+operation+'('+inp1+',processed)'
              else:
                  arch_current.append(out_name)
                  output_conn.append(out_name)
                  if inp1==input_element:
                     current_inp.append(inp1)
                     model_arch[l]=out_name+'='+operation+'(processed,'+inp2+')'
                     
                  if inp2==input_element:
                     current_inp.append(inp2)
                     model_arch[l]=out_name+'='+operation+'('+inp1+',processed)'
       
    
  for i in range(len(arch_current)):            
       # Create connections           
       diag_lines=diag_lines+str(cnt_names.index(arch_dict[current_inp[i]]))+' -> '+str(cnt_names.index(arch_current[i]))+' [label = "'+current_inp[i]+'"]\n'
   
  #  remove terminal nodes from output list
  output_conn_pruned=[]
  for node in output_conn:
      if not(terminal_node_detection(model_arch,node)):
          output_conn_pruned.append(node)
          
  [arch_current,output_conn,diag_lines]=create_block_connection(output_conn,model_arch,diag_lines,cnt_names,arch_dict)
            
  return [arch_current,output_conn_pruned,diag_lines]

        
def create_block_diagram(cfg_file):
    
    # Reading the config file
    config = configparser.ConfigParser()
    config.read(cfg_file)
    
    # readiing the model string
    model=config['model']['model']
    
    # Reading fea,lab arch architectures from the cfg file   
    pattern='(.*)=(.*)\((.*),(.*)\)'
    
    fea_lst=list(re.findall('fea_name=(.*)\n',config['dataset1']['fea'].replace(' ','')))
    lab_lst=list(re.findall('lab_name=(.*)\n',config['dataset1']['lab'].replace(' ','')))
    arch_lst=list(re.findall('arch_name=(.*)\n',open(cfg_file, 'r').read().replace(' ','')))
    
    
    out_diag_file=config['exp']['out_folder']+'/model.diag'
    
    model_arch=list(filter(None, model.replace(' ','').split('\n')))
    
    
    diag_lines='blockdiag {\n';
    
    cnt=0
    cnt_names=[]
    arch_lst=[]
    fea_lst_used=[]
    lab_lst_used=[]
    
    for line in model_arch:
        if 'err_final=' in line:
            model_arch.remove(line)
            
    # Initializations of the blocks
    for line in model_arch:
     [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])
     
     if operation!='compute':
                    
            # node architecture
            diag_lines=diag_lines+str(cnt)+' [label="'+operation+'",shape = roundedbox];\n'
            arch_lst.append(out_name)
            cnt_names.append(out_name)
            cnt=cnt+1
            
            # labels
            if inp2 in lab_lst:
                diag_lines=diag_lines+str(cnt)+' [label="'+inp2+'",shape = roundedbox];\n'
                if inp2 not in lab_lst_used:
                    lab_lst_used.append(inp2)
                    cnt_names.append(inp2)
                    cnt=cnt+1
                    
            # features
            if inp1 in fea_lst :
                diag_lines=diag_lines+str(cnt)+' [label="'+inp1+'",shape = circle];\n'
                if inp1 not in fea_lst_used:
                    fea_lst_used.append(inp1)
                    cnt_names.append(inp1)
                    cnt=cnt+1
                    
            if inp2 in fea_lst :
                diag_lines=diag_lines+str(cnt)+' [label="'+inp2+'",shape = circle];\n'
                if inp2 not in fea_lst_used:
                    fea_lst_used.append(inp2)
                    cnt_names.append(inp2)
                    cnt=cnt+1
            
            
            
     else:
        # architecture
        diag_lines=diag_lines+str(cnt)+' [label="'+inp1+'",shape = box];\n'
        arch_lst.append(inp1)
        cnt_names.append(inp1)
        cnt=cnt+1
        
        # feature
        if inp2 in fea_lst:
            diag_lines=diag_lines+str(cnt)+' [label="'+inp2+'",shape = circle];\n'
            if inp2 not in fea_lst_used:
                fea_lst_used.append(inp2)
                cnt_names.append(inp2)
                cnt=cnt+1
                   
    
    # Connections across blocks
    lst_conc=fea_lst_used+lab_lst_used
    
    arch_dict={}
    
    for elem in lst_conc:
        arch_dict[elem]=elem
        
    for model_line in model_arch:
        [out_name,operation,inp1,inp2]=list(re.findall(pattern,model_line)[0])
        if operation=='compute':
            arch_dict[out_name]=inp1
        else:
           arch_dict[out_name]=out_name 
     
    
          
    output_conn=lst_conc
    [arch_current,output_conn,diag_lines]=create_block_connection(output_conn,model_arch,diag_lines,cnt_names,arch_dict)
   
               
    diag_lines=diag_lines+'}'


    # Write the diag file describing the model
    with open(out_diag_file, "w") as text_file:
        text_file.write("%s" % diag_lines )
        
    # Create image from the diag file
    log_file=config['exp']['out_folder']+'/log.log'
    cmd='blockdiag -Tsvg '+out_diag_file+' -o '+config['exp']['out_folder']+'/model.svg'
    run_shell(cmd,log_file)
    
    
def list_fea_lab_arch(config): # cancel
    model=config['model']['model'].split('\n')
    fea_lst=list(re.findall('fea_name=(.*)\n',config['data_chunk']['fea'].replace(' ','')))
    lab_lst=list(re.findall('lab_name=(.*)\n',config['data_chunk']['lab'].replace(' ','')))

    
    fea_lst_used=[]
    lab_lst_used=[]
    arch_lst_used=[]
    
    fea_dict_used={}
    lab_dict_used={}
    arch_dict_used={}
    
    fea_lst_used_name=[]
    lab_lst_used_name=[]
    arch_lst_used_name=[]
    
    fea_field=config['data_chunk']['fea']
    lab_field=config['data_chunk']['lab']
    
    pattern='(.*)=(.*)\((.*),(.*)\)'
    
    for line in model:
        [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])
        
        if inp1 in fea_lst and inp1 not in fea_lst_used_name :
            pattern_fea="fea_name="+inp1+"\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            fea_lst_used.append((inp1+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(','))
            fea_dict_used[inp1]=(inp1+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(',')
            
            fea_lst_used_name.append(inp1)
        if inp2 in fea_lst and inp2 not in fea_lst_used_name:
            pattern_fea="fea_name="+inp2+"\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            fea_lst_used.append((inp2+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(','))
            fea_dict_used[inp2]=(inp2+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(',')
            
            fea_lst_used_name.append(inp2)
        if inp1 in lab_lst and inp1 not in lab_lst_used_name:
            pattern_lab="lab_name="+inp1+"\nlab_folder=(.*)\nlab_opts=(.*)"
            lab_lst_used.append((inp1+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(','))
            lab_dict_used[inp1]=(inp1+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(',')
            
            lab_lst_used_name.append(inp1)
            
        if inp2 in lab_lst and inp2 not in lab_lst_used_name:
            pattern_lab="lab_name="+inp2+"\nlab_folder=(.*)\nlab_opts=(.*)"
            lab_lst_used.append((inp2+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(','))
            lab_dict_used[inp2]=(inp2+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(',')
            
            lab_lst_used_name.append(inp2)
            
        if operation=='compute' and inp1 not in arch_lst_used_name:
            arch_id=cfg_item2sec(config,'arch_name',inp1)
            arch_seq_model=strtobool(config[arch_id]['arch_seq_model'])
            arch_lst_used.append([arch_id,inp1,arch_seq_model])
            arch_dict_used[inp1]=[arch_id,inp1,arch_seq_model]
            
            arch_lst_used_name.append(inp1)
            
            
    # convert to unicode (for python 2)
    for i in range(len(fea_lst_used)):
        fea_lst_used[i]=list(map(str, fea_lst_used[i]))
        
    for i in range(len(lab_lst_used)):
        lab_lst_used[i]=list(map(str, lab_lst_used[i])) 
        
    for i in range(len(arch_lst_used)):
        arch_lst_used[i]=list(map(str, arch_lst_used[i]))
     
    return [fea_lst_used,lab_lst_used,arch_lst_used]



def dict_fea_lab_arch(config):
    model=config['model']['model'].split('\n')
    fea_lst=list(re.findall('fea_name=(.*)\n',config['data_chunk']['fea'].replace(' ','')))
    lab_lst=list(re.findall('lab_name=(.*)\n',config['data_chunk']['lab'].replace(' ','')))

    
    fea_lst_used=[]
    lab_lst_used=[]
    arch_lst_used=[]
    
    fea_dict_used={}
    lab_dict_used={}
    arch_dict_used={}
    
    fea_lst_used_name=[]
    lab_lst_used_name=[]
    arch_lst_used_name=[]
    
    fea_field=config['data_chunk']['fea']
    lab_field=config['data_chunk']['lab']
    
    pattern='(.*)=(.*)\((.*),(.*)\)'
    
    for line in model:
        [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])
        
        if inp1 in fea_lst and inp1 not in fea_lst_used_name :
            pattern_fea="fea_name="+inp1+"\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            if sys.version_info[0]==2:
                fea_lst_used.append((inp1+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).encode('utf8').split(','))
                fea_dict_used[inp1]=(inp1+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).encode('utf8').split(',')
            else:
                fea_lst_used.append((inp1+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(','))
                fea_dict_used[inp1]=(inp1+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(',')
            
            fea_lst_used_name.append(inp1)
            
            
        if inp2 in fea_lst and inp2 not in fea_lst_used_name:
            pattern_fea="fea_name="+inp2+"\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            if sys.version_info[0]==2:
                fea_lst_used.append((inp2+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).encode('utf8').split(','))
                fea_dict_used[inp2]=(inp2+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).encode('utf8').split(',')
            else:
                fea_lst_used.append((inp2+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(','))
                fea_dict_used[inp2]=(inp2+","+",".join(list(re.findall(pattern_fea,fea_field)[0]))).split(',')
                
            
            fea_lst_used_name.append(inp2)
        if inp1 in lab_lst and inp1 not in lab_lst_used_name:
            pattern_lab="lab_name="+inp1+"\nlab_folder=(.*)\nlab_opts=(.*)"
            
            if sys.version_info[0]==2:
                lab_lst_used.append((inp1+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).encode('utf8').split(','))
                lab_dict_used[inp1]=(inp1+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).encode('utf8').split(',')
            else:
                lab_lst_used.append((inp1+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(','))
                lab_dict_used[inp1]=(inp1+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(',')
            
            lab_lst_used_name.append(inp1)
            
        if inp2 in lab_lst and inp2 not in lab_lst_used_name:
            pattern_lab="lab_name="+inp2+"\nlab_folder=(.*)\nlab_opts=(.*)"
            
            if sys.version_info[0]==2:
                lab_lst_used.append((inp2+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).encode('utf8').split(','))
                lab_dict_used[inp2]=(inp2+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).encode('utf8').split(',')
            else:
                lab_lst_used.append((inp2+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(','))
                lab_dict_used[inp2]=(inp2+","+",".join(list(re.findall(pattern_lab,lab_field)[0]))).split(',')              
            
            lab_lst_used_name.append(inp2)
            
        if operation=='compute' and inp1 not in arch_lst_used_name:
            arch_id=cfg_item2sec(config,'arch_name',inp1)
            arch_seq_model=strtobool(config[arch_id]['arch_seq_model'])
            arch_lst_used.append([arch_id,inp1,arch_seq_model])
            arch_dict_used[inp1]=[arch_id,inp1,arch_seq_model]
            
            arch_lst_used_name.append(inp1)
            
            
    # convert to unicode (for python 2)
    for i in range(len(fea_lst_used)):
        fea_lst_used[i]=list(map(str, fea_lst_used[i]))
        
    for i in range(len(lab_lst_used)):
        lab_lst_used[i]=list(map(str, lab_lst_used[i])) 
        
    for i in range(len(arch_lst_used)):
        arch_lst_used[i]=list(map(str, arch_lst_used[i]))
     
    return [fea_dict_used,lab_dict_used,arch_dict_used]



def is_sequential(config,arch_lst): # To cancel
    seq_model=False
    
    for [arch_id,arch_name,arch_seq] in arch_lst:
        if strtobool(config[arch_id]['arch_seq_model']):
            seq_model=True
            break
    return seq_model


def is_sequential_dict(config,arch_dict):
    seq_model=False
    
    for arch in arch_dict.keys():
        arch_id=arch_dict[arch][0]
        if strtobool(config[arch_id]['arch_seq_model']):
            seq_model=True
            break
    return seq_model


def compute_cw_max(fea_dict):
    cw_left_arr=[]
    cw_right_arr=[]
    
    for fea in fea_dict.keys():
        cw_left_arr.append(int(fea_dict[fea][3]))
        cw_right_arr.append(int(fea_dict[fea][4]))
    
    cw_left_max=max(cw_left_arr)
    cw_right_max=max(cw_right_arr)
    
    return [cw_left_max,cw_right_max]


def model_init(inp_out_dict,model,config,arch_dict,use_cuda,multi_gpu,to_do):
    
    pattern='(.*)=(.*)\((.*),(.*)\)'
     
    nns={}
    costs={}
       
    for line in model:
        [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])
        
        if operation=='compute':
            
            # computing input dim
            inp_dim=inp_out_dict[inp2][-1]
            
            # import the class
            module = importlib.import_module(config[arch_dict[inp1][0]]['arch_library'])
            nn_class=getattr(module, config[arch_dict[inp1][0]]['arch_class'])
            
            # add use cuda and todo options
            config.set(arch_dict[inp1][0],'use_cuda',config['exp']['use_cuda'])
            config.set(arch_dict[inp1][0],'to_do',config['exp']['to_do'])
            
            arch_freeze_flag=strtobool(config[arch_dict[inp1][0]]['arch_freeze'])

            
            # initialize the neural network
            net=nn_class(config[arch_dict[inp1][0]],inp_dim)
    
    
            
            if use_cuda:
                net.cuda()
                if multi_gpu:
                    net = nn.DataParallel(net)
                    
            
            if to_do=='train':
                if not(arch_freeze_flag):
                    net.train()
                else:
                   # Switch to eval modality if architecture is frozen (mainly for batch_norm/dropout functions)
                   net.eval() 
            else:
                net.eval()
    
            
            # addigng nn into the nns dict
            nns[arch_dict[inp1][1]]=net
            
            if multi_gpu:
                out_dim=net.module.out_dim
            else:
                out_dim=net.out_dim
                
            # updating output dim
            inp_out_dict[out_name]=[out_dim]
            
        if operation=='concatenate':
            
            inp_dim1=inp_out_dict[inp1][-1]
            inp_dim2=inp_out_dict[inp2][-1]
            
            inp_out_dict[out_name]=[inp_dim1+inp_dim2]
        
        if operation=='cost_nll':
            costs[out_name] = nn.NLLLoss()
            inp_out_dict[out_name]=[1]
            
            
        if operation=='cost_err':
            inp_out_dict[out_name]=[1]
            
        if operation=='mult' or operation=='sum' or operation=='mult_constant' or operation=='sum_constant' or operation=='avg' or operation=='mse':
            inp_out_dict[out_name]=inp_out_dict[inp1]    

    return [nns,costs]



def optimizer_init(nns,config,arch_dict):
    
    # optimizer init
    optimizers={}
    for net in nns.keys():
        
        lr=float(config[arch_dict[net][0]]['arch_lr'])
        
        if config[arch_dict[net][0]]['arch_opt']=='sgd':
            
            opt_momentum=float(config[arch_dict[net][0]]['opt_momentum'])
            opt_weight_decay=float(config[arch_dict[net][0]]['opt_weight_decay'])
            opt_dampening=float(config[arch_dict[net][0]]['opt_dampening'])
            opt_nesterov=strtobool(config[arch_dict[net][0]]['opt_nesterov'])
            
            
            optimizers[net]=(optim.SGD(nns[net].parameters(),
                      lr=lr,
                      momentum=opt_momentum,
                      weight_decay=opt_weight_decay,
                      dampening=opt_dampening,
                      nesterov=opt_nesterov))
            
    
        if config[arch_dict[net][0]]['arch_opt']=='adam':
            
            opt_betas=list(map(float,(config[arch_dict[net][0]]['opt_betas'].split(','))))
            opt_eps=float(config[arch_dict[net][0]]['opt_eps'])
            opt_weight_decay=float(config[arch_dict[net][0]]['opt_weight_decay'])
            opt_amsgrad=strtobool(config[arch_dict[net][0]]['opt_amsgrad'])
            
            optimizers[net]=(optim.Adam(nns[net].parameters(),
                      lr=lr,
                      betas=opt_betas,
                      eps=opt_eps,
                      weight_decay=opt_weight_decay,
                      amsgrad=opt_amsgrad))
    
            
            
            
        if config[arch_dict[net][0]]['arch_opt']=='rmsprop':
            
            opt_momentum=float(config[arch_dict[net][0]]['opt_momentum'])
            opt_alpha=float(config[arch_dict[net][0]]['opt_alpha'])
            opt_eps=float(config[arch_dict[net][0]]['opt_eps'])
            opt_centered=strtobool(config[arch_dict[net][0]]['opt_centered'])
            opt_weight_decay=float(config[arch_dict[net][0]]['opt_weight_decay'])
            
            
            optimizers[net]=(optim.RMSprop(nns[net].parameters(),
                      lr=lr,
                      momentum=opt_momentum,
                      alpha=opt_alpha,
                      eps=opt_eps,
                      centered=opt_centered,
                      weight_decay=opt_weight_decay))
            
    return optimizers



def forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs):
    
    # Forward Step
    outs_dict={}
    pattern='(.*)=(.*)\((.*),(.*)\)'
    
    # adding input features to out_dict:
    for fea in fea_dict.keys():
        if len(inp.shape)==3 and len(fea_dict[fea])>1:
            outs_dict[fea]=inp[:,:,fea_dict[fea][5]:fea_dict[fea][6]]

        if len(inp.shape)==2 and len(fea_dict[fea])>1:
            outs_dict[fea]=inp[:,fea_dict[fea][5]:fea_dict[fea][6]]

    
    
    for line in model:
        [out_name,operation,inp1,inp2]=list(re.findall(pattern,line)[0])

        if operation=='compute':
            
            if len(inp_out_dict[inp2])>1: # if it is an input feature
                
                # Selection of the right feature in the inp tensor
                if len(inp.shape)==3:
                    inp_dnn=inp[:,:,inp_out_dict[inp2][-3]:inp_out_dict[inp2][-2]]
                    if not(bool(arch_dict[inp1][2])):
                        inp_dnn=inp_dnn.view(max_len*batch_size,-1)
                                 
                if len(inp.shape)==2:
                    inp_dnn=inp[:,inp_out_dict[inp2][-3]:inp_out_dict[inp2][-2]]
                    if bool(arch_dict[inp1][2]):
                        inp_dnn=inp_dnn.view(max_len,batch_size,-1)
                    
                outs_dict[out_name]=nns[inp1](inp_dnn)

                
            else:
                if not(bool(arch_dict[inp1][2])) and len(outs_dict[inp2].shape)==3:
                    outs_dict[inp2]=outs_dict[inp2].view(max_len*batch_size,-1)
                    
                if bool(arch_dict[inp1][2]) and len(outs_dict[inp2].shape)==2:
                    outs_dict[inp2]=outs_dict[inp2].view(max_len,batch_size,-1)
                    
                outs_dict[out_name]=nns[inp1](outs_dict[inp2])
                
            if to_do=='forward' and out_name==forward_outs[-1]:
                break

        
        if operation=='cost_nll':
            
            # Put labels in the right format
            if len(inp.shape)==3:
                lab_dnn=inp[:,:,lab_dict[inp2][3]]
            if len(inp.shape)==2:
                lab_dnn=inp[:,lab_dict[inp2][3]]
            
            lab_dnn=lab_dnn.view(-1).long()
            
            # put output in the right format
            out=outs_dict[inp1]

            
            if len(out.shape)==3:
                out=out.view(max_len*batch_size,-1)
            
            if to_do!='forward':
                outs_dict[out_name]=costs[out_name](out, lab_dnn)
            
            
        if operation=='cost_err':

            if len(inp.shape)==3:
                lab_dnn=inp[:,:,lab_dict[inp2][3]]
            if len(inp.shape)==2:
                lab_dnn=inp[:,lab_dict[inp2][3]]
            
            lab_dnn=lab_dnn.view(-1).long()
            
            # put output in the right format
            out=outs_dict[inp1]
            
            if len(out.shape)==3:
                out=out.view(max_len*batch_size,-1)
            
            if to_do!='forward':
                pred=torch.max(out,dim=1)[1] 
                err = torch.mean((pred!=lab_dnn).float())
                outs_dict[out_name]=err
                #print(err)

        
        if operation=='concatenate':
           dim_conc=len(outs_dict[inp1].shape)-1
           outs_dict[out_name]=torch.cat((outs_dict[inp1],outs_dict[inp2]),dim_conc) #check concat axis
           if to_do=='forward' and out_name==forward_outs[-1]:
                break
            
        if operation=='mult':
            outs_dict[out_name]=outs_dict[inp1]*outs_dict[inp2]
            if to_do=='forward' and out_name==forward_outs[-1]:
                break
 
        if operation=='sum':
            outs_dict[out_name]=outs_dict[inp1]+outs_dict[inp2] 
            if to_do=='forward' and out_name==forward_outs[-1]:
                break
            
        if operation=='mult_constant':
            outs_dict[out_name]=outs_dict[inp1]*float(inp2)
            if to_do=='forward' and out_name==forward_outs[-1]:
                break
            
        if operation=='sum_constant':
            outs_dict[out_name]=outs_dict[inp1]+float(inp2)
            if to_do=='forward' and out_name==forward_outs[-1]:
                break
            
        if operation=='avg':
            outs_dict[out_name]=(outs_dict[inp1]+outs_dict[inp2])/2
            if to_do=='forward' and out_name==forward_outs[-1]:
                break
            
        if operation=='mse':
            outs_dict[out_name]=torch.mean((outs_dict[inp1] - outs_dict[inp2]) ** 2)
            if to_do=='forward' and out_name==forward_outs[-1]:
                break


            
    return  outs_dict

def dump_epoch_results(res_file_path, ep, tr_data_lst, tr_loss_tot, tr_error_tot, tot_time, valid_data_lst, valid_peformance_dict, lr, N_ep):
    
    #
    # Default terminal line size is 80 characters, try new dispositions to fit this limit
    #

    res_file = open(res_file_path, "a")
    res_file.write('ep=%s tr=%s loss=%s err=%s ' %(format(ep, "03d"),tr_data_lst,format(tr_loss_tot/len(tr_data_lst), "0.3f"),format(tr_error_tot/len(tr_data_lst), "0.3f")))
    print(' ')
    print('----- Summary epoch %s / %s'%(format(ep, "03d"),format(N_ep-1, "03d")))
    print('Training on %s' %(tr_data_lst))
    print('Loss = %s | err = %s '%(format(tr_loss_tot/len(tr_data_lst), "0.3f"),format(tr_error_tot/len(tr_data_lst), "0.3f")))
    print('-----')
    for valid_data in valid_data_lst:
        res_file.write('valid=%s loss=%s err=%s ' %(valid_data,format(valid_peformance_dict[valid_data][0], "0.3f"),format(valid_peformance_dict[valid_data][1], "0.3f")))
        print('Validating on %s' %(valid_data))
        print('Loss = %s | err = %s '%(format(valid_peformance_dict[valid_data][0], "0.3f"),format(valid_peformance_dict[valid_data][1], "0.3f")))

    print('-----')
    for lr_arch in lr.keys():
        res_file.write('lr_%s=%f ' %(lr_arch,lr[lr_arch]))
        print('Learning rate on %s = %f ' %(lr_arch,lr[lr_arch]))
    print('-----')
    res_file.write('time(s)=%i\n' %(int(tot_time)))
    print('Elapsed time (s) = %i\n' %(int(tot_time)))
    print(' ')
    res_file.close()

def progress(count, total, status=''):
  bar_len = 40
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  if count == total-1:
    sys.stdout.write('[%s] %s%s %s\r' % (bar, 100, '%', status))
    sys.stdout.write("\n")
  else:
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
  sys.stdout.flush()  


def export_loss_acc_to_txt(out_folder, N_ep, val_lst):

    if not os.path.exists(out_folder+'/generated_outputs') :
        os.makedirs(out_folder+'/generated_outputs')

    nb_val     = len(val_lst)
    res        = open(out_folder+'/res.res', 'r').readlines()

    tr_loss   = []
    tr_acc    = []
    val_loss  = np.ndarray((nb_val,N_ep))
    val_acc   = np.ndarray((nb_val,N_ep))

    line_cpt = 0
    for i in range(N_ep): 
        splitted = res[i].split(' ')

        # Getting uniq training loss and acc
        tr_loss.append(float(splitted[2].split('=')[1]))
        tr_acc.append(1 - float(splitted[3].split('=')[1]))

        # Getting multiple or uniq val loss and acc
        # +5 to avoird the 6 first columns of the res.res file
        for i in range(nb_val):
            val_loss[i][line_cpt] = float(splitted[(i*3)+5].split('=')[1])
            val_acc[i][line_cpt]  = 1 - float(splitted[(i*3)+6].split('=')[1])

        line_cpt += 1

    # Saving to files
    np.savetxt(out_folder+'/generated_outputs/tr_loss.txt', np.asarray(tr_loss), '%0.3f', delimiter=',')
    np.savetxt(out_folder+'/generated_outputs/tr_acc.txt', np.asarray(tr_acc), '%0.3f', delimiter=',')

    for i in range(nb_val):
        np.savetxt(out_folder+'/generated_outputs/val_'+str(i)+'_loss.txt', val_loss[i], '%0.5f', delimiter=',')
        np.savetxt(out_folder+'/generated_outputs/val_'+str(i)+'_acc.txt', val_acc[i], '%0.5f', delimiter=',')


def create_curves(out_folder, N_ep, val_lst):

    print(' ')
    print('-----')
    print('Generating output files and plots ... ')
    export_loss_acc_to_txt(out_folder, N_ep, val_lst)

    if not os.path.exists(out_folder+'/generated_outputs') :
        sys.stdacc.write('accOR: No results generated please call export_loss_err_to_txt() before')
        sys.exit(0)

    nb_epoch = len(open(out_folder+'/generated_outputs/tr_loss.txt', 'r').readlines())
    x        = np.arange(nb_epoch)
    nb_val   = len(val_lst)

    # Loading train Loss and acc
    tr_loss  = np.loadtxt(out_folder+'/generated_outputs/tr_loss.txt')
    tr_acc   = np.loadtxt(out_folder+'/generated_outputs/tr_acc.txt')
    
    # Loading val loss and acc
    val_loss = []
    val_acc  = []
    for i in range(nb_val):
        val_loss.append(np.loadtxt(out_folder+'/generated_outputs/val_'+str(i)+'_loss.txt'))
        val_acc.append(np.loadtxt(out_folder+'/generated_outputs/val_'+str(i)+'_acc.txt'))

    #
    # LOSS PLOT
    #

    # Getting maximum values
    max_loss = np.amax(tr_loss)
    for i in range(nb_val):
        if np.amax(val_loss[i]) > max_loss:
            max_loss = np.amax(val_loss[i])

    # Plot train loss and acc
    plt.plot(x, tr_loss, label="train_loss")

    # Plot val loss and acc
    for i in range(nb_val):
        plt.plot(x, val_loss[i], label='val_'+str(i)+'_loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Evolution of the loss function')
    plt.axis([0, nb_epoch-1, 0, max_loss+1])
    plt.legend()
    plt.savefig(out_folder+'/generated_outputs/loss.png')

    # Clear plot
    plt.gcf().clear()

    #
    # ACC PLOT
    #

    # Plot train loss and acc
    plt.plot(x, tr_acc, label="train_acc")

    # Plot val loss and acc
    for i in range(nb_val):
        plt.plot(x, val_acc[i], label='val_'+str(i)+'_acc')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Evolution of the accuracy')
    plt.axis([0, nb_epoch-1, 0, 1])
    plt.legend()
    plt.savefig(out_folder+'/generated_outputs/acc.png')

    print('OK')

# Replace the nth pattern in a string
def nth_replace_string(s, sub, repl, nth):
    find = s.find(sub)

    # if find is not p1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != nth:
        # find + 1 means we start at the last match start index + 1
        find = s.find(sub, find + 1)
        i += 1
    # if i  is equal to nth we found nth matches so replace
    if i == nth:
        return s[:find]+repl+s[find + len(sub):]
    return s
    

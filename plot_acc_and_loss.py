##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import sys
import configparser
import os
from utils import create_curves

# Checking arguments
if len(sys.argv) != 2:
    print('ERROR: Please provide only the path of the cfg_file as : python plot_acc_and_loss.py cfg/TIMIT_MLP_mfcc.cfg')

# Checking if the cfg_file exists and loading it
cfg_file=sys.argv[1]
if not(os.path.exists(cfg_file)):
     sys.stderr.write('ERROR: The config file %s does not exist !\n'%(cfg_file))
     sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

# Getting the parameters
valid_data_lst = config['data_use']['valid_with'].split(',')
out_folder     = config['exp']['out_folder']
N_ep=int(config['exp']['N_epochs_tr'])

# Handling call without running run_exp.py before
if not(os.path.exists(out_folder+'res.res')):
     sys.stderr.write('ERROR: Please run the experiment in order to get results to plot first !\n')
     sys.exit(0)

# Creating files and curves
create_curves(out_folder, N_ep, valid_data_lst)
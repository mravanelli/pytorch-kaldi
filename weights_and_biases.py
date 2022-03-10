##########################################################
# pytorch-kaldi-gan v.1.0
# Walter Heymans
# North West University
# 2020
##########################################################

import wandb
import yaml
import os
from sys import exit

def initialize_wandb(project, config, directory, resume, identity = "", name = ""):
    if not (identity == "") and not (name == ""):
        wandb.init(project = project,
                   config = config,
                   dir = directory,
                   id = identity,
                   name = name,
                   resume = resume,
                   reinit = True)
    else:
        wandb.init(project = project,
                   config = config,
                   dir = directory,
                   resume = resume,
                   reinit = True)

def quick_log(key, value, commit = True):
    wandb.log({key: value}, commit = commit)


def load_cfg_dict_from_yaml(cfg_filename):
    f = open(cfg_filename, 'r')
    cfg_yaml = None
    try:
        cfg_yaml = yaml.full_load(f)
    except Exception as e:
        print("Error loading WANDB config file.", e)
        exit(101)
    finally:
        f.close()
    cfg = {}
    for key in cfg_yaml.keys():
        cfg[key] = cfg_yaml[key]['value']
    return cfg


def get_api():
    return wandb.Api()


def get_run_name():
    return wandb.run.name


def get_run_id():
    return wandb.run.id

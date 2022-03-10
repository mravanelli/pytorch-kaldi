##########################################################
# pytorch-kaldi-gan
# Walter Heymans
# North West University
# 2020

# Adapted from:
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################
from __future__ import print_function

import os
import sys
import glob
import configparser
import numpy as np
from utils import (
    check_cfg,
    create_lists,
    create_configs,
    compute_avg_performance,
    read_args_command_line,
    run_shell,
    compute_n_chunks,
    get_all_archs,
    cfg_item2sec,
    dump_epoch_results,
    create_curves,
    change_lr_cfg,
    expand_str_ep,
    do_validation_after_chunk,
    get_val_info_file_path,
    get_val_cfg_file_path,
    get_chunks_after_which_to_validate,
)
from data_io import read_lab_fea_refac01 as read_lab_fea
from shutil import copyfile
from core import read_next_chunk_into_shared_list_with_subprocess, extract_data_from_shared_list, convert_numpy_to_torch
import re
from distutils.util import strtobool
import importlib
import math
import multiprocessing
import weights_and_biases as wandb
import torch

skip_decode = False


def _run_forwarding_in_subprocesses(config):
    use_cuda = strtobool(config["exp"]["use_cuda"])
    if use_cuda:
        return False
    else:
        return True


def _is_first_validation(ep, ck, N_ck_tr, config):
    def _get_nr_of_valid_per_epoch_from_config(config):
        if not "nr_of_valid_per_epoch" in config["exp"]:
            return 1
        return int(config["exp"]["nr_of_valid_per_epoch"])
    
    if ep>0:
        return False
    
    val_chunks = get_chunks_after_which_to_validate(N_ck_tr, _get_nr_of_valid_per_epoch_from_config(config))
    if ck == val_chunks[0]:
        return True

    return False


def _max_nr_of_parallel_forwarding_processes(config):
    if "max_nr_of_parallel_forwarding_processes" in config["forward"]:
        return int(config["forward"]["max_nr_of_parallel_forwarding_processes"])
    return -1


def print_version_info():
    print("")
    print("".center(40, "#"))
    print(" Pytorch-Kaldi-GAN ".center(38, " ").center(40, "#"))
    print(" Walter Heymans ".center(38, " ").center(40, "#"))
    print(" North West University ".center(38, " ").center(40, "#"))
    print(" 2020 ".center(38, " ").center(40, "#"))
    print("".center(38, " ").center(40, "#"))
    print(" Adapted form: ".center(38, " ").center(40, "#"))
    print(" Pytorch-Kaldi v.0.1 ".center(38, " ").center(40, "#"))
    print(" Mirco Ravanelli, Titouan Parcollet ".center(38, " ").center(40, "#"))
    print(" Mila, University of Montreal ".center(38, " ").center(40, "#"))
    print(" October 2018 ".center(38, " ").center(40, "#"))
    print("".center(40, "#"), end="\n\n")


# START OF EXECUTION #
print_version_info()

# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

config_file_name = str(os.path.basename(cfg_file)).replace(".cfg", "")

# Reading and parsing optional arguments from command line (e.g.,--optimization,lr=0.002)
[section_args, field_args, value_args] = read_args_command_line(sys.argv, config)


# Output folder creation
out_folder = config["exp"]["out_folder"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder + "/exp_files")

# Log file path
log_file = config["exp"]["out_folder"] + "/log.log"


# Read, parse, and check the config file
cfg_file_proto = config["cfg_proto"]["cfg_proto"]
[config, name_data, name_arch] = check_cfg(cfg_file, config, cfg_file_proto)


# Read cfg file options
is_production = strtobool(config["exp"]["production"])
cfg_file_proto_chunk = config["cfg_proto"]["cfg_proto_chunk"]

cmd = config["exp"]["cmd"]
N_ep = int(config["exp"]["N_epochs_tr"])
N_ep_str_format = "0" + str(max(math.ceil(np.log10(N_ep)), 1)) + "d"
tr_data_lst = config["data_use"]["train_with"].split(",")
valid_data_lst = config["data_use"]["valid_with"].split(",")
forward_data_lst = config["data_use"]["forward_with"].split(",")
max_seq_length_train = config["batches"]["max_seq_length_train"]
forward_save_files = list(map(strtobool, config["forward"]["save_out_file"].split(",")))


print("- Reading config file......OK!")

# Copy the global cfg file into the output folder
cfg_file = out_folder + "/conf.cfg"

with open(cfg_file, "w") as configfile:
    config.write(configfile)


# Load the run_nn function from core libriary
# The run_nn is a function that process a single chunk of data
run_nn_script = config["exp"]["run_nn_script"].split(".py")[0]
module = importlib.import_module("core")
run_nn = getattr(module, run_nn_script)


# Splitting data into chunks (see out_folder/additional_files)
create_lists(config)

# Writing the config files
create_configs(config)

print("- Chunk creation......OK!\n")

# create res_file
res_file_path = out_folder + "/res.res"
res_file = open(res_file_path, "w")
res_file.close()


# Learning rates and architecture-specific optimization parameters
arch_lst = get_all_archs(config)
lr = {}
auto_lr_annealing = {}
improvement_threshold = {}
halving_factor = {}
pt_files = {}

for arch in arch_lst:
    lr[arch] = expand_str_ep(config[arch]["arch_lr"], "float", N_ep, "|", "*")
    if len(config[arch]["arch_lr"].split("|")) > 1:
        auto_lr_annealing[arch] = False
    else:
        auto_lr_annealing[arch] = True
    improvement_threshold[arch] = float(config[arch]["arch_improvement_threshold"])
    halving_factor[arch] = float(config[arch]["arch_halving_factor"])
    pt_files[arch] = config[arch]["arch_pretrain_file"]


# If production, skip training and forward directly from last saved models
if is_production:
    ep = N_ep - 1
    N_ep = 0
    model_files = {}

    for arch in pt_files.keys():
        model_files[arch] = out_folder + "/exp_files/final_" + arch + ".pkl"


op_counter = 1  # used to dected the next configuration file from the list_chunks.txt

# Reading the ordered list of config file to process
cfg_file_list = [line.rstrip("\n") for line in open(out_folder + "/exp_files/list_chunks.txt")]
cfg_file_list.append(cfg_file_list[-1])


# A variable that tells if the current chunk is the first one that is being processed:
processed_first = True

data_name = []
data_set = []
data_end_index = []
fea_dict = []
lab_dict = []
arch_dict = []


if config["gan"]["arch_gan"] == "True":
    gan_on = True

    # Checking directories
    directory_g = os.path.join(out_folder, config["gan"]["output_path_g"])
    directory_d = os.path.join(out_folder, config["gan"]["output_path_d"])

    gan_dir = os.path.dirname(directory_g)

    if not os.path.exists(gan_dir):
        os.mkdir(gan_dir)

    if not os.path.exists(gan_dir + "/images"):
        os.mkdir(gan_dir + "/images")

    try:
        if str(config["generator"]["pretrained_file"]) != "none":
            if os.path.exists(str(config["generator"]["pretrained_file"])):
                copyfile(str(config["generator"]["pretrained_file"]), directory_g)
                print("Loaded pretrained G.")
    except KeyError:
        pass

    try:
        if str(config["discriminator"]["pretrained_file"]) != "none":
            if os.path.exists(str(config["discriminator"]["pretrained_file"])):
                copyfile(str(config["discriminator"]["pretrained_file"]), directory_d)
                print("Loaded pretrained D.")
    except KeyError:
        pass
else:
    gan_on = False


def print_settings():
    print_width = 72
    print(" SETTINGS ".center(print_width, "="))
    print("# Epochs:\t\t", N_ep)
    print("# Batch size:\t\t", int(config["batches"]["batch_size_train"]))
    print("# Seed:\t\t\t", int(config["exp"]["seed"]))
    print("# Weights and Biases:\t", str(config["wandb"]["wandb"]))
    print("# GAN training:\t\t", str(config["gan"]["arch_gan"]))

    print("")
    print(" Acoustic Model settings ".center(print_width, "-"))
    print("# Name:\t\t\t", str(config["architecture1"]["arch_name"]))
    print("# Learning rate:\t", float(config["architecture1"]["arch_lr"]))
    print("# Halving factor:\t", float(config["architecture1"]["arch_halving_factor"]))
    print("# Improvement threshold:", float(config["architecture1"]["arch_improvement_threshold"]))
    print("# Optimizer:\t\t", str(config["architecture1"]["arch_opt"]))

    try:
        if config["gan"]["double_features"] == "True":
            print("# Double features:\t", config["gan"]["double_features"])
    except KeyError:
        pass

    if gan_on:
        print("")
        print(" Generator Architecture ".center(print_width, "-"))
        print("# Name:\t\t\t", str(config["generator"]["arch_name"]))

    print("=".center(print_width, "="), end = "\n\n")


print_settings()

if str(config["wandb"]["wandb"]) == "True":
    wandb_cfg = wandb.load_cfg_dict_from_yaml(str(config["wandb"]["config"]))

    # UPDATE config file if Weights and Biases file is different
    wandb_cfg["max_epochs"] = int(config["exp"]["N_epochs_tr"])
    wandb_cfg["seed"] = int(config["exp"]["seed"])
    wandb_cfg["batch_size"] = int(config["batches"]["batch_size_train"])
    wandb_cfg["lr"] = float(config["architecture1"]["arch_lr"])
    wandb_cfg["gan_on"] = config["gan"]["arch_gan"]

    wandb_details = os.path.join(out_folder, "wandb_details.txt")

    if not os.path.exists(wandb_details):
        wandb_details_file = open(wandb_details, "w")

        wandb.initialize_wandb(project = str(config["wandb"]["project"]),
                               config = wandb_cfg,
                               directory = out_folder,
                               resume = False)

        try:
            wandb_details_file.write(wandb.get_run_id() + '\n')
            wandb_details_file.write(wandb.get_run_name())
        except TypeError:
            pass

        wandb_details_file.close()
    else:
        wandb_details_file = open(wandb_details, "r")
        file_content = wandb_details_file.read().splitlines()

        try:
            wandb_run_id = file_content[0]
            wandb_run_name = file_content[1]
        except IndexError:
            wandb_run_id = ""
            wandb_run_name = ""
            pass

        wandb_details_file.close()

        if not wandb_run_id == "":
            wandb.initialize_wandb(project = str(config["wandb"]["project"]),
                                   config = wandb_cfg,
                                   directory = out_folder,
                                   resume = True,
                                   identity = wandb_run_id,
                                   name = wandb_run_name)
        else:
            wandb.initialize_wandb(project = str(config["wandb"]["project"]),
                                   config = wandb_cfg,
                                   directory = out_folder,
                                   resume = True)

    if str(config["wandb"]["decode_only"]) == "True":
        wandb_decode_only = True
        wandb_on = False
    else:
        wandb_on = True
        wandb_decode_only = False
        wandb.quick_log("status", "training", commit = False)
else:
    wandb_on = False
    wandb_decode_only = False

create_gan_dataset = False
try:
    if config["ganset"]["create_set"] == "True":
        create_gan_dataset = True
        print("\nGAN dataset will be created.\n")

        # Output folder creation
        gan_out_folder = config["ganset"]["out_folder"]
        if not os.path.exists(gan_out_folder):
            os.makedirs(gan_out_folder)

except KeyError:
    pass

fine_tuning = True
try:
    if config["exp"]["fine_tuning"] == "False":
        fine_tuning = False
except KeyError:
    pass

# --------TRAINING LOOP--------#
for ep in range(N_ep):

    if wandb_on:
        wandb.quick_log("epoch", ep + 1, commit = False)

    processed_first = True

    tr_loss_tot = 0
    tr_error_tot = 0
    tr_time_tot = 0
    val_time_tot = 0

    print(
        "------------------------------ Epoch %s / %s ------------------------------"
        % (format(ep + 1, N_ep_str_format), format(N_ep, N_ep_str_format))
    )

    for tr_data in tr_data_lst:

        # Compute the total number of chunks for each training epoch
        N_ck_tr = compute_n_chunks(out_folder, tr_data, ep, N_ep_str_format, "train")
        N_ck_str_format = "0" + str(max(math.ceil(np.log10(N_ck_tr)), 1)) + "d"

        # ***Epoch training***
        for ck in range(N_ck_tr):
            if not fine_tuning and ck > 1:
                break
            # Get training time per chunk
            import time
            starting_time = time.time()
            print_chunk_time = False

            if wandb_on:
                wandb.quick_log("chunk", ck + 1, commit = True)

            # paths of the output files (info,model,chunk_specific cfg file)
            info_file = (
                out_folder
                + "/exp_files/train_"
                + tr_data
                + "_ep"
                + format(ep, N_ep_str_format)
                + "_ck"
                + format(ck, N_ck_str_format)
                + ".info"
            )

            if ep + ck == 0:
                model_files_past = {}
            else:
                model_files_past = model_files

            model_files = {}
            for arch in pt_files.keys():
                model_files[arch] = info_file.replace(".info", "_" + arch + ".pkl")

            config_chunk_file = (
                out_folder
                + "/exp_files/train_"
                + tr_data
                + "_ep"
                + format(ep, N_ep_str_format)
                + "_ck"
                + format(ck, N_ck_str_format)
                + ".cfg"
            )

            # update learning rate in the cfg file (if needed)
            change_lr_cfg(config_chunk_file, lr, ep)

            # if this chunk has not already been processed, do training...
            if not (os.path.exists(info_file)):
                print_chunk_time = True

                print("Training %s chunk = %i / %i" % (tr_data, ck + 1, N_ck_tr))

                # getting the next chunk
                next_config_file = cfg_file_list[op_counter]

                [data_name, data_set, data_end_index, fea_dict, lab_dict, arch_dict] = run_nn(
                    data_name,
                    data_set,
                    data_end_index,
                    fea_dict,
                    lab_dict,
                    arch_dict,
                    config_chunk_file,
                    processed_first,
                    next_config_file,
                    wandb_on = wandb_on,
                    epoch = ep,
                    chunk = ck + 1
                )

                # update the first_processed variable
                processed_first = False

                if not (os.path.exists(info_file)):
                    sys.stderr.write(
                        "ERROR: training epoch %i, chunk %i not done! File %s does not exist.\nSee %s \n"
                        % (ep, ck, info_file, log_file)
                    )
                    sys.exit(0)

            # update the operation counter
            op_counter += 1

            # update pt_file (used to initialized the DNN for the next chunk)
            for pt_arch in pt_files.keys():
                pt_files[pt_arch] = (
                    out_folder
                    + "/exp_files/train_"
                    + tr_data
                    + "_ep"
                    + format(ep, N_ep_str_format)
                    + "_ck"
                    + format(ck, N_ck_str_format)
                    + "_"
                    + pt_arch
                    + ".pkl"
                )

            # remove previous pkl files
            if len(model_files_past.keys()) > 0:
                for pt_arch in pt_files.keys():
                    if os.path.exists(model_files_past[pt_arch]):
                        os.remove(model_files_past[pt_arch])

            if do_validation_after_chunk(ck, N_ck_tr, config) and (tr_data == tr_data_lst[-1]) and not(create_gan_dataset):
                if not _is_first_validation(ep,ck, N_ck_tr, config):
                    valid_peformance_dict_prev = valid_peformance_dict
                valid_peformance_dict = {}
                for valid_data in valid_data_lst:
                    N_ck_valid = compute_n_chunks(out_folder, valid_data, ep, N_ep_str_format, "valid")
                    N_ck_str_format_val = "0" + str(max(math.ceil(np.log10(N_ck_valid)), 1)) + "d"
                    for ck_val in range(N_ck_valid):
                        info_file = get_val_info_file_path(
                            out_folder,
                            valid_data,
                            ep,
                            ck,
                            ck_val,
                            N_ep_str_format,
                            N_ck_str_format,
                            N_ck_str_format_val,
                        )
                        config_chunk_file = get_val_cfg_file_path(
                            out_folder,
                            valid_data,
                            ep,
                            ck,
                            ck_val,
                            N_ep_str_format,
                            N_ck_str_format,
                            N_ck_str_format_val,
                        )
                        if not (os.path.exists(info_file)):
                            print("Validating %s chunk = %i / %i" % (valid_data, ck_val + 1, N_ck_valid))
                            next_config_file = cfg_file_list[op_counter]

                            data_name, data_set, data_end_index, fea_dict, lab_dict, arch_dict = run_nn(
                                data_name,
                                data_set,
                                data_end_index,
                                fea_dict,
                                lab_dict,
                                arch_dict,
                                config_chunk_file,
                                processed_first,
                                next_config_file,
                                wandb_on = wandb_on,
                            )
                            processed_first = False
                            if not (os.path.exists(info_file)):
                                sys.stderr.write(
                                    "ERROR: validation on epoch %i, chunk %i, valid chunk %i of dataset %s not done! File %s does not exist.\nSee %s \n"
                                    % (ep, ck, ck_val, valid_data, info_file, log_file)
                                )
                                sys.exit(0)
                        op_counter += 1
                    valid_info_lst = sorted(
                        glob.glob(
                            get_val_info_file_path(
                                out_folder,
                                valid_data,
                                ep,
                                ck,
                                None,
                                N_ep_str_format,
                                N_ck_str_format,
                                N_ck_str_format_val,
                            )
                        )
                    )
                    valid_loss, valid_error, valid_time = compute_avg_performance(valid_info_lst)
                    valid_peformance_dict[valid_data] = [valid_loss, valid_error, valid_time]
                    val_time_tot += valid_time
                if not _is_first_validation(ep,ck, N_ck_tr, config):
                    err_valid_mean = np.mean(np.asarray(list(valid_peformance_dict.values()))[:, 1])
                    err_valid_mean_prev = np.mean(np.asarray(list(valid_peformance_dict_prev.values()))[:, 1])
                    for lr_arch in lr.keys():
                        if ep < N_ep - 1 and auto_lr_annealing[lr_arch]:
                            if ((err_valid_mean_prev - err_valid_mean) / err_valid_mean) < improvement_threshold[
                                lr_arch
                            ]:
                                new_lr_value = float(lr[lr_arch][ep]) * halving_factor[lr_arch]
                                for i in range(ep + 1, N_ep):
                                    lr[lr_arch][i] = str(new_lr_value)

            ending_time = time.time()

            if print_chunk_time:
                chunk_time = ending_time - starting_time
                print("Chunk time:", round(chunk_time), "s\n")
                if wandb_on:
                    wandb.quick_log("chunk_time", chunk_time, commit=False)

        # Training Loss and Error
        tr_info_lst = sorted(
            glob.glob(out_folder + "/exp_files/train_" + tr_data + "_ep" + format(ep, N_ep_str_format) + "*.info")
        )
        [tr_loss, tr_error, tr_time] = compute_avg_performance(tr_info_lst)

        tr_loss_tot = tr_loss_tot + tr_loss
        tr_error_tot = tr_error_tot + tr_error
        tr_time_tot = tr_time_tot + tr_time
        tot_time = tr_time + val_time_tot

    if not create_gan_dataset:
        if fine_tuning:
            # Print results in both res_file and stdout
            dump_epoch_results(
                res_file_path,
                ep,
                tr_data_lst,
                tr_loss_tot,
                tr_error_tot,
                tot_time,
                valid_data_lst,
                valid_peformance_dict,
                lr,
                N_ep,
            )

    if wandb_on:
        for lr_arch in lr.keys():
            wandb.quick_log("learning_rate", float(lr[lr_arch][ep]), commit = False)

        for valid_data in valid_data_lst:
            wandb.quick_log("valid_loss_" + str(valid_data), float(valid_peformance_dict[valid_data][0]), commit = False)
            wandb.quick_log("valid_error_" + str(valid_data), float(valid_peformance_dict[valid_data][1]), commit = False)

# Training has ended, copy the last .pkl to final_arch.pkl for production
for pt_arch in pt_files.keys():
    if os.path.exists(model_files[pt_arch]) and not os.path.exists(out_folder + "/exp_files/final_" + pt_arch + ".pkl"):
        copyfile(model_files[pt_arch], out_folder + "/exp_files/final_" + pt_arch + ".pkl")

# Terminate application if GAN dataset creation is set
try:
    if config["ganset"]["create_set"] == "True":
        print("\nGAN dataset created!")
        exit()
except KeyError:
    pass

# --------FORWARD--------#
if wandb_on or wandb_decode_only:
    wandb.quick_log("status", "forwarding", commit = True)

for forward_data in forward_data_lst:

    # Compute the number of chunks
    N_ck_forward = compute_n_chunks(out_folder, forward_data, ep, N_ep_str_format, "forward")
    N_ck_str_format = "0" + str(max(math.ceil(np.log10(N_ck_forward)), 1)) + "d"

    processes = list()
    info_files = list()
    for ck in range(N_ck_forward):

        if not is_production:
            print("Testing %s chunk = %i / %i" % (forward_data, ck + 1, N_ck_forward))
        else:
            print("Forwarding %s chunk = %i / %i" % (forward_data, ck + 1, N_ck_forward))

        # output file
        info_file = (
            out_folder
            + "/exp_files/forward_"
            + forward_data
            + "_ep"
            + format(ep, N_ep_str_format)
            + "_ck"
            + format(ck, N_ck_str_format)
            + ".info"
        )
        config_chunk_file = (
            out_folder
            + "/exp_files/forward_"
            + forward_data
            + "_ep"
            + format(ep, N_ep_str_format)
            + "_ck"
            + format(ck, N_ck_str_format)
            + ".cfg"
        )

        # Do forward if the chunk was not already processed
        if not (os.path.exists(info_file)):

            # Doing forward

            # getting the next chunk
            next_config_file = cfg_file_list[op_counter]

            # run chunk processing
            if _run_forwarding_in_subprocesses(config):
                shared_list = list()
                output_folder = config["exp"]["out_folder"]
                save_gpumem = strtobool(config["exp"]["save_gpumem"])
                use_cuda = strtobool(config["exp"]["use_cuda"])
                p = read_next_chunk_into_shared_list_with_subprocess(
                    read_lab_fea, shared_list, config_chunk_file, is_production, output_folder, wait_for_process=True
                )
                data_name, data_end_index_fea, data_end_index_lab, fea_dict, lab_dict, arch_dict, data_set_dict = extract_data_from_shared_list(
                    shared_list
                )
                data_set_inp, data_set_ref = convert_numpy_to_torch(data_set_dict, save_gpumem, use_cuda)
                data_set = {"input": data_set_inp, "ref": data_set_ref}
                data_end_index = {"fea": data_end_index_fea, "lab": data_end_index_lab}
                p = multiprocessing.Process(
                    target=run_nn,
                    kwargs={
                        "data_name": data_name,
                        "data_set": data_set,
                        "data_end_index": data_end_index,
                        "fea_dict": fea_dict,
                        "lab_dict": lab_dict,
                        "arch_dict": arch_dict,
                        "cfg_file": config_chunk_file,
                        "processed_first": False,
                        "next_config_file": None,
                    },
                )
                processes.append(p)
                if _max_nr_of_parallel_forwarding_processes(config) != -1 and len(
                    processes
                ) > _max_nr_of_parallel_forwarding_processes(config):
                    processes[0].join()
                    del processes[0]
                p.start()
            else:
                [data_name, data_set, data_end_index, fea_dict, lab_dict, arch_dict] = run_nn(
                    data_name,
                    data_set,
                    data_end_index,
                    fea_dict,
                    lab_dict,
                    arch_dict,
                    config_chunk_file,
                    processed_first,
                    next_config_file,
                    wandb_on = wandb_on,
                )
                processed_first = False
                if not (os.path.exists(info_file)):
                    sys.stderr.write(
                        "ERROR: forward chunk %i of dataset %s not done! File %s does not exist.\nSee %s \n"
                        % (ck, forward_data, info_file, log_file)
                    )
                    sys.exit(0)

            info_files.append(info_file)

        # update the operation counter
        op_counter += 1
    if _run_forwarding_in_subprocesses(config):
        for process in processes:
            process.join()
        for info_file in info_files:
            if not (os.path.exists(info_file)):
                sys.stderr.write(
                    "ERROR: File %s does not exist. Forwarding did not suceed.\nSee %s \n" % (info_file, log_file)
                )
                sys.exit(0)


# --------DECODING--------#
if wandb_on or wandb_decode_only:
    wandb.quick_log("status", "decoding", commit = True)

dec_lst = glob.glob(out_folder + "/exp_files/*_to_decode.ark")

forward_data_lst = config["data_use"]["forward_with"].split(",")
forward_outs = config["forward"]["forward_out"].split(",")
forward_dec_outs = list(map(strtobool, config["forward"]["require_decoding"].split(",")))


def get_wer_stats(word_error_rate_string):
    wer_stats = word_error_rate_string.split(" ")
    word_error_rate = float(wer_stats[1])
    word_tot = wer_stats[5]
    word_tot = int(word_tot.replace(",", ""))
    word_ins = int(wer_stats[6])
    word_del = int(wer_stats[8])
    word_sub = int(wer_stats[10])
    return word_error_rate, word_tot, word_ins, word_del, word_sub


def get_unique_filename(results_file_name):
    file_unique_var = False

    if not os.path.exists(results_file_name):  # File does not exist yet
        return results_file_name

    # File does exist, determine number to append

    results_file_name = results_file_name.replace(".txt", "")   # no number added

    file_number = 1

    while not file_unique_var:
        temp_filename = results_file_name + "__" + str(file_number) + ".txt"

        if not os.path.exists(temp_filename):
            file_unique_var = True
            results_file_name = temp_filename
        else:
            file_number += 1

    return results_file_name


def store_wer_stats(run_name, dataset, word_error_rate_string):
    if not os.path.exists("results"):
        os.makedirs("results")

    results_file_name = "results/" + config["exp"]["dataset_name"] + "__" + dataset + "__" + run_name + ".txt"
    results_file_name = get_unique_filename(results_file_name)
    results_file = open(results_file_name, "w")

    results_file.write(word_error_rate_string)
    results_file.close()

if skip_decode:
    exit(0)

for data in forward_data_lst:
    for k in range(len(forward_outs)):
        if forward_dec_outs[k]:

            print("Decoding %s output %s" % (data, forward_outs[k]))

            info_file = out_folder + "/exp_files/decoding_" + data + "_" + forward_outs[k] + ".info"

            # create decode config file
            config_dec_file = out_folder + "/decoding_" + data + "_" + forward_outs[k] + ".conf"
            config_dec = configparser.ConfigParser()
            config_dec.add_section("decoding")

            for dec_key in config["decoding"].keys():
                config_dec.set("decoding", dec_key, config["decoding"][dec_key])

            # add graph_dir, datadir, alidir
            lab_field = config[cfg_item2sec(config, "data_name", data)]["lab"]

            # Production case, we don't have labels
            if not is_production:
                pattern = "lab_folder=(.*)\nlab_opts=(.*)\nlab_count_file=(.*)\nlab_data_folder=(.*)\nlab_graph=(.*)"
                alidir = re.findall(pattern, lab_field)[0][0]
                config_dec.set("decoding", "alidir", os.path.abspath(alidir))

                datadir = re.findall(pattern, lab_field)[0][3]
                config_dec.set("decoding", "data", os.path.abspath(datadir))

                graphdir = re.findall(pattern, lab_field)[0][4]
                config_dec.set("decoding", "graphdir", os.path.abspath(graphdir))
            else:
                pattern = "lab_data_folder=(.*)\nlab_graph=(.*)"
                datadir = re.findall(pattern, lab_field)[0][0]
                config_dec.set("decoding", "data", os.path.abspath(datadir))

                graphdir = re.findall(pattern, lab_field)[0][1]
                config_dec.set("decoding", "graphdir", os.path.abspath(graphdir))

                # The ali dir is supposed to be in exp/model/ which is one level ahead of graphdir
                alidir = graphdir.split("/")[0 : len(graphdir.split("/")) - 1]
                alidir = "/".join(alidir)
                config_dec.set("decoding", "alidir", os.path.abspath(alidir))

            with open(config_dec_file, "w") as configfile:
                config_dec.write(configfile)

            out_folder = os.path.abspath(out_folder)
            files_dec = out_folder + "/exp_files/forward_" + data + "_ep*_ck*_" + forward_outs[k] + "_to_decode.ark"
            out_dec_folder = out_folder + "/decode_" + data + "_" + forward_outs[k]

            if not (os.path.exists(info_file)):

                # Run the decoder
                cmd_decode = (
                    cmd
                    + config["decoding"]["decoding_script_folder"]
                    + "/"
                    + config["decoding"]["decoding_script"]
                    + " "
                    + os.path.abspath(config_dec_file)
                    + " "
                    + out_dec_folder
                    + ' "'
                    + files_dec
                    + '"'
                )
                run_shell(cmd_decode, log_file)

                # remove ark files if needed
                if not forward_save_files[k]:
                    list_rem = glob.glob(files_dec)
                    for rem_ark in list_rem:
                        os.remove(rem_ark)

            # Print WER results and write info file
            cmd_res = "./check_res_dec.sh " + out_dec_folder
            wers = run_shell(cmd_res, log_file).decode("utf-8")
            res_file = open(res_file_path, "a")
            res_file.write("%s\n" % wers)
            print(wers)

            try:
                if len(wers) > 0:
                    w_error_rate, w_tot, w_ins, w_del, w_sub = get_wer_stats(wers)

                    store_wer_stats(config_file_name, data, wers)

                    if wandb_on or wandb_decode_only:
                        wandb.quick_log("WER_" + data, w_error_rate, commit=True)

            except IOError:
                pass


if wandb_on or wandb_decode_only:
    wandb.quick_log("status", "complete", commit = True)

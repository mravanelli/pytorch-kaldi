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

import sys
import configparser
import os
from utils import is_sequential_dict, model_init, optimizer_init, forward_model, progress
from data_io import load_counts
import numpy as np
import random
import torch
from distutils.util import strtobool
import time
import threading
import itertools
import torch.nn.functional as functional

from data_io import read_lab_fea, open_or_fd, write_mat
from utils import shift
import gan_networks
import weights_and_biases as wandb


def save_tensor_list_to_png(array, titles=[], fig_name="tensor.png"):
    import matplotlib.pyplot as plt
    plt.figure()

    for i in range(1, len(array) + 1):
        plt.subplot(len(array), 1, i)

        if len(array) == 4 and i <= 2:
            graph_colour = "b"
        elif len(array) == 4:
            graph_colour = "r"
        elif i == 2:
            graph_colour = "r"
        else:
            graph_colour = "b"

        plt.plot(array[i - 1].detach().numpy(), graph_colour)

        if len(titles) == len(array):
            plt.title(titles[i - 1])


    plt.tight_layout(True)
    plt.savefig(fig_name)
    plt.close()


def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    from torch.autograd import Variable
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 440)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_pearson_correlation(tensor1, tensor2):
    from scipy.stats import pearsonr

    output1 = tensor1.detach().cpu().numpy()
    output2 = tensor2.detach().cpu().numpy()

    if output1.shape == output2.shape:

        # calculate Pearson's correlation
        if len(output1.shape) > 1:
            correlation = 0
            for i in range(output1.shape[0]):
                try:
                    temp_corr, _ = pearsonr(output1[i], output2[i])
                except:
                    temp_corr = 0
                correlation += temp_corr
            if output1.shape[0] > 0:
                correlation = correlation / output1.shape[0]
        else:
            correlation, _ = pearsonr(output1, output2)

        return correlation
    else:
        return 0


def get_mean_squared_error(tensor1, tensor2):

    output1 = tensor1.detach().cpu()
    output2 = tensor2.detach().cpu()

    if output1.shape == output2.shape:
        if len(output1.shape) > 1:
            error = 0
            for i in range(output1.shape[0]):
                error += torch.mean(torch.abs(torch.abs(output1) - torch.abs(output2)))
            if output1.shape[0] > 0:
                error = error / output1.shape[0]
        else:
            error = torch.mean(torch.abs(torch.abs(output1) - torch.abs(output2)))

        return error.numpy()
    else:
        return 0


def read_next_chunk_into_shared_list_with_subprocess(
    read_lab_fea, shared_list, cfg_file, is_production, output_folder, wait_for_process
):
    p = threading.Thread(target=read_lab_fea, args=(cfg_file, is_production, shared_list, output_folder))
    p.start()
    if wait_for_process:
        p.join()
        return None
    else:
        return p


def extract_data_from_shared_list(shared_list):
    data_name = shared_list[0]
    data_end_index_fea = shared_list[1]
    data_end_index_lab = shared_list[2]
    fea_dict = shared_list[3]
    lab_dict = shared_list[4]
    arch_dict = shared_list[5]
    data_set = shared_list[6]
    return data_name, data_end_index_fea, data_end_index_lab, fea_dict, lab_dict, arch_dict, data_set


def convert_numpy_to_torch(data_set_dict, save_gpumem, use_cuda):
    if not (save_gpumem) and use_cuda:
        data_set_inp = torch.from_numpy(data_set_dict["input"]).float().cuda()
        data_set_ref = torch.from_numpy(data_set_dict["ref"]).float().cuda()
    else:
        data_set_inp = torch.from_numpy(data_set_dict["input"]).float()
        data_set_ref = torch.from_numpy(data_set_dict["ref"]).float()
    data_set_ref = data_set_ref.view((data_set_ref.shape[0], 1))
    return data_set_inp, data_set_ref


def get_labels(batch_size, label):
    return torch.ones((batch_size, 1)) * label


def wgan_loss_d(dx, dz):
    return -torch.mean(dx) + torch.mean(dz)


def run_nn(
    data_name, data_set, data_end_index, fea_dict, lab_dict, arch_dict, cfg_file, processed_first, next_config_file,
        epoch=1, wandb_on=False, chunk=0
):
    # This function processes the current chunk using the information in cfg_file. In parallel, the next chunk is load into the CPU memory

    # Reading chunk-specific cfg file (first argument-mandatory file)
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config.read(cfg_file)

    # Setting torch seed
    seed = int(config["exp"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reading config parameters
    output_folder = config["exp"]["out_folder"]
    use_cuda = strtobool(config["exp"]["use_cuda"])
    multi_gpu = strtobool(config["exp"]["multi_gpu"])

    try:
        torch.cuda.set_device(int(config["exp"]["cuda_device"]))
    except KeyError:
        torch.cuda.set_device(0)

    to_do = config["exp"]["to_do"]
    info_file = config["exp"]["out_info"]

    model = config["model"]["model"].split("\n")

    forward_outs = config["forward"]["forward_out"].split(",")
    forward_normalize_post = list(map(strtobool, config["forward"]["normalize_posteriors"].split(",")))
    forward_count_files = config["forward"]["normalize_with_counts_from"].split(",")
    require_decodings = list(map(strtobool, config["forward"]["require_decoding"].split(",")))

    use_cuda = strtobool(config["exp"]["use_cuda"])
    save_gpumem = strtobool(config["exp"]["save_gpumem"])
    is_production = strtobool(config["exp"]["production"])

    if to_do == "train":
        batch_size = int(config["batches"]["batch_size_train"])
        try:
            gan_batch_size = int(config["gan"]["batch_size"])
        except KeyError:
            pass


    if to_do == "valid":
        batch_size = int(config["batches"]["batch_size_valid"])

    if to_do == "forward":
        batch_size = 1

    if config["gan"]["arch_gan"] == "True" and to_do == "train":
        gan_on = True
    else:
        gan_on = False


    # ***** Reading the Data ********
    if processed_first:
        # Reading all the features and labels for this chunk
        shared_list = []

        p = threading.Thread(target=read_lab_fea, args=(cfg_file, is_production, shared_list, output_folder))

        p.start()
        p.join()

        data_name = shared_list[0]
        data_end_index = shared_list[1]
        fea_dict = shared_list[2]
        lab_dict = shared_list[3]
        arch_dict = shared_list[4]
        data_set = shared_list[5]

        # converting numpy tensors into pytorch tensors and put them on GPUs if specified
        if not (save_gpumem) and use_cuda:
            data_set = torch.from_numpy(data_set).float().cuda()
        else:
            data_set = torch.from_numpy(data_set).float()

        try:
            if config["ganset"]["create_set"] == "True":
                gan_out_folder = config["ganset"]["out_folder"]
                smallset = data_set[:,:40].clone()
                smallset = torch.cat((smallset, torch.unsqueeze(data_set[:,-1], dim = 1)), dim = 1)
                torch.save(smallset, os.path.join(gan_out_folder, "chunk_" + str(chunk) + ".pt"))
        except KeyError:
            pass
    else:
        try:
            if config["ganset"]["create_set"] == "True":
                gan_out_folder = config["ganset"]["out_folder"]
                smallset = data_set[:,:40].clone()
                smallset = torch.cat((smallset, torch.unsqueeze(data_set[:,-1], dim = 1)), dim = 1)
                torch.save(smallset, os.path.join(gan_out_folder, "chunk_" + str(chunk) + ".pt"))
        except KeyError:
            pass


    # Reading all the features and labels for the next chunk
    shared_list = []
    p = threading.Thread(target=read_lab_fea, args=(next_config_file, is_production, shared_list, output_folder))
    p.start()

    # Reading model and initialize networks
    inp_out_dict = fea_dict

    [nns, costs] = model_init(inp_out_dict, model, config, arch_dict, use_cuda, multi_gpu, to_do)

    if config["gan"]["arch_gan"] == "True":

        # Create Generator
        generator_class = getattr(gan_networks, config["generator"]["arch_name"])
        generator = generator_class(nns[str(config["architecture1"]["arch_name"])].get_input_dim(),
                                    nns[str(config["architecture1"]["arch_name"])].get_output_dim(),
                                    config["generator"])

        if use_cuda:
            generator = generator.cuda()

        directory_g = os.path.join(config["exp"]["out_folder"], config["gan"]["output_path_g"])

        if os.path.exists(directory_g):

            try:
                if int(config["exp"]["cuda_device"]) == 0:
                    generator.load_state_dict(torch.load(directory_g, map_location="cuda:0"))
                elif int(config["exp"]["cuda_device"]) == 1:
                    generator.load_state_dict(torch.load(directory_g, map_location="cuda:1"))

            except RuntimeError:
                print("Load error loading G, network will be recreated.")


    # optimizers initialization
    optimizers = optimizer_init(nns, config, arch_dict)

    # pre-training and multi-gpu init
    for net in nns.keys():
        pt_file_arch = config[arch_dict[net][0]]["arch_pretrain_file"]

        if pt_file_arch != "none":
            if use_cuda:
                try:
                    if int(config["exp"]["cuda_device"]) == 0:
                        checkpoint_load = torch.load(pt_file_arch, map_location="cuda:0")
                    elif int(config["exp"]["cuda_device"]) == 1:
                        checkpoint_load = torch.load(pt_file_arch, map_location="cuda:1")

                except FileNotFoundError:
                    # File does not exist, load most recent model
                    exp_file_names = os.path.dirname(pt_file_arch)
                    if os.path.exists(exp_file_names):
                        exp_file_list = os.listdir(exp_file_names)

                        new_pt_file_arch = ''

                        for exp_file in exp_file_list:
                            if exp_file.__contains__('final') and exp_file.__contains__('.pkl'):
                                new_pt_file_arch = os.path.join(exp_file_names, exp_file)
                                break
                            elif exp_file.__contains__('.pkl'):
                                new_pt_file_arch = os.path.join(exp_file_names, exp_file)


                    if int(config["exp"]["cuda_device"]) == 0:
                        checkpoint_load = torch.load(new_pt_file_arch, map_location="cuda:0")
                    elif int(config["exp"]["cuda_device"]) == 1:
                        checkpoint_load = torch.load(new_pt_file_arch, map_location="cuda:1")

                except EOFError:
                    if int(config["exp"]["cuda_device"]) == 0:
                        checkpoint_load = torch.load(os.path.join(output_folder, "exp_files/backup.pkl"), map_location="cuda:0")
                    elif int(config["exp"]["cuda_device"]) == 1:
                        checkpoint_load = torch.load(os.path.join(output_folder, "exp_files/backup.pkl"), map_location="cuda:1")
            else:
                checkpoint_load = torch.load(pt_file_arch, map_location="cpu")
            nns[net].load_state_dict(checkpoint_load["model_par"])
            optimizers[net].load_state_dict(checkpoint_load["optimizer_par"])
            optimizers[net].param_groups[0]["lr"] = float(
                config[arch_dict[net][0]]["arch_lr"]
            )  # loading lr of the cfg file for pt

        if multi_gpu:
            nns[net] = torch.nn.DataParallel(nns[net])

    if to_do == "forward":

        post_file = {}
        for out_id in range(len(forward_outs)):
            if require_decodings[out_id]:
                out_file = info_file.replace(".info", "_" + forward_outs[out_id] + "_to_decode.ark")
            else:
                out_file = info_file.replace(".info", "_" + forward_outs[out_id] + ".ark")
            post_file[forward_outs[out_id]] = open_or_fd(out_file, output_folder, "wb")

    # Save the model
    if to_do == "train":
        for net in nns.keys():
            checkpoint = {}
            if multi_gpu:
                checkpoint["model_par"] = nns[net].module.state_dict()
            else:
                checkpoint["model_par"] = nns[net].state_dict()

            checkpoint["optimizer_par"] = optimizers[net].state_dict()

            torch.save(checkpoint, os.path.join(output_folder, "exp_files/backup.pkl"))

    # check automatically if the model is sequential
    seq_model = is_sequential_dict(config, arch_dict)

    # ***** Minibatch Processing loop********
    if seq_model or to_do == "forward":
        N_snt = len(data_name)
        N_batches = int(N_snt / batch_size)
    else:
        N_ex_tr = data_set.shape[0]
        N_batches = int(N_ex_tr / batch_size)

    beg_batch = 0
    end_batch = batch_size

    snt_index = 0
    beg_snt = 0

    start_time = time.time()

    # array of sentence lengths
    arr_snt_len = shift(shift(data_end_index, -1, 0) - data_end_index, 1, 0)
    arr_snt_len[0] = data_end_index[0]

    loss_sum = 0
    err_sum = 0

    inp_dim = data_set.shape[1]

    double_features = False
    try:
        if config["gan"]["double_features"] == "True":
            double_features = True
    except KeyError:
        pass

    for i in range(N_batches):
        max_len = 0

        if seq_model:

            max_len = int(max(arr_snt_len[snt_index : snt_index + batch_size]))
            inp = torch.zeros(max_len, batch_size, inp_dim).contiguous()

            for k in range(batch_size):

                snt_len = data_end_index[snt_index] - beg_snt
                N_zeros = max_len - snt_len

                # Appending a random number of initial zeros, tge others are at the end.
                N_zeros_left = random.randint(0, N_zeros)

                # randomizing could have a regularization effect
                inp[N_zeros_left : N_zeros_left + snt_len, k, :] = data_set[beg_snt : beg_snt + snt_len, :]

                beg_snt = data_end_index[snt_index]
                snt_index = snt_index + 1

        else:
            # features and labels for batch i
            if to_do != "forward":
                inp = data_set[beg_batch:end_batch, :].contiguous()
            else:
                snt_len = data_end_index[snt_index] - beg_snt
                inp = data_set[beg_snt : beg_snt + snt_len, :].contiguous()
                beg_snt = data_end_index[snt_index]
                snt_index = snt_index + 1

        # use cuda
        if use_cuda:
            inp = inp.cuda()

        if to_do == "train":
            # Forward input, with autograd graph active

            if gan_on:
                outs_dict = forward_model(
                    fea_dict,
                    lab_dict,
                    arch_dict,
                    model,
                    nns,
                    costs,
                    inp,
                    inp_out_dict,
                    max_len,
                    batch_size,
                    to_do,
                    forward_outs,
                    generator=generator,
                    gan_on=True,
                    double_features=double_features,
                )

            else:
                outs_dict = forward_model(
                    fea_dict,
                    lab_dict,
                    arch_dict,
                    model,
                    nns,
                    costs,
                    inp,
                    inp_out_dict,
                    max_len,
                    batch_size,
                    to_do,
                    forward_outs,
                    double_features = double_features,
                )

            for opt in optimizers.keys():
                optimizers[opt].zero_grad()

            outs_dict["loss_final"].backward()

            for opt in optimizers.keys():
                if not (strtobool(config[arch_dict[opt][0]]["arch_freeze"])):
                    optimizers[opt].step()
        else:
            with torch.no_grad():  # Forward input without autograd graph (save memory)
                if config["gan"]["arch_gan"] == "True":   # Validation and forward
                    outs_dict = forward_model(
                        fea_dict,
                        lab_dict,
                        arch_dict,
                        model,
                        nns,
                        costs,
                        inp,
                        inp_out_dict,
                        max_len,
                        batch_size,
                        to_do,
                        forward_outs,
                        generator=generator,
                        gan_on=True,
                        double_features=double_features,
                    )
                else:
                    outs_dict = forward_model(
                        fea_dict,
                        lab_dict,
                        arch_dict,
                        model,
                        nns,
                        costs,
                        inp,
                        inp_out_dict,
                        max_len,
                        batch_size,
                        to_do,
                        forward_outs,
                        double_features = double_features,
                    )

        if to_do == "forward":
            for out_id in range(len(forward_outs)):

                out_save = outs_dict[forward_outs[out_id]].data.cpu().numpy()

                if forward_normalize_post[out_id]:
                    # read the config file
                    counts = load_counts(forward_count_files[out_id])
                    out_save = out_save - np.log(counts / np.sum(counts))

                # save the output
                write_mat(output_folder, post_file[forward_outs[out_id]], out_save, data_name[i])
        else:
            loss_sum = loss_sum + outs_dict["loss_final"].detach()
            err_sum = err_sum + outs_dict["err_final"].detach()

        # update it to the next batch
        beg_batch = end_batch
        end_batch = beg_batch + batch_size

        # Progress bar
        if to_do == "train":
            status_string = (
                "Training | (Batch "
                + str(i + 1)
                + "/"
                + str(N_batches)
                + ")"
                + " | L:"
                + str(round(loss_sum.cpu().item() / (i + 1), 3))
            )
            if i == N_batches - 1:
                status_string = "Training | (Batch " + str(i + 1) + "/" + str(N_batches) + ")"

        if to_do == "valid":
            status_string = "Validating | (Batch " + str(i + 1) + "/" + str(N_batches) + ")"
        if to_do == "forward":
            status_string = "Forwarding | (Batch " + str(i + 1) + "/" + str(N_batches) + ")"

        progress(i, N_batches, status=status_string)

    elapsed_time_chunk = time.time() - start_time

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    if wandb_on and to_do == "train" :
        wandb.quick_log("train_loss", loss_tot.cpu().numpy(), commit = False)
        wandb.quick_log("train_error", err_tot.cpu().numpy(), commit = False)

    # clearing memory
    del inp, outs_dict, data_set

    # save the model
    if to_do == "train":

        for net in nns.keys():
            checkpoint = {}
            if multi_gpu:
                checkpoint["model_par"] = nns[net].module.state_dict()
            else:
                checkpoint["model_par"] = nns[net].state_dict()

            checkpoint["optimizer_par"] = optimizers[net].state_dict()

            out_file = info_file.replace(".info", "_" + arch_dict[net][0] + ".pkl")
            torch.save(checkpoint, out_file)

    if to_do == "forward":
        for out_name in forward_outs:
            post_file[out_name].close()

    # Write info file
    with open(info_file, "w") as text_file:
        text_file.write("[results]\n")
        if to_do != "forward":
            text_file.write("loss=%s\n" % loss_tot.cpu().numpy())
            text_file.write("err=%s\n" % err_tot.cpu().numpy())
        text_file.write("elapsed_time_chunk=%f\n" % elapsed_time_chunk)

    text_file.close()

    # Getting the data for the next chunk (read in parallel)
    p.join()
    data_name = shared_list[0]
    data_end_index = shared_list[1]
    fea_dict = shared_list[2]
    lab_dict = shared_list[3]
    arch_dict = shared_list[4]
    data_set = shared_list[5]

    # converting numpy tensors into pytorch tensors and put them on GPUs if specified
    if not (save_gpumem) and use_cuda:
        data_set = torch.from_numpy(data_set).float().cuda()
    else:
        data_set = torch.from_numpy(data_set).float()

    return [data_name, data_set, data_end_index, fea_dict, lab_dict, arch_dict]

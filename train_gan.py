##########################################################
# pytorch-kaldi-gan
# Walter Heymans
# North West University
# 2020
##########################################################

import sys
import configparser
import os
import time

import numpy
import numpy as np
import random
import torch
import torch.nn.functional as functional
from torch.optim.optimizer import Optimizer
import gan_networks
import itertools
from shutil import copyfile
import math
import matplotlib.pyplot as plt
import weights_and_biases as wandb
import importlib
import warnings

warnings.filterwarnings("ignore", '', UserWarning)


def print_version_info():
    print("")
    print("".center(40, "#"))
    print(" Pytorch-Kaldi-GAN ".center(38, " ").center(40, "#"))
    print(" Walter Heymans ".center(38, " ").center(40, "#"))
    print(" North West University ".center(38, " ").center(40, "#"))
    print(" 2020 ".center(38, " ").center(40, "#"))
    print("".center(40, "#"), end="\n\n")


def save_tensor_list_to_png(array, titles=[], fig_name="tensor.png"):
    plt.figure(figsize=(8, 6), dpi=300)

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

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def format_time(time_in_seconds):
    hours_remaining = math.floor(time_in_seconds / 3600)
    minutes_remaining = math.floor(time_in_seconds / 60) - (hours_remaining * 60)
    seconds_remaining = math.floor(time_in_seconds) - (minutes_remaining * 60) - (hours_remaining * 3600)

    if hours_remaining > 0:
        return "{}h {}m {}s ".format(hours_remaining, minutes_remaining, seconds_remaining)
    elif minutes_remaining > 0:
        return "{}m {}s ".format(minutes_remaining, seconds_remaining)
    else:
        return "{}s ".format(seconds_remaining)


def get_labels(bs, label):
    return torch.ones((bs, 1)) * label


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


def get_g_performance(clean, noisy, generated):
    ''' Performance metric using Pearson correlation, mean squared error and L1 loss.
        Metric is comparing generator relative to noisy signal.
        Higher is better. '''

    l1_loss_noisy = torch.nn.functional.l1_loss(clean, noisy).item()
    l1_loss_gen = torch.nn.functional.l1_loss(clean, generated).item()

    r_clean_noisy = get_pearson_correlation(clean, noisy)
    r_clean_gen = get_pearson_correlation(clean, generated)

    mse_clean_noisy = get_mean_squared_error(clean, noisy)
    mse_clean_gen = get_mean_squared_error(clean, generated)

    l1_performance = l1_loss_noisy - l1_loss_gen
    r_performance = r_clean_gen - r_clean_noisy
    mse_performance = mse_clean_noisy - mse_clean_gen

    performance_metric = r_performance + mse_performance + l1_performance

    return performance_metric


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


script_start_time = time.time()

print_version_info()

# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

# Output folder creation
out_folder = config["exp"]["out_folder"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Copy the global cfg file into the output folder
cfg_file = out_folder + "/conf.cfg"

with open(cfg_file, "w") as configfile:
    config.write(configfile)

# Read hyper-parameters from config file
seed = int(config['hyperparameters']['seed'])
max_epochs = int(config['hyperparameters']['max_epochs'])
batch_size = int(config['hyperparameters']['batch_size'])

lr_g = float(config['hyperparameters']['lr_g'])
lr_d = float(config['hyperparameters']['lr_d'])

try:
    d_updates = int(config['hyperparameters']['d_updates'])
except KeyError:
    d_updates = 1

real_label = float(config['hyperparameters']['real_label'])

criterion = str(config['hyperparameters']['criterion'])
optimizer = str(config['hyperparameters']['optimizer'])

cycle_consistency_lambda = int(config['hyperparameters']['cycle_consistency_lambda'])
acoustic_model_lambda = float(config['hyperparameters']['acoustic_model_lambda'])
gp_lambda = int(config['hyperparameters']['gp_lambda'])

try:
    l1_lambda = int(config['hyperparameters']['l1_lambda'])
    l2_lambda = int(config['hyperparameters']['l2_lambda'])
except KeyError:
    pass

if config.getboolean("exp", "use_cuda"):
    try:
        cuda_device = int(config['exp']['cuda_device'])
    except ValueError:
        cuda_device = 'cpu'
else:
    cuda_device = 'cpu'

if config["wandb"]["wandb"] == "True":
    wandb_on = True
else:
    wandb_on = False

torch.manual_seed(seed = seed)
random.seed(seed)

clean_dataset_path = str(config['datasets']['clean_dataset'])
noisy_dataset_path = str(config['datasets']['noisy_dataset'])
valid_dataset_path = str(config['datasets']['valid_dataset'])

cw_left = int(config['datasets']['cw_left'])
cw_right = int(config['datasets']['cw_right'])
frames_per_sample = cw_left + cw_right + 1

double_features = False
try:
    if config["hyperparameters"]["double_features"] == "True":
        double_features = True
except KeyError:
    pass

early_stopping = False
try:
    if config["hyperparameters"]["early_stopping"] == "True":
        early_stopping = True
except KeyError:
    pass

train_d_with_noisy = False
try:
    if config["hyperparameters"]["train_d_with_noisy"] == "True":
        train_d_with_noisy = True
except KeyError:
    pass


print("@ Progress: Reading config complete\n")


def print_settings():
    print_width = 64
    print(" Hyper-parameters ".center(print_width, "="))
    print("# Seed:\t\t\t", seed)
    print("# Epochs:\t\t", max_epochs)
    print("# Batch size:\t\t", batch_size)
    print("# Learning rate G:\t", lr_g)
    print("# Learning rate D:\t", lr_d)
    print("# Acoustic model lambda:", acoustic_model_lambda)
    print("# Gradient penalty lambda:", gp_lambda)

    print("# Real label:\t\t", real_label)
    print("# Criterion:\t\t", criterion)
    print("# Optimizer:\t\t", optimizer)

    print("# Cuda device:\t\t", cuda_device)
    print("# Weights and Biases:\t", wandb_on)
    print("# Output directory:\t", out_folder)
    print("# Double features:\t", double_features)
    print("# Early stopping:\t", early_stopping)
    print("=".center(print_width, "="), end = "\n\n")


print_settings()


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_path_clean, dataset_path_noisy, chunk):

        self.validation_set = False

        if not os.path.exists(dataset_path_noisy) or dataset_path_noisy == "":
            self.validation_set = True

        'Initialization'
        self.dataset_path_clean = dataset_path_clean

        files = sorted(os.listdir(self.dataset_path_clean))

        if chunk <= len(files):
            self.dataset_object = torch.load(os.path.join(self.dataset_path_clean, ("chunk_" + str(chunk) + ".pt")), map_location = 'cpu')

        clean_length = int(math.floor(self.dataset_object.shape[0] / frames_per_sample))

        if not self.validation_set:
            # Noisy dataset
            self.dataset_path_noisy = dataset_path_noisy
            files = sorted(os.listdir(self.dataset_path_noisy))

            if chunk <= len(files):
                self.dataset_object_noisy = torch.load(os.path.join(self.dataset_path_noisy, ("chunk_" + str(chunk) + ".pt")), map_location = 'cpu')

            noisy_lenght = int(math.floor(self.dataset_object_noisy.shape[0] / frames_per_sample))

            self.dataset_len = min([clean_length, noisy_lenght])
        else:
            self.dataset_len = clean_length

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataset_len

    def __getitem__(self, index):
        'Generates one sample of data'

        for frame in range(frames_per_sample):
            label = self.dataset_object[index,-1]
            if frame == 0:
                clean = self.dataset_object[index + frame, :40]
            else:
                clean = torch.cat((clean, self.dataset_object[index + frame, :40]), dim = 0)

        if not self.validation_set:
            for frame in range(frames_per_sample):
                label_noisy = self.dataset_object_noisy[index, -1]
                if frame == 0:
                    noisy = self.dataset_object_noisy[index + frame, :40]
                else:
                    noisy = torch.cat((noisy, self.dataset_object_noisy[index + frame, :40]), dim = 0)

            return clean, noisy, label, label_noisy
        else:
            return clean, label

    def getbatch(self, index, batch_size):
        clean, noisy, _, _ = self.__getitem__(index)

        clean = torch.unsqueeze(clean, dim = 0)
        noisy = torch.unsqueeze(noisy, dim = 0)

        for bs in range(batch_size-1):
            tempclean, tempnoisy, _, _ = self.__getitem__(index+bs+1)

            tempclean = torch.unsqueeze(tempclean, dim = 0)
            tempnoisy = torch.unsqueeze(tempnoisy, dim = 0)
            clean = torch.cat((clean, tempclean), dim = 0)
            noisy = torch.cat((noisy, tempnoisy), dim = 0)

        return clean, noisy


number_of_chunks = len(os.listdir(clean_dataset_path))

train_set = Dataset(clean_dataset_path, noisy_dataset_path, 1)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           num_workers = 4)

validation_set = Dataset(valid_dataset_path, "", 1)
valid_loader = torch.utils.data.DataLoader(validation_set,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           num_workers = 4)

print("@ Progress: Dataset loaded")

if cuda_device != 'cpu':
    torch.cuda.set_device(cuda_device)

print("@ Progress: Cuda device set to", cuda_device)

# Create acoustic model
acoustic_model_path = str(config["acoustic_model"]["pretrained_file"])

train_with_am = False

use_external_model = False
try:
    if str(config["acoustic_model"]["use_external_model"]) == "True":
        use_external_model = True
except KeyError:
    pass

if use_external_model:

    if os.path.exists(acoustic_model_path):

        def get_number_hidden_layers(dictionary_keys):
            layer_count = 0
            for key in dictionary_keys:

                if 'wx' in key:
                    layer_count += 1

            layer_count /= 2
            return int(layer_count)

        def get_n_out_dim(dictionary):
            num_layers = get_number_hidden_layers(dictionary.keys())
            last_layer_key = 'wx.' + str(num_layers - 1) + '.weight'

            for key in dictionary.keys():

                if last_layer_key == key:
                    return dictionary[key].shape[0]

            return 0

        try:
            if cuda_device != 'cpu':
                if int(config["exp"]["cuda_device"]) == 0:
                    checkpoint_load = torch.load(acoustic_model_path, map_location="cuda:0")
                elif int(config["exp"]["cuda_device"]) == 1:
                    checkpoint_load = torch.load(acoustic_model_path, map_location="cuda:1")
            else:
                checkpoint_load = torch.load(acoustic_model_path, map_location="cpu")

            N_out_lab_cd = get_n_out_dim(checkpoint_load["model_par"])

            # import the class
            module = importlib.import_module(config["acoustic_model"]["arch_library"])
            nn_class = getattr(module, config["acoustic_model"]["arch_class"])
            config["acoustic_model"]["dnn_lay"] = config["acoustic_model"]["dnn_lay"].replace('N_out_lab_cd', str(N_out_lab_cd))


            if double_features:
                acoustic_model = nn_class(config["acoustic_model"], int(2 * frames_per_sample * 40))
            else:
                acoustic_model = nn_class(config["acoustic_model"], int(frames_per_sample * 40))

            acoustic_model.load_state_dict(checkpoint_load["model_par"])

            acoustic_model = acoustic_model.cuda()

        except RuntimeError:
            print("Error loading acoustic model! Check that models in config file match.")

else:
    if os.path.exists(acoustic_model_path):

        def get_number_hidden_layers(dictionary_keys):
            layer_count = 0
            for key in dictionary_keys:

                if 'wx' in key:
                    layer_count += 1

            layer_count /= 2
            return int(layer_count)

        def get_n_out_dim(dictionary):
            num_layers = get_number_hidden_layers(dictionary.keys())
            last_layer_key = 'wx.' + str(num_layers - 1) + '.weight'

            for key in dictionary.keys():

                if last_layer_key == key:
                    return dictionary[key].shape[0]

            return 0

        try:
            if cuda_device != 'cpu':
                if int(config["exp"]["cuda_device"]) == 0:
                    checkpoint_load = torch.load(acoustic_model_path, map_location="cuda:0")
                elif int(config["exp"]["cuda_device"]) == 1:
                    checkpoint_load = torch.load(acoustic_model_path, map_location="cuda:1")
            else:
                checkpoint_load = torch.load(acoustic_model_path, map_location="cpu")

            N_out_lab_cd = get_n_out_dim(checkpoint_load["model_par"])

            # import the class
            module = importlib.import_module(config["acoustic_model"]["arch_library"])
            nn_class = getattr(module, config["acoustic_model"]["arch_class"])
            config["acoustic_model"]["dnn_lay"] = config["acoustic_model"]["dnn_lay"].replace('N_out_lab_cd', str(N_out_lab_cd))

            if double_features:
                acoustic_model = nn_class(config["acoustic_model"], int(2 * frames_per_sample * 40))
            else:
                acoustic_model = nn_class(config["acoustic_model"], int(frames_per_sample * 40))

            acoustic_model.load_state_dict(checkpoint_load["model_par"])

            acoustic_model = acoustic_model.cuda()

            train_with_am = True
        except RuntimeError:
            print("Error loading acoustic model! Check that models in config file match.")
    else:
        print("Acoustic model path doesnt exist!")

# Create networks and optimizers

# Create Generator
input_dim = train_set.__getitem__(0)[0].shape[0]

generator_class = getattr(gan_networks, config["generator"]["arch_name"])
generator = generator_class(input_dim,
                            input_dim,
                            config["generator"])

if config["hyperparameters"]["criterion"] == "cycle":
    generator_f = generator_class(input_dim,
                                  input_dim,
                                  config["generator"])

# Create Discriminator

discriminator_class = getattr(gan_networks, config["discriminator"]["arch_name"])
discriminator = discriminator_class(input_dim, config["discriminator"])

if config["hyperparameters"]["criterion"] == "cycle":
    discriminator_h = discriminator_class(input_dim, config["discriminator"])


generator = generator.cuda()
discriminator = discriminator.cuda()

if config["hyperparameters"]["criterion"] == "cycle":
    generator_f = generator_f.cuda()
    discriminator_h = discriminator_h.cuda()


# Creating directories
directory_g = os.path.join(out_folder, config["gan"]["output_path_g"])
directory_d = os.path.join(out_folder, config["gan"]["output_path_d"])

gan_dir = os.path.dirname(directory_g)

if not os.path.exists(gan_dir):
    os.mkdir(gan_dir)

if not os.path.exists(gan_dir + "/images"):
    os.mkdir(gan_dir + "/images")

# Copy pretrained models into directory if it is set
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

# Load pretrained models
if os.path.exists(directory_g):
    try:
        generator.load_state_dict(torch.load(directory_g))

        if criterion == "cycle":
            generator_f.load_state_dict(torch.load(os.path.dirname(directory_g) + "/generator_f.pt"))

    except RuntimeError:
        print("Load error loading G, network will be recreated.")

if os.path.exists(directory_d):
    try:
        discriminator.load_state_dict(torch.load(directory_d))

        if criterion == "cycle":
            discriminator_h.load_state_dict(torch.load(os.path.dirname(directory_d) + "/discriminator_h.pt"))

    except RuntimeError:
        print("Load error loading D, network will be recreated.")


# Optimizer initialization
if config["hyperparameters"]["optimizer"] == "adam":
    if criterion == "cycle":
        optimizer_g = torch.optim.Adam(itertools.chain(generator.parameters(), generator_f.parameters()), lr = lr_g)

        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = lr_d)
        optimizer_h = torch.optim.Adam(discriminator_h.parameters(), lr = lr_d)
    else:
        optimizer_g = torch.optim.Adam(generator.parameters(), lr = lr_g, betas = (0.5, 0.999), weight_decay = 0.001)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = lr_d, betas = (0.5, 0.999), weight_decay = 0.001)
elif config["hyperparameters"]["optimizer"] == "rmsprop":
    if criterion == "cycle":
        optimizer_g = torch.optim.RMSprop(itertools.chain(generator.parameters(), generator_f.parameters()), lr = lr_g)

        optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr = lr_d)
        optimizer_h = torch.optim.RMSprop(discriminator_h.parameters(), lr = lr_d)
    else:
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr = lr_g)
        optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr = lr_d)
elif config["hyperparameters"]["optimizer"] == "sgd":
    if criterion == "cycle":
        optimizer_g = torch.optim.SGD(itertools.chain(generator.parameters(), generator_f.parameters()), lr = lr_g)

        optimizer_d = torch.optim.SGD(discriminator.parameters(), lr = lr_d)
        optimizer_h = torch.optim.SGD(discriminator_h.parameters(), lr = lr_d)
    else:
        optimizer_g = torch.optim.SGD(generator.parameters(), lr = lr_g)
        optimizer_d = torch.optim.SGD(discriminator.parameters(), lr = lr_d)


# Start training
print("\n@ Progress: Starting training")

train_start_time = time.time()
number_of_batches = len(train_loader)

if str(config["wandb"]["wandb"]) == "True":
    wandb_cfg = wandb.load_cfg_dict_from_yaml(str(config["wandb"]["config"]))

    # UPDATE config file if Weights and Biases file is different
    wandb_cfg["max_epochs"] = max_epochs
    wandb_cfg["seed"] = seed
    wandb_cfg["batch_size"] = batch_size
    wandb_cfg["lr_g"] = lr_g
    wandb_cfg["lr_d"] = lr_d
    wandb_cfg["criterion"] = criterion
    wandb_cfg["optimizer"] = optimizer
    wandb_cfg["generator"] = str(config["generator"]["arch_name"])
    wandb_cfg["discriminator"] = str(config["discriminator"]["arch_name"])
    wandb_cfg["dataset"] = str(config["exp"]["dataset_name"])
    wandb_cfg["acoustic_model_lambda"] = acoustic_model_lambda
    wandb_cfg["cycle_consistency_lambda"] = cycle_consistency_lambda
    wandb_cfg["gp_lambda"] = gp_lambda

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

        try:
            file_content = wandb_details_file.read().splitlines()
            wandb_run_id = file_content[0]
            wandb_run_name = file_content[1]
        except IndexError:
            pass

        wandb_details_file.close()

        try:
            wandb.initialize_wandb(project = str(config["wandb"]["project"]),
                                   config = wandb_cfg,
                                   directory = out_folder,
                                   resume = True,
                                   identity = wandb_run_id,
                                   name = wandb_run_name)
        except NameError:
            wandb.initialize_wandb(project = str(config["wandb"]["project"]),
                                   config = wandb_cfg,
                                   directory = out_folder,
                                   resume = True)


def create_log_file():
    if not os.path.exists(os.path.join(out_folder, 'log.log')):
        log_file = open(os.path.join(out_folder, 'log.log'), "w")
        log_file.close()


def update_log_file(text_str):
    log_file = open(os.path.join(out_folder, 'log.log'), "a")
    log_file.write(text_str + "\n")
    log_file.close()


def get_last_trained_epoch():
    log_file = open(os.path.join(out_folder, 'log.log'), "r")
    file_lines = log_file.readlines()
    log_file.close()
    if len(file_lines) > 0:
        epoch_last, chunk_last = (file_lines[-1].replace("epoch_", "")).split("_")
        return int(epoch_last), int(chunk_last)
    else:
        return 0, 0


def validate_generator_results():
    with torch.no_grad():
        number_of_valid_batches = len(valid_loader)
        validation_loss = 0
        correct = 0
        total_samples = 0

        for valid_batch, valid_label_batch in valid_loader:
            valid_batch = valid_batch.cuda()
            valid_label_batch = valid_label_batch.cuda()

            if criterion == "am-gan":
                gen_output, _ = generator(valid_batch)
            else:
                gen_output = generator(valid_batch)

            if g_output.shape[0] > 1:
                if double_features:
                    am_evaluation = acoustic_model(torch.cat((valid_batch, gen_output), dim = 1))
                else:
                    am_evaluation = acoustic_model(gen_output)

            validation_loss += functional.nll_loss(am_evaluation, valid_label_batch.long()).item()
            pred = am_evaluation.data.max(1, keepdim = True)[1]
            correct += torch.sum(pred.eq(valid_label_batch.data.view_as(pred))).item()
            total_samples += valid_label_batch.shape[0]

        validation_loss = validation_loss / number_of_valid_batches
        validation_error = 1 - (correct / total_samples)
        return validation_loss, validation_error


def check_discriminator_classification():
    with torch.no_grad():
        v_set = Dataset(clean_dataset_path, noisy_dataset_path, chunk)
        v_loader = torch.utils.data.DataLoader(v_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

        nob = len(v_loader)
        validation_loss = 0
        correct = 0
        total_samples = 0

        for v_clean_batch, v_noisy_batch, _, _ in v_loader:
            nob += 1
            v_clean_batch, v_noisy_batch = v_clean_batch.cuda(), v_noisy_batch.cuda()

            v_input = torch.cat((v_clean_batch, v_noisy_batch), dim=0)
            v_target = torch.cat((torch.ones(v_clean_batch.shape[0]).long(), torch.zeros(v_noisy_batch.shape[0]).long()), dim=0).to(cuda_device)
            v_output = discriminator(v_input)

            validation_loss += functional.cross_entropy(v_output, v_target).item()

            pred = v_output.data.max(1, keepdim=True)[1]

            correct += torch.sum(pred.eq(v_target.data.view_as(pred))).item()
            total_samples += v_target.shape[0]

        validation_loss = validation_loss / nob
        validation_error = 1 - (correct / total_samples)
        return validation_loss, validation_error


create_log_file()

if wandb_on:
    wandb.quick_log("status", "training", commit = False)

epochs_skipped = 0
lowest_valid_error = 1
early_stopping_epoch = 0

file_loss  = open(os.path.join(out_folder, "losses"), "w")
file_loss.close()

for epoch in range(1, max_epochs+1):

    # Check if epoch has been processed
    last_ep, last_ch = get_last_trained_epoch()
    if (epoch < last_ep) or (last_ep == epoch and last_ch == number_of_chunks):
        print("")
        print(" Previously completed epoch: {} ".format(epoch).center(64, "#"))
        epochs_skipped += 1
        continue

    if wandb_on:
        wandb.quick_log("epoch", epoch, commit = False)

    # Training
    epoch_start_time = time.time()
    print("")
    print(" Optimizing epoch: {}/{} ".format(epoch, max_epochs).center(64, "#"))

    for chunk in range(1, number_of_chunks + 1):
        # Check if chunk has been processed
        if (last_ep == epoch) and (chunk <= last_ch):
            continue

        if wandb_on:
            wandb.quick_log("chunk", chunk, commit = True)

        current_batch = 0
        g_loss = 0
        d_loss = 0
        tot_am_loss = 0

        train_set = Dataset(clean_dataset_path, noisy_dataset_path, chunk)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size = batch_size,
                                                   shuffle = True,
                                                   num_workers = 6)

        number_of_batches = len(train_loader)
        print(" Chunk: {}/{} ".format(chunk, number_of_chunks).center(64, "-"))

        for clean_batch, noisy_batch, label_batch, label_noisy_batch in train_loader:
            current_batch += 1
            # Transfer to GPU
            clean_batch, noisy_batch = clean_batch.cuda(), noisy_batch.cuda()
            label_batch, label_noisy_batch = label_batch.cuda(), label_noisy_batch.cuda()

            d_clean_batch = clean_batch
            d_noisy_batch = noisy_batch

            if d_clean_batch.shape[0] > 1 and  d_noisy_batch.shape[0] > 1:
                for k in range(d_updates):
                    # TRAIN DISCRIMINATOR
                    optimizer_d.zero_grad()
                    d_output_clean = discriminator(d_clean_batch)

                    if criterion == "am-gan":
                        g_output, am_gan_output = generator(d_noisy_batch)
                        g_output = g_output.detach()
                    else:
                        g_output = generator(d_noisy_batch).detach()

                    d_output_g = discriminator(g_output)

                    real_labels = get_labels(d_clean_batch.shape[0], real_label).cuda()
                    fake_labels = get_labels(d_clean_batch.shape[0], 0).cuda()

                    if criterion == "bce" or criterion == "bce-l1" or criterion == "bce-l2" or criterion == "bce-all" or criterion == "am-gan":
                        loss_clean = functional.binary_cross_entropy(d_output_clean, real_labels)
                        loss_noisy = functional.binary_cross_entropy(d_output_g, fake_labels)

                        loss_discriminator = loss_clean + loss_noisy

                        loss_discriminator.backward()
                        optimizer_d.step()

                        d_loss += loss_discriminator.item()

                        file_loss = open(os.path.join(out_folder, "losses"), "a")
                        file_loss.write(str(epoch) + "," + str(chunk) + "," + str(loss_discriminator.item()) + ",")
                        file_loss.close()

                    elif criterion == "wgan":
                        if gp_lambda > 0:
                            gp_loss = compute_gradient_penalty(discriminator, d_clean_batch, g_output)
                        else:
                            gp_loss = 0

                        if train_d_with_noisy:
                            loss_discriminator = - torch.mean(d_output_clean) + torch.mean(d_output_g) + (0.1*torch.mean(discriminator(d_noisy_batch))) + (gp_lambda * gp_loss)
                        else:
                            loss_discriminator = - torch.mean(d_output_clean) + torch.mean(d_output_g) + (gp_lambda * gp_loss)

                        loss_discriminator.backward()
                        optimizer_d.step()

                        d_loss += loss_discriminator.item()
                        temp_d_loss = - torch.mean(d_output_clean) + torch.mean(d_output_g)

                        if gp_lambda > 0:
                            file_loss = open(os.path.join(out_folder, "losses"), "a")
                            file_loss.write(str(epoch) + "," + str(chunk) + "," + str(temp_d_loss.item()) + "," + str(gp_loss.item()) + ",")
                            file_loss.close()

                    elif criterion == "cycle":
                        optimizer_h.zero_grad()

                        h_output_noisy = discriminator_h(d_noisy_batch)  # H_f output of noisy signal
                        h_output_f = discriminator_h(generator_f(d_clean_batch))  # H_f output of F

                        criterion_GAN = torch.nn.MSELoss()
                        criterion_GAN = criterion_GAN.cuda()

                        # TRAIN Discriminator D

                        # Real loss
                        loss_real = criterion_GAN(d_output_clean, real_labels)
                        # Fake loss
                        loss_fake = criterion_GAN(d_output_g, fake_labels)

                        # Total loss
                        loss_d = loss_real + loss_fake

                        loss_d.backward()
                        optimizer_d.step()

                        # TRAIN Discriminator H

                        # Real loss
                        loss_real = criterion_GAN(h_output_noisy, real_labels)
                        # Fake loss
                        loss_fake = criterion_GAN(h_output_f, fake_labels)

                        # Total loss
                        loss_h = loss_real + loss_fake

                        loss_h.backward()
                        optimizer_h.step()

                        d_loss += (loss_d.item() + loss_h.item()) / 2

                    if k < (d_updates - 1):
                        d_clean_batch, d_noisy_batch = train_set.getbatch(random.randint(0, train_set.__len__() - batch_size - 1), batch_size)
                        d_clean_batch = d_clean_batch.to(cuda_device)
                        d_noisy_batch = d_noisy_batch.to(cuda_device)

                # TRAIN GENERATOR
                optimizer_g.zero_grad()

                if criterion == "am-gan":
                    g_output, am_gan_output = generator(noisy_batch)
                else:
                    g_output = generator(noisy_batch)

                d_verdict = discriminator(g_output)

                am_loss = 0

                if train_with_am:
                    if g_output.shape[0] > 1:
                        if double_features:
                            am_output = acoustic_model(torch.cat((noisy_batch, g_output), dim = 1))
                        else:
                            am_output = acoustic_model(g_output)

                        am_loss = functional.nll_loss(am_output, label_noisy_batch.long())
                        f = open(os.path.join(out_folder, "am_loss.txt"), 'a')
                        f.writelines(str(am_loss))
                        f.close()
                        tot_am_loss += am_loss.item()
                    else:
                        am_loss = 0
                elif use_external_model:
                    if g_output.shape[0] > 1:
                        am_output = acoustic_model(g_output)
                        numpy_output = am_output.detach().cpu().numpy()
                        imported_am_output = torch.from_numpy(numpy_output).cuda()
                        am_loss = functional.nll_loss(imported_am_output, label_noisy_batch.long())

                        f = open(os.path.join(out_folder, "am_loss.txt"), 'a')
                        f.writelines(str(am_loss))
                        f.close()
                        tot_am_loss += am_loss.item()
                    else:
                        am_loss = 0

                if criterion == "bce":
                    gen_labels = get_labels(clean_batch.shape[0], real_label).cuda()

                    bce_loss = functional.binary_cross_entropy(d_verdict, gen_labels)

                    loss_generator = bce_loss + (acoustic_model_lambda * am_loss)

                    loss_generator.backward()
                    optimizer_g.step()
                    if am_loss > 0:
                        g_loss += loss_generator.item() - (acoustic_model_lambda * am_loss.item())

                    file_loss = open(os.path.join(out_folder, "losses"), "a")
                    file_loss.write(str(bce_loss.item()) + "," + str(am_loss.item()) + "\n")
                    file_loss.close()

                elif criterion == "wgan":

                    loss_generator = -torch.mean(d_verdict) + (acoustic_model_lambda * am_loss)

                    loss_generator.backward()
                    optimizer_g.step()

                    if am_loss > 0:
                        g_loss += loss_generator.item() - (acoustic_model_lambda * am_loss.item())

                    file_loss = open(os.path.join(out_folder, "losses"), "a")
                    temp_g_loss = -torch.mean(d_verdict)
                    file_loss.write(str(temp_g_loss.item()) + "," + str(am_loss.item()) + "\n")
                    file_loss.close()

                elif criterion == "cycle":

                    criterion_GAN = torch.nn.MSELoss()
                    criterion_cycle = torch.nn.L1Loss()
                    criterion_identity = torch.nn.L1Loss()

                    criterion_GAN = criterion_GAN.cuda()
                    criterion_cycle = criterion_cycle.cuda()
                    criterion_identity = criterion_identity.cuda()

                    f_verdict = discriminator_h(generator_f(clean_batch))

                    # TRAIN CYCLE GENERATORS

                    # GAN loss
                    loss_GAN_g = criterion_GAN(d_verdict, real_labels)
                    loss_GAN_f = criterion_GAN(f_verdict, real_labels)

                    loss_GAN = (loss_GAN_g + loss_GAN_f) / 2

                    # Cycle loss
                    cycle_input_g = torch.unsqueeze(generator_f(clean_batch), dim = 1).cuda()
                    cycle_input_g = torch.cat((cycle_input_g, torch.randn(cycle_input_g.shape).cuda()), dim = 1)

                    cycle_input_f = torch.unsqueeze(generator(noisy_batch), dim = 1).cuda()
                    cycle_input_f = torch.cat((cycle_input_f, torch.randn(cycle_input_f.shape).cuda()), dim = 1)

                    recov_A = generator(generator_f(clean_batch))
                    loss_cycle_A = criterion_cycle(recov_A, clean_batch)

                    recov_B = generator_f(generator(noisy_batch))
                    loss_cycle_B = criterion_cycle(recov_B, noisy_batch)

                    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                    # Total loss
                    loss_generator = loss_GAN + (cycle_consistency_lambda * loss_cycle) + (acoustic_model_lambda * am_loss)

                    loss_generator.backward()
                    optimizer_g.step()

                    g_loss += loss_generator.item()
                elif criterion == "am-gan":
                    gen_labels = get_labels(clean_batch.shape[0], real_label).cuda()

                    bce_loss = functional.binary_cross_entropy(d_verdict, gen_labels)

                    am_gan_loss = functional.nll_loss(am_gan_output, label_noisy_batch.long())

                    loss_generator = bce_loss + (acoustic_model_lambda * am_gan_loss) + (acoustic_model_lambda * am_loss)

                    loss_generator.backward()
                    optimizer_g.step()

                    g_loss += loss_generator.item()

            # Print end of batch results
            print("Processing batch", current_batch, "/", number_of_batches, "\r", end = '')

            try:
                am_loss = am_loss.item()
            except:
                pass

        print("\nD-loss:\t %.4f  | G-loss:\t %.4f  | AM-loss:\t %.4f" % (round(d_loss / current_batch, 4), round(g_loss / current_batch, 4), round(tot_am_loss / current_batch, 4)))

        f = open(os.path.join(out_folder, "performance.txt"), 'a')
        f.writelines(["Epoch " + str(epoch) + " Chunk: " + str(chunk),
                      "\nD-loss:\t %.4f  | G-loss:\t %.4f  | AM-loss:\t %.4f\n" % (round(d_loss / current_batch, 4), round(g_loss / current_batch, 4), round(tot_am_loss / current_batch, 4))])
        f.close()

        if wandb_on:
            wandb.quick_log("d-loss", (d_loss / current_batch), commit = False)
            wandb.quick_log("g-loss", (g_loss / current_batch), commit = False)
            if train_with_am:
                wandb.quick_log("am-loss", (tot_am_loss / current_batch), commit = False)

        torch.save(generator.state_dict(), directory_g)
        torch.save(discriminator.state_dict(), directory_d)

        if criterion == "cycle":
            torch.save(generator_f.state_dict(), os.path.dirname(directory_g) + "/generator_f.pt")
            torch.save(discriminator_h.state_dict(), os.path.dirname(directory_d) + "/discriminator_h.pt")

        if config["gan"]["save_figures"] == "True":

            figure_name = out_folder + '/gan/images/e' + str(epoch) + 'c' + str(chunk) + '.png'

            with torch.no_grad():
                numpyarr = [clean_batch[0].cpu(), noisy_batch[0].cpu(), g_output[0].cpu()]
                titles = ["Clean", "Encoded", "Generator"]
                save_tensor_list_to_png(numpyarr, titles, figure_name)

        update_log_file('epoch_' + str(epoch) + '_' +  str(chunk))

    print(" ".center(30, " "), end = '\r')

    if train_with_am:
        print("")
        print(" Validation ".center(64, "-"))
        valid_loss, valid_error = validate_generator_results()
        print("Validation-loss:  %.4f  | Validation-error:   %.4f" % (valid_loss, valid_error))

        f = open(os.path.join(out_folder, "performance.txt"), 'a')
        f.writelines(["\nValidation\n", "Validation-loss:  %.4f  | Validation-error:   %.4f\n" % (valid_loss, valid_error)])
        f.close()

        if early_stopping:
            if valid_error <= lowest_valid_error:
                torch.save(generator.state_dict(), os.path.dirname(directory_g) + "/generator_es.pt")
                torch.save(discriminator.state_dict(), os.path.dirname(directory_d) + "/discriminator_es.pt")
                lowest_valid_error = valid_error
                early_stopping_epoch = epoch

                if wandb_on:
                    wandb.quick_log("early-stopping-epoch", early_stopping_epoch, commit=False)
                    wandb.quick_log("early-stopping-valid-error", lowest_valid_error, commit=False)

        if wandb_on:
            wandb.quick_log("valid-loss", valid_loss, commit = False)
            wandb.quick_log("valid-error", valid_error, commit = False)

    # Print end of epoch summary
    epoch_end_time = time.time()
    ave_epoch_time = (epoch_end_time - train_start_time) / (epoch - epochs_skipped)
    epochs_remaining = max_epochs - (epoch - epochs_skipped)
    estimated_time_left = ave_epoch_time * epochs_remaining

    d_loss = d_loss / number_of_batches
    g_loss = g_loss / number_of_batches

    print(" Epoch {} completed in: {} | ETA: {} ".format(epoch,
                                                         format_time(epoch_end_time - epoch_start_time),
                                                         format_time(estimated_time_left)).center(64, "#"))

    f = open(os.path.join(out_folder, "performance.txt"), 'a')
    f.writelines("\nEpoch {} completed in: {} | ETA: {} \n".format(epoch,
                                                         format_time(epoch_end_time - epoch_start_time),
                                                         format_time(estimated_time_left)))
    f.close()

    if wandb_on:
        wandb.quick_log("ETA", format_time(estimated_time_left), commit = False)
        wandb.quick_log("epoch_time", format_time(epoch_end_time - epoch_start_time), commit = False)

    if early_stopping:
        print("Early-stopping | Epoch:  %d  | Validation-error:   %.4f" % (early_stopping_epoch, lowest_valid_error))
        f = open(os.path.join(out_folder, "performance.txt"), 'a')
        f.writelines("\nEarly-stopping | Epoch:  %d  | Validation-error:   %.4f\n\n" % (early_stopping_epoch, lowest_valid_error))
        f.close()

print("\n@ Progress: Training complete\n")

if wandb_on:
    wandb.quick_log("status", "complete", commit = True)

print("Completed in:", format_time(time.time() - script_start_time))

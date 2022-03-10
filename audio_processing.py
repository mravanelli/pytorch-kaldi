import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
import sys
import random
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

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

clean_dataset_dir = config["exp"]["clean_dataset"]
noisy_dataset_dir = config["exp"]["noisy_dataset"]

print("- Reading config file......OK!")


def normalize_batch(tensor):
    ''' Normalize batch of tensors between -1 and 1 '''
    max_val = torch.abs(torch.max(tensor))
    min_val = torch.abs(torch.min(tensor))
    return torch.mul(torch.sub(torch.div(torch.add(tensor, min_val), torch.add(max_val, min_val)), 0.5), 2)

def validate_dir(dir_list):
    ''' Remove hidden files from directory list '''
    for dir in dir_list:
        if dir.__contains__('.'):
            dir_list.remove(dir)
    return dir_list


def get_utterances(dir_list):
    ''' Seperate utterances and transcription files '''
    transcriptions = ''
    for dir in dir_list:
        if dir.__contains__('.txt'):
            transcriptions = dir
            dir_list.remove(dir)

    return transcriptions, dir_list

def plot_spectrogram(specgram):
    with torch.no_grad():
        plt.figure()
        plt.imshow(specgram.log2()[0, :, :].numpy(), cmap = 'gray')
        plt.show()

def plot_spectrogram_list(specgram_list):
    with torch.no_grad():
        plt.figure()
        for i in range(len(specgram_list)):
            plt.subplot(len(specgram_list), 1 , i+1)
            plt.imshow(specgram_list[i].log2()[0, :, :].numpy(), cmap = 'gray')

        plt.show()


def plot_waveform(waveform_tensor):
    ''' Plot tensor in a figure '''
    with torch.no_grad():
        plt.figure()
        try:
            plt.plot(waveform_tensor.t().detach().to("cpu").numpy())
        except AttributeError:
            plt.plot(waveform_tensor.detach().to("cpu").numpy())
        plt.show()

def plot_waveform_list(waveform_tensor_list):
    ''' Plot tensor in a figure '''
    with torch.no_grad():
        plt.figure()
        for i in range(len(waveform_tensor_list)):
            plt.subplot(len(waveform_tensor_list), 1 , i+1)
            plt.plot(waveform_tensor_list[i].detach().to("cpu").numpy())

        plt.show()

def normalize_tensor(tensor):
    ''' Normalize tensor between -1 and 1 '''
    max_val = torch.abs(torch.max(tensor))
    min_val = torch.abs(torch.min(tensor))
    return torch.mul(torch.sub(torch.div(torch.add(tensor, min_val), torch.add(max_val, min_val)), 0.5), 2)


def get_context(mfcc_tensor, context_width_left, context_width_right):
    for i in range(context_width_left, mfcc_tensor.shape[2] - context_width_right):
        mfcc_frame = mfcc_tensor[:,1:,(i-context_width_left):(i+context_width_right+1)]


def get_batch(mfcc_tensor, batch_nr, batchsize):
    batch_nr = batch_nr * batchsize
    batch_tensor = mfcc_tensor[batch_nr:batch_nr+batchsize,:]
    return batch_tensor


def reshape_utterance(mfcc_tensor):
    ''' Transpose and remove MFCC 0 '''
    mfcc_tensor = torch.squeeze(mfcc_tensor, dim = 0)
    return torch.transpose(mfcc_tensor, dim0 = 1, dim1 = 0)


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


def get_labels(batch_size, label):
    return torch.ones((batch_size, 1)) * label


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size



class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = ((self.kernel_size[0] - 1) // 2)


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        self.input_block = nn.Sequential(
            Conv1dAuto(in_channels = 2, out_channels = 32, kernel_size = 3, bias = True),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            Conv1dAuto(in_channels = 64, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            Conv1dAuto(in_channels = 64, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            Conv1dAuto(in_channels = 32, out_channels = 1, kernel_size = 3, bias = False),
        )

    def forward(self, x):
        x1 = self.input_block(x)
        x2 = self.block1(x1)
        x3 = self.block2(torch.cat((x1, x2), dim = 1))
        x4 = self.block3(torch.cat((x2, x3), dim = 1))
        return torch.squeeze(x4, dim = 1)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim

        self.block = nn.Sequential(

            Conv1dAuto(in_channels = 1, out_channels = 32, kernel_size = 3, bias = True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.15),

            spectral_norm(Conv1dAuto(in_channels = 32, out_channels = 32, kernel_size = 3, bias = False)),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.15),

            spectral_norm(Conv1dAuto(in_channels = 32, out_channels = 64, kernel_size = 3, bias = False)),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.15),

            spectral_norm(Conv1dAuto(in_channels = 64, out_channels = 64, kernel_size = 3, bias = False)),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.15),

        )

        self.out_block = nn.Sequential(
            spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block(x.view(-1, 1, self.input_dim))
        x = self.out_block(x.view(-1, 128))
        return x


if config["exp"]["dataset_name"] == "LibriSpeech":
    speaker_lst = os.listdir(clean_dataset_dir)
    speaker_lst = validate_dir(speaker_lst)

    N_epochs_tr = int(config['gan']['N_epochs_tr'])
    batch_size = int(config['gan']['batch_size'])

    # Create networks and optimizers
    input_dim = 40

    generator = Generator(input_dim).cuda()
    discriminator = Discriminator(input_dim).cuda()

    if os.path.exists('gan/generator_audio.pt'):
        try:
            generator.load_state_dict(torch.load('gan/generator_audio.pt'))

        except RuntimeError:
            print("Load error loading G, network will be recreated.")

    if os.path.exists('gan/discriminator_audio.pt'):
        try:
            discriminator.load_state_dict(torch.load('gan/discriminator_audio.pt'))

        except RuntimeError:
            print("Load error loading D, network will be recreated.")


    if config["gan"]["optimizer"] == "adam":
        optimizer_g = torch.optim.Adam(generator.parameters(), lr = float(config["gan"]["learning_rate_g"]))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = float(config["gan"]["learning_rate_d"]))
    elif config["gan"]["optimizer"] == "rmsprop":
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr = float(config["gan"]["learning_rate_g"]))
        optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr = float(config["gan"]["learning_rate_d"]))


    # Training
    print("\nTraining started.\n")

    for epoch in range(1, N_epochs_tr + 1):
        print("------------------ Epoch", epoch, "/", N_epochs_tr, "------------------")
        generator.train()
        discriminator.train()

        d_loss = 0
        g_loss = 0

        correlation_noisy = 0
        correlation_g = 0

        mse_noisy = 0
        mse_g = 0

        performance_metric = 0

        total_batches = 0

        for speaker in speaker_lst:
            speaker_dir_clean = os.path.join(clean_dataset_dir, speaker)
            speaker_dir_noisy = os.path.join(noisy_dataset_dir, speaker)

            # Get chapters by speaker
            chapter_lst = os.listdir(speaker_dir_clean)
            chapter_lst = validate_dir(chapter_lst)

            for chap in chapter_lst:
                chapter_dir_clean = os.path.join(speaker_dir_clean, chap)
                chapter_dir_noisy = os.path.join(speaker_dir_noisy, chap)

                # Get utterances by speaker per chapter
                utterance_lst = os.listdir(chapter_dir_clean)
                utt_transcripitons, utterance_lst = get_utterances(utterance_lst)
                j = 0
                for utt in utterance_lst:
                    utterance_dir_clean = os.path.join(chapter_dir_clean, utt)
                    utterance_dir_noisy = os.path.join(chapter_dir_noisy, utt)

                    audio_clean, sample_rate = torchaudio.load(utterance_dir_clean)
                    audio_noisy, _ = torchaudio.load(utterance_dir_noisy)

                    #CMN_extraction = torchaudio.transforms.SlidingWindowCmn(cmn_window = 600, min_cmn_window = 100)
                    MFCC_extraction = torchaudio.transforms.MFCC(sample_rate = sample_rate, n_mfcc = 40).cuda()
                    #Spectrogram_extraction = torchaudio.transforms.Spectrogram()

                    mfcc_clean = MFCC_extraction(audio_clean.cuda())
                    mfcc_noisy = MFCC_extraction(audio_noisy.cuda())

                    utterance_features_clean = reshape_utterance(mfcc_clean)
                    utterance_features_noisy = reshape_utterance(mfcc_noisy)

                    number_of_batches = int(utterance_features_clean.shape[0] / batch_size)
                    last_batch_size = utterance_features_clean.shape[0] % batch_size

                    if last_batch_size > 0:
                        number_of_batches += 1

                    for batch in range(number_of_batches):
                        total_batches += 1
                        print("Batch: {}   \r".format(total_batches), end = '')

                        data_clean = get_batch(utterance_features_clean, batch, batch_size)
                        data_noisy = get_batch(utterance_features_noisy, batch, batch_size)
                        #=== TRAINING ==================================================================================

                        real_labels = get_labels(data_clean.shape[0], float(config['gan']['real_label'])).cuda()
                        fake_labels = get_labels(data_clean.shape[0], 0).cuda()

                        # Train Discriminator
                        optimizer_d.zero_grad()

                        d_output_real = discriminator(data_clean)

                        z_noise = torch.randn(torch.unsqueeze(data_noisy, dim = 1).shape).cuda()

                        g_output = generator(torch.cat((z_noise, torch.unsqueeze(data_noisy, dim = 1)), dim = 1))
                        d_output_fake = discriminator(g_output)

                        loss_real = F.binary_cross_entropy(d_output_real, real_labels)
                        loss_fake = F.binary_cross_entropy(d_output_fake, fake_labels)

                        loss_d = loss_real + loss_fake

                        loss_d.backward()
                        optimizer_d.step()

                        # Train Generator
                        optimizer_g.zero_grad()

                        d_verdict = discriminator(generator(torch.cat((z_noise, torch.unsqueeze(data_noisy, dim = 1)), dim = 1)))

                        bce_loss = F.binary_cross_entropy(d_verdict, real_labels)
                        cycle_loss = F.l1_loss(data_clean, generator(torch.cat((z_noise, torch.unsqueeze(data_noisy, dim = 1)), dim = 1)))

                        loss_g = bce_loss + cycle_loss

                        loss_g.backward()
                        optimizer_g.step()

                        #==== Statistics ====

                        d_loss += loss_d.item()
                        g_loss += loss_g.item()

                        correlation_noisy += get_pearson_correlation(data_clean, data_noisy)
                        correlation_g += get_pearson_correlation(data_clean, g_output)

                        mse_noisy += get_mean_squared_error(data_clean, data_noisy)
                        mse_g += get_mean_squared_error(data_clean, g_output)

                        performance_metric += get_g_performance(data_clean, data_noisy, g_output)



                print("Discriminator loss:\t %.4f  | Generator loss:\t %.4f" % (round(d_loss/total_batches, 4), round(g_loss/total_batches, 4)))
                print("Correlation G:\t\t %.4f  | Correlation noisy:\t %.4f" % (
                    round(correlation_g/total_batches, 4), round(correlation_noisy/total_batches, 4)))
                print("MSE G:\t\t\t %.4f  | MSE noisy:\t\t %.4f" % (round(mse_g/total_batches, 4), round(mse_noisy/total_batches, 4)))
                print("Performance:", round(performance_metric/total_batches, 4))

            # Save after each speaker
            torch.save(generator.state_dict(), 'gan/generator_audio.pt')
            torch.save(discriminator.state_dict(), 'gan/discriminator_audio.pt')


        print("")

        d_loss = d_loss / total_batches
        g_loss = g_loss / total_batches

        correlation_noisy = correlation_noisy / total_batches
        correlation_g = correlation_g / total_batches

        mse_noisy = mse_noisy / total_batches
        mse_g = mse_g / total_batches

        performance_metric = performance_metric / total_batches

        print("\nEpoch complete")
        print("Discriminator loss:\t %.4f  | Generator loss:\t %.4f" % (round(d_loss, 4), round(g_loss, 4)))
        print("Correlation G:\t\t %.4f  | Correlation noisy:\t %.4f" % (
        round(correlation_g, 4), round(correlation_noisy, 4)))
        print("MSE G:\t\t\t %.4f  | MSE noisy:\t\t %.4f" % (round(mse_g, 4), round(mse_noisy, 4)))
        print("Performance:", round(performance_metric, 4))
        print("Total batches", total_batches)

#===============================================================================================================


    print("\n\nTraining complete\n")

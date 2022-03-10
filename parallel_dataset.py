import configparser
import sox
import logging
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil
import subprocess
import shlex
import sys
import math


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


def plot_waveform(waveform_tensor):
    ''' Plot tensor in a figure '''
    with torch.no_grad():
        plt.figure()
        plt.plot(waveform_tensor.t().detach().to("cpu").numpy())
        plt.show()


def normalize_tensor(tensor):
    ''' Normalize tensor between -1 and 1 '''
    max_val = torch.abs(torch.max(tensor))
    min_val = torch.abs(torch.min(tensor))

    if max_val < min_val:
        max_val = min_val

    return torch.div(tensor, max_val)


def get_average_power(clip):
    return torch.sqrt(torch.mean(clip ** 2))


def add_noise(clean_wav, noise_wav, signal_to_noise, rate_of_repeat = 0, sample_rate = 16000):
    output_len = len(clean_wav)
    noise_len = len(noise_wav)

    if rate_of_repeat > 0:
        # Add silence gaps between noise files
        temp_noise = torch.zeros(noise_len + sample_rate * rate_of_repeat)
        temp_noise[0:noise_len] = noise_wav

        noise_wav = temp_noise
        noise_len = len(noise_wav)

    if output_len < noise_len:
        # Choose a random part from the noise file
        if (noise_len - output_len - 1) > 10:
            rnd = random.randrange(0, (noise_len - output_len - 1))
        else:
            rnd = 1

        new_noise = noise_wav[rnd:(rnd + output_len)]

        clean_power = get_average_power(clean_wav)
        noisy_power = get_average_power(new_noise)

        factor = (clean_power / noisy_power) / (10 ** (signal_to_noise / 20.0))  # noise Coefficient for target SNR
        combined_signal = torch.add(clean_wav, torch.mul(new_noise, torch.sqrt(factor)))

    elif output_len > noise_len:

        # Repeat noise file to get same length as output file
        n_repeat = int(output_len / noise_len)
        n_remain = output_len - (n_repeat * noise_len)

        new_noise = torch.zeros(output_len)

        for i in range(n_repeat):
            new_noise[i*noise_len:(i+1)*noise_len] = noise_wav

        new_noise[n_repeat*noise_len:] = noise_wav[:n_remain]

        clean_power = get_average_power(clean_wav)
        noisy_power = get_average_power(new_noise)

        factor = (clean_power / noisy_power) / (10 ** (signal_to_noise / 20.0))  # noise Coefficient for target SNR
        combined_signal = torch.add(clean_wav, torch.mul(new_noise, torch.sqrt(factor)))

    else:
        # Both lengths are the same
        clean_power = get_average_power(clean_wav)
        noisy_power = get_average_power(noise_wav)

        factor = (clean_power / noisy_power) / (10 ** (signal_to_noise / 20.0))  # noise Coefficient for target SNR
        combined_signal = torch.add(clean_wav, torch.mul(noise_wav, torch.sqrt(factor)))

    combined_signal = normalize_tensor(combined_signal)

    return combined_signal.view(1, -1)


def convolve_impulse(clean_wav, noise_wav, signal_to_noise):
    output_len = len(clean_wav)
    noise_len = len(noise_wav)

    clean_power = get_average_power(clean_wav)
    noisy_power = get_average_power(noise_wav)

    factor = math.sqrt((clean_power / noisy_power) / (10 ** (signal_to_noise / 20.0))) # noise Coefficient for target SNR

    if output_len < noise_len:
        # Choose a random part from the noise file
        rnd = random.randrange(0, (noise_len - output_len - 1))
        new_noise = noise_wav[rnd:(rnd + int(output_len / 10))]

    else:
        # Noise is equal or shorther than clean sample
        new_noise = noise_wav

    new_noise = torch.mul(new_noise, factor)

    clean_np = clean_wav.numpy()
    noisy_np = new_noise.numpy()

    convolution = np.convolve(clean_np, noisy_np, 'same')
    combined_signal = torch.from_numpy(convolution)

    return combined_signal.view(1, -1)


def get_random_noise_file(data_dir):
    ''' Return random noise file from the given directory '''
    audio_list = os.listdir(data_dir)
    rnd = random.randrange(0, len(audio_list))
    data_dir = os.path.join(data_dir, audio_list[rnd])
    noise_tensor, n_sample_rate = torchaudio.load(data_dir)
    return noise_tensor


def backup_and_replace(chapter_file, clean_word, noisy_word):
    if os.path.exists(chapter_file):
        chapter_txt = open(chapter_file, "r")

        temp_lines = []

        entries = chapter_txt.readlines()

        for line in entries:
            temp_lines.append(line.replace(clean_word, noisy_word))

        from_dir = chapter_file
        to_dir = chapter_file.replace(".TXT", "_BACKUP.TXT")
        shutil.copyfile(from_dir, to_dir)

        new_file = open(from_dir, "w")
        new_file.writelines(temp_lines)

        new_file.close()
        chapter_txt.close()


def update_metadata_files(root_folder):
    chapter_file = os.path.join(root_folder, "CHAPTERS.TXT")
    speaker_file = os.path.join(root_folder, "SPEAKERS.TXT")

    clean_word = config["data"]["clean_word"]
    noisy_word = config["data"]["noisy_word"]

    backup_and_replace(chapter_file, clean_word, noisy_word)
    backup_and_replace(speaker_file, clean_word, noisy_word)


def run_shell(cmd):
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()
    p.wait()

    return output.decode('utf-8')


def invoke_process_popen_poll_live(command, shellType=False, stdoutType=subprocess.PIPE):
    """runs subprocess with Popen/poll so that live stdout is shown"""
    try:
        process = subprocess.Popen(
            shlex.split(command), shell=shellType, stdout=stdoutType)
    except:
        print("ERROR {} while running {}".format(sys.exc_info()[1], command))
        return None
    while True:
        output = process.stdout.readline()
        # used to check for empty output in Python2, but seems
        # to work with just poll in 2.7.12 and 3.5.2
        # if output == '' and process.poll() is not None:
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode())
    rc = process.poll()
    return rc


def invoke_process_silent(command, shellType=False, stdoutType=subprocess.PIPE):
    try:
        process = subprocess.Popen(
            shlex.split(command), shell=shellType, stdout=stdoutType)
    except:
        print("ERROR {} while running {}".format(sys.exc_info()[1], command))
        return None
    while True:
        output = process.stdout.readline()

        if process.poll() is not None:
            break
        if output:
            print()
    rc = process.poll()
    return rc


logging.getLogger('sox').setLevel(logging.ERROR)

# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

# Set seed for noise adding
random.seed(int(config["noise"]["seed"]))
torch.manual_seed(int(config["noise"]["seed"]))

# Output folder creation
out_folder = config["data"]["out_folder"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

data_folder = config["data"]["data_folder"]
# Read cfg file options
snr_db = int(config["noise"]["snr_db"])
snr = math.pow(10, snr_db / 10)

snr_array = np.array(list(map(int, config["impulse"]["snrs"].split(","))))
snr_list = 10 ** (snr_array / 10)

print("- Reading config file......OK!")

if config["data"]["dataset"] == "librispeech":
    update_metadata_files(config["data"]["root_folder"])

    speaker_lst = os.listdir(data_folder)
    speaker_lst = validate_dir(speaker_lst)

    # Create parallel dataset
    print("\n- Starting dataset parallelization.\n")
    speaker_count = 1

    for speaker in speaker_lst:
        print(" Speaker {} / {} ".format(speaker_count, len(speaker_lst)).center(40, "="))
        speaker_count += 1

        speaker_dir = os.path.join(data_folder, speaker)

        # Get chapters by speaker
        chapter_lst = os.listdir(speaker_dir)
        chapter_lst = validate_dir(chapter_lst)

        chapter_count = 1

        for chap in chapter_lst:
            print("Chapter {} / {}    \r".format(chapter_count, len(chapter_lst)), end = '')
            chapter_count += 1

            chapter_dir = os.path.join(speaker_dir, chap)

            # Get utterances by speaker per chapter
            utterance_lst = os.listdir(chapter_dir)
            utt_transcripitons, utterance_lst = get_utterances(utterance_lst)

            output_dir = os.path.join(out_folder, speaker, chap)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            transcription_from_dir = os.path.join(chapter_dir, utt_transcripitons)
            transcription_to_dir = os.path.join(output_dir, utt_transcripitons)
            shutil.copyfile(transcription_from_dir, transcription_to_dir)

            for utt in utterance_lst:
                utterance_dir = os.path.join(chapter_dir, utt)
                utt_save_dir = os.path.join(output_dir, utt)

                if config["styles"]["change_speed"] == "True":
                    random_number = random.randint(0, 1)

                    if random_number == 1:
                        vol_fraction = 1 + float(config["impulse"]["volume_change"])
                    else:
                        vol_fraction = 1 - float(config["impulse"]["volume_change"])
                else:
                    vol_fraction = 1

                if config["styles"]["change_volume"] == "True":
                    random_number = random.randint(0, 1)

                    if random_number == 1:
                        speed_fraction = 1 + float(config["impulse"]["speed_change"])
                    else:
                        speed_fraction = 1 - float(config["impulse"]["speed_change"])
                else:
                    speed_fraction = 1

                if config["styles"]["change_speed"] == "True" or config["styles"]["change_volume"] == "True":
                    # create a transformer
                    tfm = sox.Transformer()
                    tfm.tempo(speed_fraction, 's')
                    tfm.vol(vol_fraction)

                    tfm.build_file(
                        input_filepath = utterance_dir, sample_rate_in = int(config["impulse"]["sample_rate"]),
                        output_filepath = utt_save_dir
                    )

                    utterance_dir = utt_save_dir

                if config["styles"]["additive_noise"] == "True":
                    recording, sample_rate = torchaudio.load(utterance_dir)
                    noise = get_random_noise_file(config["noise"]["noise_dir"])

                    recording = normalize_tensor(recording)
                    recording = add_noise(recording[0], noise[0], snr)

                    torchaudio.save(utt_save_dir, recording, sample_rate = sample_rate)
                    utterance_dir = utt_save_dir


                if config["styles"]["add_impulse"] == "True":
                    recording, sample_rate = torchaudio.load(utterance_dir)

                    noise = get_random_noise_file(config["impulse"]["impulse_dir"])

                    recording = normalize_tensor(recording)
                    random_snr_value = random.randrange(len(snr_list))

                    recording = convolve_impulse(recording[0], noise[0], snr_list[random_snr_value])

                    recording = normalize_tensor(recording)

                    torchaudio.save(utt_save_dir, recording, sample_rate = sample_rate)
                    utterance_dir = utt_save_dir

                downsample_clean = False
                if config["styles"]["wav49_encode"] == "True":
                    output_sampling_rate = "16000"

                    if utterance_dir == utt_save_dir:
                        encode = "sox -G " + utterance_dir + " -r 8000 -c 1 -e gsm " + utterance_dir.replace(".flac", ".wav")

                        invoke_process_silent(encode)
                        removed = "rm " + utterance_dir
                        invoke_process_silent(removed)

                        flac_convert = "sox -G " + utterance_dir.replace(".flac", ".wav") + " -r " + output_sampling_rate + " " + utterance_dir
                        invoke_process_silent(flac_convert)
                        removed_wav = "rm " + utterance_dir.replace(".flac", ".wav")
                        invoke_process_silent(removed_wav)
                    else:
                        encode = "sox -G " + utterance_dir + " -r 8000 -c 1 -e gsm " + utt_save_dir.replace(".flac", ".wav")

                        invoke_process_silent(encode)

                        flac_convert = "sox -G " + utt_save_dir.replace(".flac", ".wav") + " -r " + output_sampling_rate + " " + utt_save_dir
                        invoke_process_silent(flac_convert)
                        removed_wav = "rm " + utt_save_dir.replace(".flac", ".wav")
                        invoke_process_silent(removed_wav)
                elif downsample_clean:
                    encode = "sox -G " + utterance_dir + " -r 8000 -c 1 " + utt_save_dir
                    invoke_process_silent(encode)



    cmd = "kaldi_decoding_scripts/create_parallel_dataset.sh " \
          + os.path.basename(config["data"]["out_folder"]) + " " \
          + os.path.dirname(config["data"]["root_folder"])

    invoke_process_popen_poll_live(cmd)

    print("\n\nDataset created successfully\n")

elif config["data"]["dataset"] == "swahili":
    speaker_lst = os.listdir(data_folder)
    speaker_lst = validate_dir(speaker_lst)

    # Create parallel dataset
    print("\n- Starting dataset parallelization.\n")
    speaker_count = 1


    for speaker in speaker_lst:
        print(" Speaker {} / {} ".format(speaker_count, len(speaker_lst)).center(40, "="))
        speaker_count += 1

        speaker_dir = os.path.join(data_folder, speaker)

        # Get utterances by speaker per chapter
        utterance_lst = os.listdir(speaker_dir)
        output_dir = os.path.join(out_folder, speaker)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for utt in utterance_lst:
            utterance_dir = os.path.join(speaker_dir, utt)
            utt_save_dir = os.path.join(output_dir, utt)

            if config["styles"]["change_speed"] == "True":
                random_number = random.randint(0, 1)

                if random_number == 1:
                    vol_fraction = 1 + float(config["impulse"]["volume_change"])
                else:
                    vol_fraction = 1 - float(config["impulse"]["volume_change"])
            else:
                vol_fraction = 1

            if config["styles"]["change_volume"] == "True":
                random_number = random.randint(0, 1)

                if random_number == 1:
                    speed_fraction = 1 + float(config["impulse"]["speed_change"])
                else:
                    speed_fraction = 1 - float(config["impulse"]["speed_change"])
            else:
                speed_fraction = 1

            if config["styles"]["change_speed"] == "True" or config["styles"]["change_volume"] == "True":
                # create a transformer
                tfm = sox.Transformer()
                tfm.tempo(speed_fraction, 's')
                tfm.vol(vol_fraction)

                tfm.build_file(
                    input_filepath = utterance_dir, sample_rate_in = int(config["impulse"]["sample_rate"]),
                    output_filepath = utt_save_dir
                )

                utterance_dir = utt_save_dir

            if config["styles"]["additive_noise"] == "True":
                recording, sample_rate = torchaudio.load(utterance_dir)
                noise = get_random_noise_file(config["noise"]["noise_dir"])

                recording = normalize_tensor(recording)
                recording = add_noise(recording[0], noise[0], snr)

                torchaudio.save(utt_save_dir, recording, sample_rate = sample_rate)
                utterance_dir = utt_save_dir

            if config["styles"]["add_impulse"] == "True":
                recording, sample_rate = torchaudio.load(utterance_dir)

                noise = get_random_noise_file(config["impulse"]["impulse_dir"])

                recording = normalize_tensor(recording)
                random_snr_value = random.randrange(len(snr_list))

                recording = convolve_impulse(recording[0], noise[0], snr_list[random_snr_value])

                recording = normalize_tensor(recording)

                torchaudio.save(utt_save_dir, recording, sample_rate = sample_rate)
                utterance_dir = utt_save_dir


    '''cmd = "kaldi_decoding_scripts/create_parallel_dataset.sh " \
          + os.path.basename(config["data"]["out_folder"]) + " " \
          + os.path.dirname(config["data"]["root_folder"])

    invoke_process_popen_poll_live(cmd)'''

    print("\n\nDataset created successfully\n")
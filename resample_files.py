import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
import sys
import random
import shutil

# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

# Output folder creation
out_folder = config["resample"]["out_folder"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

data_folder = config["resample"]["data_folder"]

print("- Reading config file......OK!")


# Preparing speakers

def normalize_tensor(tensor):
    ''' Normalize tensor between -1 and 1 '''
    max_val = torch.abs(torch.max(tensor))
    min_val = torch.abs(torch.min(tensor))
    return torch.mul(torch.sub(torch.div(torch.add(tensor, min_val), torch.add(max_val, min_val)), 0.5), 2)



if config["resample"]["dataset"] == "qutnoise":
    audio_files = os.listdir(data_folder)

    # Create parallel dataset
    print("\n- Starting resampling.\n")

    sample_rate = int(config["resample"]["sample_rate"])

    for sound in audio_files:
        sound_dir = os.path.join(data_folder, sound)
        recording, o_sample_rate = torchaudio.load(sound_dir)

        recording = normalize_tensor(recording)
        save_dir = os.path.join(out_folder, sound)

        torchaudio.save(save_dir, recording, sample_rate = sample_rate)
        print("Saved:", sound)
from augmentation_utils import *
import configparser
import sox
import logging

logging.getLogger('sox').setLevel(logging.ERROR)

# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

# Output folder creation
out_folder = config["data"]["out_folder"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

data_folder = config["data"]["data_folder"]
# Read cfg file options

snr_array = np.array(list(map(int, config["impulse"]["snrs"].split(","))))
snr_list = 10 ** (snr_array / 10)

print("- Reading config file......OK!")

if config["data"]["dataset"] == "librispeech":


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

                if config["styles"]["add_impulse"] == "True":
                    recording, sample_rate = torchaudio.load(utt_save_dir)

                    noise = get_random_noise_file(config["impulse"]["impulse_dir"])

                    recording = normalize_tensor(recording)
                    random_snr_value = random.randrange(len(snr_list))

                    recording = convolve_impulse(recording[0], noise[0], snr_list[random_snr_value])

                    recording = normalize_tensor(recording)

                    torchaudio.save(utt_save_dir, recording, sample_rate = sample_rate)


    cmd = "kaldi_decoding_scripts/create_parallel_dataset.sh " \
          + os.path.basename(config["data"]["out_folder"]) + " " \
          + os.path.dirname(config["data"]["root_folder"])

    invoke_process_popen_poll_live(cmd)

    print("\n\nDataset created successfully\n")

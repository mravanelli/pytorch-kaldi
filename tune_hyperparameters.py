#!/usr/bin/env python
##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
#
# Description:
# This scripts generates config files with the random hyperparamters specified by the user.
# python tune_hyperparameters.py cfg_file out_folder N_exp hyperparameters_spec
# e.g., python tune_hyperparameters.py cfg/TIMIT_MLP_mfcc.cfg exp/TIMIT_MLP_mfcc_tuning 10 arch_lr=randfloat(0.001,0.01) batch_size_train=randint(32,256) dnn_act=choose_str{relu,relu,relu,relu,softmax|tanh,tanh,tanh,tanh,softmax}
##########################################################


import random
import re
import os
import sys
from random import randint

if __name__ == "__main__":
    cfg_file = sys.argv[1]
    output_folder = sys.argv[2]
    N_exp = int(sys.argv[3])
    hyperparam_list = sys.argv[4:]
    seed = 1234

    print("Generating config file for hyperparameter tuning...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    random.seed(seed)

    for i in range(N_exp):

        cfg_file_out = output_folder + "/exp" + str(i) + ".cfg"

        with open(cfg_file_out, "wt") as cfg_out, open(cfg_file, "rt") as cfg_in:
            for line in cfg_in:

                key = line.split("=")[0]

                if key == "out_folder":
                    line = "out_folder=" + output_folder + "/exp" + str(i) + "\n"

                hyper_found = False
                for hyperparam in hyperparam_list:

                    key_hyper = hyperparam.split("=")[0]

                    if key == key_hyper:

                        if "randint" in hyperparam:
                            lower, higher = re.search("randint\((.+?)\)", hyperparam).group(1).split(",")
                            value_hyper = randint(int(lower), int(higher))
                            hyper_found = True

                        if "randfloat" in hyperparam:
                            lower, higher = re.search("randfloat\((.+?)\)", hyperparam).group(1).split(",")
                            value_hyper = random.uniform(float(lower), float(higher))
                            hyper_found = True

                        if "choose_str" in hyperparam:
                            value_hyper = random.choice(re.search("\{(.+?)\}", hyperparam).group(1).split("|"))
                            hyper_found = True

                        if "choose_int" in hyperparam:
                            value_hyper = int(random.choice(re.search("\{(.+?)\}", hyperparam).group(1).split("|")))
                            hyper_found = True

                        if "choose_float" in hyperparam:
                            value_hyper = float(random.choice(re.search("\{(.+?)\}", hyperparam).group(1).split("|")))
                            hyper_found = True

                        line_out = key + "=" + str(value_hyper) + "\n"

                if not hyper_found:
                    line_out = line

                cfg_out.write(line_out)

            print("Done %s" % cfg_file_out)

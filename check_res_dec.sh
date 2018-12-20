#!/bin/bash

# Compiles best WERs from decodings generated in passed dirs (exp dir containing decode_*)
for d in $@; do for ddecode in $d/decode_*; do echo $ddecode & grep Sum $ddecode/*scor*/*ys | ./best_wer.sh; done; done
exit 0


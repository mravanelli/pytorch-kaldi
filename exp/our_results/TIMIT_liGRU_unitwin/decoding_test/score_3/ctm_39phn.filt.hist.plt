set samples 1000
set xrange [0.000000:1.000000]
set autoscale y
set size 0.78, 1.0
set nogrid
set ylabel 'Counts'
set xlabel 'Confidence Measure'
set title  'Confidence scores for /scratch/ravanelm/exp/TIMIT_liGRU_official_unitwin_l06_v1/decoding_test/score_3/ctm_39phn.filt'
plot '/scratch/ravanelm/exp/TIMIT_liGRU_official_unitwin_l06_v1/decoding_test/score_3/ctm_39phn.filt.hist.dat' using 1:2 '%f%f' title 'All Conf.' with lines, \
     '/scratch/ravanelm/exp/TIMIT_liGRU_official_unitwin_l06_v1/decoding_test/score_3/ctm_39phn.filt.hist.dat' using 1:2 '%f%*s%f' title 'Correct Conf.' with lines, \
     '/scratch/ravanelm/exp/TIMIT_liGRU_official_unitwin_l06_v1/decoding_test/score_3/ctm_39phn.filt.hist.dat' using 1:2 '%f%*s%*s%f' title 'Incorrect Conf.' with lines
set size 1.0, 1.0

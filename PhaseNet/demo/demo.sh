#!/usr/bin/env bash

# demo on how to use this modified version on a SDS archive
# provide the root of the SDS tree in data_dir
# use --input_mseed
# use --save_result if you want to save the probability functions for P and S
#   this produces :
#       - a hdf5 archihve in output_dir with the details of the P and S probability functions
#       - a new SDS tree with mseed files that contain the detection functions
# provide a csv file with the data_list argument,
#       - header : network,station,location,channel,dataquality,year,julday
#       - one line per day of 3 component data
#       - use wildcards only for unkown fields and for the component letter (see demo/fname.csv)
# output picks in output_dir/picks.csv

# to generate a csv file use
sds_to_csv.py ./sds > fname.csv

phasenet_run.py \
    --mode pred \
    --model_dir ../model/190703-214543 \
    --data_dir sds \
    --data_list fname.csv \
    --output_dir output \
    --batch_size 20 \
    --input_mseed \
    --save_result \
#    --plot_figure  # disabled in this version

# phasenet_disp.py \
#     --mode all \
#     --data_dir sds \
#     --output_dir output \
#     --seedid "*.*.*.?H?.D" \
#     --time 2000.223.09.15 2000.223.09.30   \
#     --decim 5000
    
#    --time 2000.223.10.00 2000.223.10.05   \
#    --decim 5000

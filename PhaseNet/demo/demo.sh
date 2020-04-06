#!/usr/bin/env bash

phasenet_run.py \
    --mode=pred \
    --model_dir=../model/190703-214543 \
    --data_dir=sds \
    --data_list=fname.csv \
    --output_dir=output \
    --batch_size=20 \
    --input_mseed \
#    --save_result \
#    --plot_figure

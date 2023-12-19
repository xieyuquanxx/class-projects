#!/bin/bash


python model_eval.py \
        --mode dev \
        --threshold 0.45 \
        --answer robert-chi0.json \
        --check_point /shao_rui/xyq/nlp_el/demos/check_points/chi-robert/robert-chiepoch=3.ckpt
#!/bin/bash

CONFIG_FILE="configs-emod/FR/FR_R50_FPN_HABCAM_SP1.py"
GPU_NUM=2

CUDA_VISIBLE_DEVICES=0,1 PORT=29500 tools/dist_train.sh \
${CONFIG_FILE} \
${GPU_NUM}


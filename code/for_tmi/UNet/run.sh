#!/bin/sh
python setup.py install
python /home/data/zhaoqianfei/new_pymic/PyMIC/pymic/net_run/train.py /home/data/zhaoqianfei/new_pymic/for_tmi/UNet/config/unet.cfg

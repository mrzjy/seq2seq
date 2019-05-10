#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

@author: zjy
"""
import os
from utils.Hparam_utils import create_hparams
from utils.Common_utils import Monitor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # hparams
    hparams = create_hparams()
    hparams.get_all_beams = False

    # prepare estimator, data and running logic
    monitor = Monitor(hparams)

    # run
    monitor.run()

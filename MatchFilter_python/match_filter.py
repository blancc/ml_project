#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:40:21 2019

@author: liuruoyu
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import matchedFilter, resampling, hardDec, sym2bit, BER_Cal
from readfiles import readfiles


wvform_rx, puls_seq, sym_tx = readfiles(roll_off = 9)

wvform_rx = matchedFilter(wvform_rx,puls_seq)    #match filter
sym_rx = resampling(wvform_rx,sps = 2**7,rsp_rate = 1)    #resampling
plt.figure(figsize=(7,7))
plt.scatter(np.array(sym_rx.real.reshape(1, -1)), np.array(sym_rx.imag.reshape(1, -1)))


moduFormat = 4

sym_dec = hardDec(sym_rx,moduFormat)
bit_rx = sym2bit(sym_dec,moduFormat)
bit_tx = sym2bit(sym_tx,moduFormat)

ber = BER_Cal(bit_tx,bit_rx)
print("ber:", ber)

noise_seq = sym_rx - sym_tx
snr = np.dot(sym_rx.H, sym_rx)/len(sym_rx)/np.var(noise_seq)
snr = 10*np.log10(snr);
print("snr:", snr)

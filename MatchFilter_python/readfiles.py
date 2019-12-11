#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:04:11 2019

@author: liuruoyu
"""

import numpy as np

def readfiles(roll_off):    
    
    path_f_3 =  '/Users/liuruoyu/Documents/cours_de_laval/machine learning/projet/ml_project/dataset/SNR10/pulseShape_rollOff7'
    f_3 = open(path_f_3, 'r')
    puls_seq = []
    for line in f_3.readlines():
        puls_seq.append(line)
    del(puls_seq[0])
    for i in range(len(puls_seq)):
        puls_seq[i] = float(puls_seq[i])
    f_3.close()
    puls_seq = np.array(puls_seq)    
    
    wvform_rx_real = []
    wvform_rx_imag = []
    sym_tx_real = []
    sym_tx_imag = []
    for batch in range(1, 501):
        path_f_1 = '/Users/liuruoyu/Documents/cours_de_laval/machine learning/projet/ml_project/dataset/SNR10/wvform_rx_real_rollOff{}_batch{}'.format(roll_off, batch)
        f_1 = open(path_f_1, 'r')
        for line in f_1.readlines()[1:]:        
            wvform_rx_real.append(line)
        f_1.close()
        
        path_f_2 = '/Users/liuruoyu/Documents/cours_de_laval/machine learning/projet/ml_project/dataset/SNR10/wvform_rx_imag_rollOff{}_batch{}'.format(roll_off, batch)
        f_2 = open(path_f_2, 'r')
        for line in f_2.readlines()[1:]:        
            wvform_rx_imag.append(line)
        f_2.close()     
        
        path_f_4 = '/Users/liuruoyu/Documents/cours_de_laval/machine learning/projet/ml_project/dataset/SNR10/sym_tx_rollOff{}_batch{}'.format(roll_off, batch)
        f_4 = open(path_f_4, 'r')
        for line in f_4.readlines()[1:]:
            sym_tx_real.append(line.split()[0])
            sym_tx_imag.append(line.split()[1])
        f_4.close()
    
    
    wvform_rx = []
    wvform_rx_real = np.array(wvform_rx_real)   
    wvform_rx_imag = np.array(wvform_rx_imag)   
    for i in range(len(wvform_rx_real)):
        wvform_rx.append(complex(float(wvform_rx_real[i]), float(wvform_rx_imag[i])))
    wvform_rx = np.array(wvform_rx)
        
    sym_tx = []
    sym_tx_real = np.array(sym_tx_real)
    sym_tx_imag = np.array(sym_tx_imag)
    for i in range(len(sym_tx_real)):
        sym_tx.append(complex(float(sym_tx_real[i]), float(sym_tx_imag[i])))
    sym_tx = np.matrix(sym_tx).T
        
    
    return wvform_rx, puls_seq, sym_tx
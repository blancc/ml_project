#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:02:09 2019

@author: liuruoyu
"""

import numpy as np


def matchedFilter(wvform,filtCoeff):
    """
    Match filter 
    Correlator + sampling 
    Ref: Poor, H. V. (1994). An introduction to signal detection and 
         estimation, Springer New York.
"""

    filtCoeff = filtCoeff/np.dot(filtCoeff, filtCoeff.reshape(-1,1))   
    
    wvform_cor = np.convolve(wvform,filtCoeff,'same')

    return wvform_cor


def resampling(wvform,sps,rsp_rate):
    
    """
    % Re-sample the waveform by down sampling factor (down_factor)
    % sps(Input args) is the sampling number of input waveform
    % rsp is the sps after sampling
    """
    
    N_sym = int(len(wvform)/sps)
    N_point = int(N_sym*rsp_rate)
    step = int(len(wvform)/N_point)
    
    sampled_wv = np.matrix(wvform[int(step/2)-1::step]).T
    
    return sampled_wv


def hardDec(sym_rx,moduFormat):
    '''
    Hard decision for noisy symbols 
    QPSK/16QAM supproted
    Created date:2019/11/15
    '''
    
    sym_dec = (1+1j) * np.ones((np.shape(sym_rx)[0], 1), dtype = complex) # Initial setting
    
    sym_real = sym_rx.real.reshape(-1, 1)
    sym_imag = sym_rx.imag.reshape(-1, 1)
    
    if moduFormat == 4:
        
        bd1 = 0 # decision boundray
        dec_real = -1*(sym_real<bd1)+1*(sym_real>=bd1)
        dec_imag = -1*(sym_imag<bd1)+1*(sym_imag>=bd1)
        sym_dec.real = dec_real
        sym_dec.imag = dec_imag
        return sym_dec
        
    elif moduFormat == 16:

        bd1 = -2; bd2 = 0; bd3 = 2
        dec_real = -3*(sym_real<bd1)+(-1)*np.dot((sym_real>=bd1), (sym_real<bd2))+\
        1*np.dot((sym_real>=bd2), (sym_real<bd3))+3*(sym_real>=bd3)
        dec_imag = -3*(sym_imag<bd1)+(-1)*np.dot((sym_imag>=bd1), (sym_imag<bd2))+\
        1*np.dot((sym_imag>=bd2), (sym_imag<bd3))+3*(sym_imag>=bd3)
        sym_dec = complex(dec_real,dec_imag)
        return sym_dec
                    
    else:
        print('Unsupported modulation format')


def dec2bin(dec_num, bit_wide=16):
    _, bin_num_abs = bin(dec_num).split('b')
    if len(bin_num_abs) > bit_wide:
        raise ValueError
    else:
        if dec_num >= 0:
            bin_num = bin_num_abs.rjust(bit_wide, '0')
        else:
            _, bin_num = bin(2**bit_wide + dec_num).split('b')
    return bin_num


def sym2bit(symbol,moduFormat):
    '''
    Un-Mapping from symbols to bits
    QPSK/16QAM supproted
    Created date:2019/11/15
    '''
    M = int(np.log2(moduFormat))
    N = len(symbol)*M
    bit = np.zeros((N,1))
        
    if moduFormat == 4:
        codeBook = [[-1-1j,-1+1j],[1+1j,1-1j]]
        codeBook = np.array(codeBook).T.reshape(-1,1)
    elif moduFormat == 16:
        codeBook = [[-3-3j,-3-1j,-3+1j,-3+3j],\
                    [-1+3j,-1+1j,-1-1j,-1-3j],\
                    [1-3j,1-1j,1+1j,1+3j],\
                    [3+3j,3+1j,3-1j,3-3j]]
        codeBook = np.array(codeBook).T.reshape(-1,1)
    else:
        print('Unsupported modulation format')
        return None

    for i in range(len(symbol)):
        index = np.where(codeBook==symbol[i])[0][0]
        bin_seq = dec2bin(index, M)
        for j in range(M):
            bit[i*M+j] = float(bin_seq[j])
    return bit
            
            
def BER_Cal(bit_rec,bit_ref):
    N = 0
    N_err = 0
    for i in range(min(len(bit_ref), len(bit_rec))):
        if bit_rec[i] != bit_ref[i]:
            N_err = N_err + 1
        N = N+1  
    ber = N_err/N
    return ber
    


    
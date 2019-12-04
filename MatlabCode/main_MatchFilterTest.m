%% Machine learning final project - Matched filter
clear;
close all;

%% Control console
N = 2^14;         % number of transmit bits
moduFormat = 4;   
SNR = 8;
roll_off = 0.8;  

rsp_rate = 1;

%% Random bit sequence generation
bit_tx = randi([0,1],N,1);

%% Symbol generation
sym_tx = bit2sym(bit_tx,moduFormat);

%% Pulse shaping
%%%% Here only root raised cosine pulse will be considered,
%%%% Controlled by roll-off factor.

[wvform_tx,puls_seq,sps_max] = pulseShaping(sym_tx,roll_off);

%% AWGN channel z
wvform_rx = awgn(wvform_tx,SNR,'measured');

%% Matched filter (Can be commented and see the results without MF)
% if roll_off>0  % if roll_off=0, no need of MF
%     wvform_rx = matchedFilter(wvform_rx,puls_seq);
% end

%% Resampling
sym_rx = resampling(wvform_rx,sps_max,rsp_rate);

%% BER / SNR
sym_dec = hardDec(sym_rx,moduFormat);
bit_rx = sym2bit(sym_dec,moduFormat);

ber = BER_Cal(bit_tx,bit_rx);
ber

noise_seq = sym_rx-sym_tx;
snr = (sym_rx'*sym_rx)/length(sym_rx)/cov(noise_seq);
snr = 10*log10(snr);
snr

scatterplot(sym_rx)
title('Received constellation')

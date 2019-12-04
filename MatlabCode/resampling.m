function sampled_wv= resampling(wvform,sps,rsp_rate)
% Re-sample the waveform by down sampling factor (down_factor)
% sps(Input args) is the sampling number of input waveform
% rsp is the sps after sampling
%% 
N_sym = floor(length(wvform)/sps);
N_point = floor(N_sym*rsp_rate);
step = floor(length(wvform)/N_point);

sampled_wv = wvform(floor(step/2):step:end).';

end


function [wvform,puls,sps] = pulseShaping(symbols,roll_off)
% Modulation with given symbols and raied cosine pulse
% Pulse id defined by roll-off factor
% Rs defines the output time sequence t_seq
% max_sampling_rate is a local variable, determines the how many points are
% usd for one pulse, i.e., simulation resolution, default at 2^8=256
%% 

sps = 2^7;  % default

N = length(symbols);

%%%% Upsampling for symbol sequence
code = zeros(1,N*sps);
k = 0:1:N-1;
idx = k*sps+sps/2;
code(idx) = symbols.';

%%%% Pulse generation
span = 128;
filtCoeff = rcosdesign(roll_off,span,sps,'sqrt');
filtCoeff = filtCoeff(1:end-1)/max(filtCoeff);

%%%% Shaping
wvform = conv(code,filtCoeff,'same');
puls = filtCoeff;


%% 
figure
stem(real(code)); hold on;
plot(real(wvform)); grid on;
xlabel('N'); xlim([1,2048]);
ylabel('Amp.');
title('Signal output @Tx.');
legend('symbol(real part)','waveform(real part)');

end


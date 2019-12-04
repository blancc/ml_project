function ber = BER_Cal(bit_rec,bit_ref)
% Bit-error-rate calculated
% BER.loc returns the locations index of error bit
% BER.N returns the number of bit for BER calculation
% bit_rec is the received bits, bit_ref is the codebook
%% 
loc = [];
N = 0;
N_err = 0;
for i = 1:min(length(bit_ref),length(bit_rec))
    if bit_rec(i)~=bit_ref(i)
        loc = [loc,i];
        N_err = N_err+1;
    end
    N = N+1;
end

BER.ber = N_err/N;
BER.loc = loc;

ber = BER.ber;

end


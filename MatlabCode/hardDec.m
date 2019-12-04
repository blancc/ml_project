function sym_dec = hardDec(sym_rx,moduFormat)
% Hard decision for noisy symbols 
% QPSK/16QAM supproted
% Created date:2019/11/15
%% 

sym_dec = (1+1j)*ones(size(sym_rx,1),size(sym_rx,2)); % Initial setting

sym_real = real(sym_rx);
sym_imag = imag(sym_rx);

switch moduFormat
    
    case 4
        bd1 = 0; % decision boundray
        dec_real = -1*(sym_real<bd1)+1*(sym_real>=bd1);
        dec_imag = -1*(sym_imag<bd1)+1*(sym_imag>=bd1);
        sym_dec = complex(dec_real,dec_imag);
    case 16
        bd1 = -2; bd2 = 0; bd3 = 2;
        dec_real = -3*(sym_real<bd1)+(-1)*((sym_real>=bd1).*(sym_real<bd2))+...
                    1*((sym_real>=bd2).*(sym_real<bd3))+3*(sym_real>=bd3);
        dec_imag = -3*(sym_imag<bd1)+(-1)*((sym_imag>=bd1).*(sym_imag<bd2))+...
                    1*((sym_imag>=bd2).*(sym_imag<bd3))+3*(sym_imag>=bd3);
        sym_dec = complex(dec_real,dec_imag);
                    
    otherwise
        print('Unsupported modulation format');
        return;
end

end


function symbol = bit2sym(bit,moduFormat)
% Mapping from bits to symbols
% QPSK/16QAM supproted
% Created date:2019/11/13
%% 
M = floor(log2(moduFormat));
N = floor(length(bit)/M);
symbol = zeros(N,1);

switch moduFormat
    
    case 4
        codeBook = [-1-1j,-1+1j;1+1j,1-1j];
    case 16
        codeBook = [-3-3j,-3-1j,-3+1j,-3+3j;...
                    -1+3j,-1+1j,-1-1j,-1-3j;...
                    1-3j,1-1j,1+1j,1+3j;...
                    3+3j,3+1j,3-1j,3-3j];
    otherwise
        print('Unsupported modulation format');
        return;
end

for i=1:N
    code = bit((i-1)*M+1:i*M);
    symbol(i) = codeBook(1+bin2dec(num2str(code')));
end

end


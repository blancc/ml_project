function bit = sym2bit(symbol,moduFormat)
% Un-Mapping from symbols to bits
% QPSK/16QAM supproted
% Created date:2019/11/15
%% 
M = floor(log2(moduFormat));
N = length(symbol)*M;
bit = zeros(N,1);

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

for i=1:length(symbol)
    index = find(symbol(i)==codeBook)-1;
    bin_seq = dec2bin(index,M);
    for j=1:M
        bit((i-1)*M+j)=str2double(bin_seq(j));
    end
end

end

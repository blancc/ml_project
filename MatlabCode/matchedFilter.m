function wvform_cor = matchedFilter(wvform,filtCoeff)
% Match filter 
% Correlator + sampling 
% Ref: Poor, H. V. (1994). An introduction to signal detection and 
%      estimation, Springer New York.
%% Correlator 

filtCoeff = filtCoeff/(filtCoeff*filtCoeff');

wvform_cor = conv(wvform,filtCoeff,'same');




end


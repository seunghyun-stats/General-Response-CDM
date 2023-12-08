function [ind] = bin2ind(bin_vec)
% This function outputs a vector of positive integers x in binary form

    ind = sum(bsxfun(@times, bin_vec, 2.^((size(bin_vec,2)-1):-1:0)),2);

end


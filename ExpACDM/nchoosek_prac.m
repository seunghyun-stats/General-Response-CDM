function [y] = nchoosek_prac(n, k)
% works when n, k are real numbers
    y = gamma(n+1) ./ (gamma(k+1) .* gamma(n-k+1));

end
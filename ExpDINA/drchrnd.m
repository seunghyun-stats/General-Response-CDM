function r = drchrnd(a,n)
% sample from a Dirichlet distribution with parameters (a, a, ..., a)
% n is the dimension

    p = length(a);
    r = gamrnd(repmat(a,n,1),1,n,p);
    r = r ./ repmat(sum(r,2),1,p);
end
function C = ftn_A(eta)
% lognormal normalizing constant A

    eta_1 = eta(:,1);
    eta_2 = eta(:,2);
    C = eta_2 .^2 ./ (4 * eta_1) + log( 1 ./ (2 * eta_1)) /2;
end

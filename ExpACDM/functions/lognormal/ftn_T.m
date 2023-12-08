function T = ftn_T(X)
% lognormal sufficient statistics T

    T_1 = - (log(X)).^2;
    T_2 = log(X);
    T = [T_1; T_2];
end


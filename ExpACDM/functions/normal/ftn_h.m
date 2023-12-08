function C = ftn_h(Y)
% Normal link function h
% Y: size 2 x J

    Y_1 = Y(1, :);
    Y_2 = Y(2, :);
    C = [1 ./ (2*Y_2); Y_1 ./ Y_2];
end


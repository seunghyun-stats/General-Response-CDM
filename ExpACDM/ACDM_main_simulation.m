%% This code was used for the simulation studies in Section 5

% Note that to fit lognormal or Poisson-ACDMs, one has to use the
% distribution-specific functions (ftn_A, ftn_h, ftn_T) in the folder
% `functions`

% Here, we set the sample size N = 500 but this can be adjusted


%% define Q-matrix and proportion parameters
K = 5;
A = binary(0:(2^K-1), K);
prop_true = ones(2^K,1)/2^K;
nu_true = prop_true;

N = 500;

Q = [eye(K); eye(K); eye(K); eye(K)];
for k = 1:(K-1)
    Q(k, k+1) = 1;
    Q(k+1, k) = 1;
end
[J, K] = size(Q);

rng(2023);

%% Simulation under the lognormal-ACDM model
% we can also simulate under other normal-based-ACDMs

% set item parameters
g = -1 * ones(J,1);
c = 2 * ones(J,1);

beta_true = zeros(J, K + 1);
beta_part_in = zeros(J, K);
for j = 1:J
    beta_true(j, 1) = g(j);
    beta_part_in(j, Q(j,:) == 1) = (c(j)-g(j))/sum(Q(j,:) == 1);
end
beta_true(:, 2:end) = beta_part_in;

gamma_true = 1*ones(J, 1);

% simulate data and fit
n_sim = 100;
nu_est = cell(n_sim,1);
beta_est = cell(n_sim,1);
gamma_est = cell(n_sim,1);
loglik = zeros(n_sim,1);
itera = zeros(n_sim,1);
time = zeros(n_sim,1);

parpool(4)
parfor (c = 1:n_sim, 4)
    [X, ~] = generate_X_ACDM_lognormal(N, prop_true, beta_true, Q, A, gamma_true);
    % [X, ~] = generate_X_ACDM_normal(N, prop_true, beta_true, Q, A, gamma_true);

    nu_in = drchrnd(ones(2^K,1)',1)';
    beta_in = zeros(J, K + 1);
    beta_part_in = zeros(J, K);
    for j = 1:J
        beta_in(j, 1) = - 2*rand(1);
        beta_part_in(j, Q(j,:) == 1) = 4*rand(1);
    end
    beta_in(:, 2:end) = beta_part_in;
    gamma_in = rand(J,1);

    tic;    
    [nu_est{c}, beta_est{c}, gamma_est{c}, loglik(c), itera(c)] = get_EM_ACDM(X, Q, A, nu_in, beta_in, gamma_in);
    time(c) = toc;
end

% computing performance measures
mean(time)
mean(itera)

err_nu = 0;
err_beta = 0;
err_gamma = 0;
err_eta = 0;

for i = 1:n_sim
    err_nu = err_nu + mean((nu_true - nu_est{i}).^2);
    err_beta = err_beta + mean((beta_true - beta_est{i}).^2, 'all');
    err_gamma = err_gamma + mean((gamma_true - gamma_est{i}).^2);
end

err_nu = sqrt(err_nu / n_sim);
err_beta = sqrt(err_beta / n_sim);
err_gamma = sqrt(err_gamma / n_sim);

%% Simulation under the Poisson-ACDM model
% set item parameters
g = 1 * ones(J,1);
c = 3 * ones(J,1);

beta_true = zeros(J, K + 1);
beta_part_in = zeros(J, K);
for j = 1:J
    beta_true(j, 1) = g(j);
    beta_part_in(j, Q(j,:) == 1) = (c(j)-g(j))/sum(Q(j,:) == 1);
end
beta_true(:, 2:end) = beta_part_in;

gamma_in = ones(J,1);   % dummy values that are not used for Poisson

% simulate data and fit
n_sim = 100;
nu_est = cell(n_sim,1);
beta_est = cell(n_sim,1);
loglik = zeros(n_sim,1);

parpool(4)
parfor (i = 1:n_sim, 4)
    [X, ~] = generate_X_ACDM_poisson(N, nu_true, beta_true, Q, A);
    nu_in = drchrnd(ones(2^K,1)',1)';

    beta_in = zeros(J, K + 1);
    beta_part_in = zeros(J, K);
    for j = 1:J
        beta_in(j, 1) = 0.5 + rand(1);
        beta_part_in(j, Q(j,:) == 1) = 0.4 + 0.5*rand(1);
    end
    beta_in(:, 2:end) = beta_part_in;

    [nu_est{i}, beta_est{i}, ~, loglik(i), ~] = get_EM_ACDM(X, Q, A, nu_in, beta_in, gamma_in);
    fprintf('Simulation number %d completed \n', i);
end

% computing performance measures
err_nu = 0;
err_beta = 0;

for i = 1:n_sim
    err_nu = err_nu + mean((nu_true - nu_est{i}).^2);
    err_beta = err_beta + sum((beta_true - beta_est{i}).^2, 'all')/(J + sum(Q,'all'));
end

err_nu = sqrt(err_nu / n_sim);
err_beta = sqrt(err_beta / n_sim);


%% Simulation under the negbin-DINA model
% set item parameters
g = 1 * ones(J,1);
c = 3 * ones(J,1);

beta_true = zeros(J, K + 1);
beta_part_in = zeros(J, K);
for j = 1:J
    beta_true(j, 1) = g(j);
    beta_part_in(j, Q(j,:) == 1) = (c(j)-g(j))/sum(Q(j,:) == 1);
end
beta_true(:, 2:end) = beta_part_in;

gamma_true = 0.5*ones(J, 1);

% simulate data and fit
n_sim = 100;
nu_est = cell(n_sim,1);
beta_est = cell(n_sim,1);
gamma_est = cell(n_sim,1);
loglik = zeros(n_sim,1);

parpool(4)

parfor (i = 1:n_sim, 4)
    [X, ~] = generate_X_ACDM_negbin(N, prop_true, beta_true, Q, A, gamma_true);
    nu_in = drchrnd(ones(2^K,1)',1)';

    beta_in = zeros(J, K + 1);
    beta_part_in = zeros(J, K);
    for j = 1:J
        beta_in(j, 1) = 1.5*rand(1); 
        beta_part_in(j, Q(j,:) == 1) = rand(1);
    end
    beta_in(:, 2:end) = beta_part_in;
    p_in = 0.1 + 0.6*rand(J,1);

    [nu_est{i}, beta_est{i}, gamma_est{i}, loglik(i), phi] = get_EM_ACDM_negbin(X, Q, A_in, nu_in, beta_in, p_in);
    fprintf('Simulation number %d completed \n', i);
end

% computing performance measures
err_nu = 0;
err_beta = 0;
err_gamma = 0;

for i = 1:n_sim
    err_nu = err_nu + mean((nu_true - nu_est{i}).^2);
    err_beta = err_beta + sum((beta_true - beta_est{i}).^2, 'all')/(J + sum(Q,'all'));
    err_gamma = err_gamma + mean((gamma_true - gamma_est{i}).^2);
end

err_nu = sqrt(err_nu / n_sim);
err_beta = sqrt(err_beta / n_sim);
err_gamma = sqrt(err_gamma / n_sim);


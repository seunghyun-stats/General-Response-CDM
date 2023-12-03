% This code was used for the simulation studies in Section 5
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


%% Simulation under the lognormal-DINA model
% set item parameters
mu_0 = ones(J,1) * (-1);
sigma_0 = ones(J,1) * 1;
mu_1 = ones(J,1) * 2;
sigma_1 = ones(J,1) * 1;

% simulate data and fit
n_sim = 100;

nu_est = cell(n_sim,1);
mu_0_est = cell(n_sim,1);
mu_1_est = cell(n_sim,1);
sigma_0_est = cell(n_sim,1);
sigma_1_est = cell(n_sim,1);
time = zeros(n_sim,1);
itera = zeros(n_sim, 1);

rng(2023);

parpool(4)
parfor (i = 1:n_sim, 4)
    [X, ~] = generate_X_DINA_lognormal(N, prop_true, mu_0, mu_1, sigma_0, sigma_1, Q);
    
    nu_in = drchrnd(ones(2^K,1)',1);
    mu_0_in = - 0.5 - rand(J,1);
    mu_1_in = 1 + rand(J,1);

    tic;
    [nu_est{i}, mu_0_est{i}, mu_1_est{i}, sigma_0_est{i}, sigma_1_est{i}, ~, itera(i)] = get_EM_DINA_lognormal(X, Q, A, nu_in, ...
        mu_0_in, mu_1_in, sigma_0, sigma_1);
    fprintf('Simulation number %d completed \n', i);
    time(i) = toc;
end

mean(itera)
mean(time)

err_nu = 0;
err_mu = 0;
err_sigma = 0;
err_all = 0;

for i = 1:n_sim
    err_nu = err_nu + mean((nu_true - nu_est{i}).^2);
    err_mu = err_mu + (mean((mu_0 - mu_0_est{i}).^2, 'all') + ...
        mean((mu_1 - mu_1_est{i}).^2, 'all')) / 2;
    err_sigma = err_sigma + (mean((sigma_0 - sigma_0_est{i}).^2, 'all') + ...
        mean((sigma_1 - sigma_1_est{i}).^2, 'all')) / 2;
    err_all = (err_mu + err_sigma) / 2;
end

err_nu = sqrt(err_nu / n_sim);
err_mu = sqrt(err_mu / n_sim);
err_sigma = sqrt(err_sigma / n_sim);
err_all = sqrt(err_all / n_sim);


%% Simulation under the negbin-DINA model
% set item parameters
r_0 = 1*ones(J,1);
r_1 = 3*ones(J,1);
p_0 = 0.5*ones(J,1);
p_1 = 0.5*ones(J,1);


% simulate data and fit
n_sim = 100;
nu_est = cell(n_sim,1);
r_0_est = cell(n_sim,1);
r_1_est = cell(n_sim,1);
p_0_est = cell(n_sim,1);
p_1_est = cell(n_sim,1);

parpool(4)

parfor (i = 1:n_sim, 4)
    rng(i);
    [X, X_A] = generate_X_DINA_negbin(N, nu_true, r_0, r_1, p_0, p_1, Q);

    nu_in = drchrnd(5*ones(2^K,1)',1)';
    r_0_in = r_0 - 0.1 + 0.2*rand(J,1);
    r_1_in = r_1 - 0.1 + 0.2*rand(J,1);
    p_0_in = p_0;
    p_1_in = p_1;

    [nu_est{i}, r_0_est{i}, r_1_est{i}, p_0_est{i}, p_1_est{i}, phi, ~] = get_EM_DINA_negbin(X, ...
        Q, A, nu_in, r_0_in, r_1_in, p_0_in, p_1_in);
    fprintf('Simulation number %d completed \n', i);
end

err_nu = 0;
err_r = 0;
err_p = 0;

for i = 1:n_sim
    err_nu = err_nu + mean((nu_true - nu_est{i}).^2);
    err_r = err_r + (mean((r_0 - r_0_est{i}).^2, 'all') + mean((r_1 - r_1_est{i}).^2, 'all'))/2;
    err_p = err_p + (mean((p_0 - p_0_est{i}).^2, 'all') + mean((p_1 - p_1_est{i}).^2, 'all'))/2;
end

err_beta = (err_r + err_p)/2;

err_nu = sqrt(err_nu / n_sim);
err_r = sqrt(err_r / n_sim);
err_p = sqrt(err_p / n_sim);
err_beta = sqrt(err_beta / n_sim);


%% Simulation under the Poisson-DINA model
% set item parameters
lam_0 = 1*ones(J,1);
lam_1 = 3*ones(J,1);

err_A_vec = zeros(n_sim,1);
err_A_el = zeros(n_sim,1);

parpool(4)

parfor (i = 1:n_sim, 4)
    rng(i);
    [X, X_A] = generate_X_DINA_poisson(N, nu_true, lam_0, lam_1, Q);

    nu_in = drchrnd(5*ones(2^K,1)',1)';
    lam_0_in = 0.5 + rand(J,1);
    lam_1_in = lam_0_in + 1.5 + rand(J,1);

    [nu_est{i}, lam_0_est{i}, lam_1_est{i}, phi, ~] = get_EM_DINA_poisson(X, ...
        Q, A, nu_in, lam_0_in, lam_1_in);
    fprintf('Simulation number %d completed \n', i);
end

err_nu = 0;
err_lam = 0;

for i = 1:n_sim
    err_nu = err_nu + mean((nu_true - nu_est{i}).^2);
    err_lam = err_lam + (mean((lam_0 - lam_0_est{i}).^2, 'all') + mean((lam_1 - lam_1_est{i}).^2, 'all'))/2;
end

err_nu = sqrt(err_nu / n_sim);
err_lam = sqrt(err_lam / n_sim);

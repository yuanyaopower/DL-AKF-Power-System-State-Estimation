function M = forceSPD(M, eps_min)
% forceSPD
% Symmetrize a covariance matrix and add progressively increasing diagonal
% jitter when needed to obtain a positive-definite matrix.

if nargin < 2
    eps_min = 1e-12;
end

M = 0.5 * (M + M');

[~, p] = chol(M, 'upper');
if p == 0
    return;
end

jit = eps_min;

while p ~= 0 && jit <= 1e-2
    M = M + jit * eye(size(M));
    [~, p] = chol(M, 'upper');
    jit = jit * 10;
end

M = 0.5 * (M + M');

end

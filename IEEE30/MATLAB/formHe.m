function He = formHe(x, G, B, n, L, line_equ, slack, U_idx, theta_idx, num_U, num_theta, gen_bus)
% formHe
% Jacobian of the IEEE 30-bus measurement model used by the EKF/AKF
% baselines.
%
% Measurement vector:
%   [voltage magnitudes at U_idx;
%    active-power injections at all buses;
%    reactive-power injections at all buses]
%
% State vector:
%   [phase angles at theta_idx;
%    voltage magnitudes at U_idx]

theta = zeros(n, 1);
theta(theta_idx) = x(1:num_theta);

u = ones(n, 1);
u(U_idx) = x(num_theta + 1:end);

% Full Jacobian with respect to all bus angles and voltage magnitudes.
He = zeros(3*n, 2*n);

% Voltage-magnitude measurements.
He(1:n, n + 1:2*n) = eye(n);

% Power-injection Jacobian blocks.
H = zeros(n, n);
M = zeros(n, n);
N = zeros(n, n);
L_1 = zeros(n, n);

for i = 1:n
    for j = 1:n
        if j == i
            for t1 = 1:n
                H(i,j) = H(i,j) + u(i)*u(t1) * ...
                    (-G(i,t1)*sin(theta(i)-theta(t1)) + ...
                      B(i,t1)*cos(theta(i)-theta(t1)));

                M(i,j) = M(i,j) + u(i)*u(t1) * ...
                    ( G(i,t1)*cos(theta(i)-theta(t1)) + ...
                      B(i,t1)*sin(theta(i)-theta(t1)));

                N(i,j) = N(i,j) + u(t1) * ...
                    ( G(i,t1)*cos(theta(i)-theta(t1)) + ...
                      B(i,t1)*sin(theta(i)-theta(t1)));

                L_1(i,j) = L_1(i,j) + u(t1) * ...
                    ( G(i,t1)*sin(theta(i)-theta(t1)) - ...
                      B(i,t1)*cos(theta(i)-theta(t1)));
            end

            H(i,j) = -u(i)^2 * B(i,i) + H(i,j);
            M(i,j) = -u(i)^2 * G(i,i) + M(i,j);
            N(i,j) = N(i,j) + u(i) * G(i,i);
            L_1(i,j) = L_1(i,j) - u(i) * B(i,i);

        else
            H(i,j) = u(i)*u(j) * ...
                ( G(i,j)*sin(theta(i)-theta(j)) - ...
                  B(i,j)*cos(theta(i)-theta(j)));

            M(i,j) = u(i)*u(j) * ...
                (-G(i,j)*cos(theta(i)-theta(j)) - ...
                  B(i,j)*sin(theta(i)-theta(j)));

            N(i,j) = u(i) * ...
                ( G(i,j)*cos(theta(i)-theta(j)) + ...
                  B(i,j)*sin(theta(i)-theta(j)));

            L_1(i,j) = u(i) * ...
                ( G(i,j)*sin(theta(i)-theta(j)) - ...
                  B(i,j)*cos(theta(i)-theta(j)));
        end
    end
end

Je = [H, N; M, L_1];
He(n + 1:3*n, :) = Je;

% Remove fixed-voltage generator-bus states and the slack-bus angle.
He(:, n + gen_bus) = [];
He(:, slack) = [];

% Voltage-magnitude measurements are available only at U_idx.
He(gen_bus, :) = [];

end

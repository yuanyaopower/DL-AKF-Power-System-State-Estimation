function He = formHe_large( ...
    x, G, B, n, L, line_equ, slack, U_idx, theta_idx, ...
    num_U, num_theta, gen_bus, U ...
)
% Jacobian of the nonlinear measurement function for the IEEE 118-bus baseline.
%
% Measurement vector:
%   h(x) = [U(U_idx); P; Q]
%
% State vector:
%   x = [theta(theta_idx); U(U_idx)]

theta = zeros(n, 1);
theta(theta_idx) = x(1:num_theta);
theta(slack) = deg2rad(30);

u = U;
u(U_idx) = x(num_theta + 1:end);

He = zeros(3*n, 2*n);

% Voltage-magnitude measurements
He(1:n, n + 1:2*n) = eye(n);

% Jacobian blocks for nodal active/reactive power injections
H = zeros(n, n);
M = zeros(n, n);
N = zeros(n, n);
L_1 = zeros(n, n);

for i = 1:n
    for j = 1:n
        if j == i
            for t1 = 1:n
                H(i,j) = H(i,j) + u(i) * u(t1) * ( ...
                    -G(i,t1) * sin(theta(i) - theta(t1)) ...
                    + B(i,t1) * cos(theta(i) - theta(t1)) ...
                );

                M(i,j) = M(i,j) + u(i) * u(t1) * ( ...
                    G(i,t1) * cos(theta(i) - theta(t1)) ...
                    + B(i,t1) * sin(theta(i) - theta(t1)) ...
                );

                N(i,j) = N(i,j) + u(t1) * ( ...
                    G(i,t1) * cos(theta(i) - theta(t1)) ...
                    + B(i,t1) * sin(theta(i) - theta(t1)) ...
                );

                L_1(i,j) = L_1(i,j) + u(t1) * ( ...
                    G(i,t1) * sin(theta(i) - theta(t1)) ...
                    - B(i,t1) * cos(theta(i) - theta(t1)) ...
                );
            end

            H(i,j) = -u(i)^2 * B(i,i) + H(i,j);
            M(i,j) = -u(i)^2 * G(i,i) + M(i,j);
            N(i,j) = N(i,j) + u(i) * G(i,i);
            L_1(i,j) = L_1(i,j) - u(i) * B(i,i);

        else
            H(i,j) = u(i) * u(j) * ( ...
                G(i,j) * sin(theta(i) - theta(j)) ...
                - B(i,j) * cos(theta(i) - theta(j)) ...
            );

            M(i,j) = u(i) * u(j) * ( ...
                -G(i,j) * cos(theta(i) - theta(j)) ...
                - B(i,j) * sin(theta(i) - theta(j)) ...
            );

            N(i,j) = u(i) * ( ...
                G(i,j) * cos(theta(i) - theta(j)) ...
                + B(i,j) * sin(theta(i) - theta(j)) ...
            );

            L_1(i,j) = u(i) * ( ...
                G(i,j) * sin(theta(i) - theta(j)) ...
                - B(i,j) * cos(theta(i) - theta(j)) ...
            );
        end
    end
end

He(n + 1:3*n, :) = [H, N; M, L_1];

% Remove fixed-voltage states and the slack-bus angle.
He(:, n + gen_bus) = [];
He(:, slack) = [];

% Remove voltage-magnitude measurement rows associated with fixed-voltage buses.
He(gen_bus, :) = [];

end

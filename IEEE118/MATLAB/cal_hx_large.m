function hx = cal_hx_large( ...
    x_e, G_e, B_e, line_equ, numbranch_e, numbus_e, ...
    slack, U_idx, theta_idx, num_U, num_theta, U ...
)
% Nonlinear measurement function for the IEEE 118-bus baseline.
%
% Measurement vector:
%   h(x) = [U(U_idx); P; Q]
%
% State vector:
%   x = [theta(theta_idx); U(U_idx)]

theta = zeros(numbus_e, 1);
theta(theta_idx) = x_e(1:num_theta);
theta(slack) = deg2rad(30);

u = U;
u(U_idx) = x_e(num_theta + 1:end);

p = zeros(numbus_e, 1);
q = zeros(numbus_e, 1);

for i = 1:numbus_e
    for j = 1:numbus_e
        angle_diff = theta(i) - theta(j);

        p(i) = p(i) + u(i) * u(j) * ( ...
            G_e(i,j) * cos(angle_diff) ...
            + B_e(i,j) * sin(angle_diff) ...
        );

        q(i) = q(i) + u(i) * u(j) * ( ...
            G_e(i,j) * sin(angle_diff) ...
            - B_e(i,j) * cos(angle_diff) ...
        );
    end
end

u_i = u(U_idx);
hx = [u_i; p; q];

end

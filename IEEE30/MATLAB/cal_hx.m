function hx = cal_hx(x_e, G_e, B_e, line_equ, numbranch_e, numbus_e, slack, U_idx, theta_idx, num_U, num_theta)
% cal_hx
% Nonlinear measurement function used by the IEEE 30-bus EKF/AKF
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
%
% line_equ and numbranch_e are retained in the interface for consistency
% with the original baseline code.

theta = zeros(numbus_e, 1);
theta(theta_idx) = x_e(1:num_theta);

u = ones(numbus_e, 1);
u(U_idx) = x_e(num_theta + 1:end);

p = zeros(numbus_e, 1);
q = zeros(numbus_e, 1);

% Nodal active/reactive power injections.
for i = 1:numbus_e
    if i == slack
        for j = 1:numbus_e
            if j == slack
                p(i) = p(i) + u(i)*u(j) * ...
                    (G_e(i,j)*cos(0) + B_e(i,j)*sin(0));
                q(i) = q(i) + u(i)*u(j) * ...
                    (G_e(i,j)*sin(0) - B_e(i,j)*cos(0));
            else
                p(i) = p(i) + u(i)*u(j) * ...
                    (G_e(i,j)*cos(-theta(j)) + ...
                     B_e(i,j)*sin(-theta(j)));
                q(i) = q(i) + u(i)*u(j) * ...
                    (G_e(i,j)*sin(-theta(j)) - ...
                     B_e(i,j)*cos(-theta(j)));
            end
        end
    else
        for j = 1:numbus_e
            if j == slack
                p(i) = p(i) + u(i)*u(j) * ...
                    (G_e(i,j)*cos(theta(i)) + ...
                     B_e(i,j)*sin(theta(i)));
                q(i) = q(i) + u(i)*u(j) * ...
                    (G_e(i,j)*sin(theta(i)) - ...
                     B_e(i,j)*cos(theta(i)));
            else
                p(i) = p(i) + u(i)*u(j) * ...
                    (G_e(i,j)*cos(theta(i)-theta(j)) + ...
                     B_e(i,j)*sin(theta(i)-theta(j)));
                q(i) = q(i) + u(i)*u(j) * ...
                    (G_e(i,j)*sin(theta(i)-theta(j)) - ...
                     B_e(i,j)*cos(theta(i)-theta(j)));
            end
        end
    end
end

u_i = u(U_idx);
hx = [u_i; p; q];

end

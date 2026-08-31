# IEEE-118 power-system model used by the second-stage DL-AKF.

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

from sklearn.preprocessing import StandardScaler
from Dual_LSTM import DualHeadLSTM_large

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

systemdata = sio.loadmat('./IEEE118_bus.mat')
bus = systemdata['bus']
branch = systemdata['branch']
Y = systemdata['Ybus']
Bsh = systemdata['Bsh']

n = bus.shape[0]
G = np.real(Y)
B = np.imag(Y)

slack_bus = int(np.flatnonzero(bus[:, 1] == 3)[0])
non_slack_bus = np.flatnonzero(bus[:, 1] != 3)


class Powermodel(nn.Module):
    def __init__(
            self,
            n,
            z_dim,
            G,
            B,
            B_shunt,
            branch,
            T_his,
            predicted_indices,
            device: torch.device = DEVICE
    ):
        super().__init__()

        self.device = device
        self.n = n
        self.z_dim = z_dim
        self.T_his = T_his

        self.register_buffer(
            'G', torch.as_tensor(G, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'B', torch.as_tensor(B, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'B_shunt', torch.as_tensor(B_shunt, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'branch', torch.as_tensor(branch, dtype=torch.float32, device=device)
        )

        self.register_buffer(
            'predicted_indices',
            torch.as_tensor(predicted_indices, dtype=torch.long, device=device)
        )
        self.register_buffer(
            'predicted_indices_sorted',
            torch.sort(self.predicted_indices).values
        )

        self.register_buffer(
            'nonindices',
            torch.tensor(
                [i for i in range(n) if i not in set(predicted_indices)],
                dtype=torch.long,
                device=device
            )
        )
        self.register_buffer(
            'slack_bus',
            torch.tensor(slack_bus, dtype=torch.long, device=device)
        )
        self.register_buffer(
            'non_slack_bus',
            torch.tensor(
                [i for i in range(n) if i != slack_bus],
                dtype=torch.long,
                device=device
            )
        )
        self.register_buffer(
            'theta_idx',
            torch.tensor(
                [i for i in range(n, 2 * n) if i != n + slack_bus],
                dtype=torch.long,
                device=device
            )
        )
        self.register_buffer(
            'feature_idx',
            torch.cat([
                self.predicted_indices,
                torch.tensor(
                    [i for i in range(n, 2 * n) if i != n + slack_bus],
                    dtype=torch.long,
                    device=device
                )
            ])
        )

        row_mask = torch.ones(3 * n, dtype=torch.bool, device=device)
        row_mask[self.nonindices] = False
        self.register_buffer('row_mask', row_mask)

        col_mask = torch.ones(2 * n, dtype=torch.bool, device=device)
        col_mask[n + slack_bus] = False
        col_mask[self.nonindices] = False
        self.register_buffer('col_mask', col_mask)

        # Load the pretrained LSTM predictor and scaler metadata.
        # weights_only=False is required because the checkpoint stores
        # scaler metadata in addition to the model state_dict.
        ckpt = torch.load(
            './best_dualhead_LSTM_model_large2.pth',
            map_location=device,
            weights_only=False
        )

        self.fLSTM = DualHeadLSTM_large(
            ckpt['in_dim'], ckpt['hidden_dim']
        ).to(device)
        self.fLSTM.load_state_dict(ckpt['state_dict'])
        self.fLSTM.eval()

        self.scaler_u = StandardScaler()
        self.scaler_u.__dict__ = ckpt['scaler_u']
        self.mean_u = torch.as_tensor(
            self.scaler_u.mean_, device=device, dtype=torch.float32
        )
        self.scale_u = torch.as_tensor(
            self.scaler_u.scale_, device=device, dtype=torch.float32
        )

        self.scaler_theta = StandardScaler()
        self.scaler_theta.__dict__ = ckpt['scaler_theta']
        self.mean_th = torch.as_tensor(
            self.scaler_theta.mean_, device=device, dtype=torch.float32
        )
        self.scale_th = torch.as_tensor(
            self.scaler_theta.scale_, device=device, dtype=torch.float32
        )

        self.U_dim = self.predicted_indices.numel()
        self.theta_dim = n - 1

    @torch.no_grad()
    def measurement_function(self, x: torch.Tensor, x_rea: torch.Tensor):
        x = x.flatten().to(torch.float32)
        x_rea = x_rea.flatten().to(torch.float32)

        u_t_all = x_rea[:self.n].clone()
        u_t_all[self.predicted_indices] = x[:self.U_dim]

        theta_t_all = torch.zeros(
            self.n, dtype=torch.float32, device=self.device
        )
        theta_t_all[self.slack_bus] = torch.deg2rad(
            torch.tensor(30.0, dtype=torch.float32, device=self.device)
        )
        theta_t_all[self.non_slack_bus] = x[self.U_dim:]

        zu = u_t_all[self.predicted_indices_sorted]

        delta_theta = theta_t_all[:, None] - theta_t_all[None, :]
        cos_dth = torch.cos(delta_theta)
        sin_dth = torch.sin(delta_theta)
        uuT = u_t_all[:, None] * u_t_all[None, :]

        zp = (
            uuT * (self.G * cos_dth + self.B * sin_dth)
        ).sum(dim=1).reshape(-1, 1)
        zq = (
            uuT * (self.G * sin_dth - self.B * cos_dth)
        ).sum(dim=1).reshape(-1, 1)

        return torch.cat([zu.reshape(-1, 1), zp, zq], dim=0)

    @torch.no_grad()
    def Jacobian(
            self,
            x: torch.Tensor,
            z_predict: torch.Tensor,
            x_rea: torch.Tensor
    ):
        x = x.flatten().to(torch.float32)
        x_rea = x_rea.flatten().to(torch.float32)
        z_predict = z_predict.to(torch.float32)

        u = x_rea[:self.n].clone()
        u[self.predicted_indices] = x[:self.U_dim]

        theta = torch.zeros(
            self.n, dtype=torch.float32, device=self.device
        )
        theta[self.slack_bus] = torch.deg2rad(
            torch.tensor(30.0, dtype=torch.float32, device=self.device)
        )
        theta[self.non_slack_bus] = x[self.U_dim:]

        dth = theta[:, None] - theta[None, :]
        cosd = torch.cos(dth)
        sind = torch.sin(dth)
        uuT = u[:, None] * u[None, :]

        dP_dU = u[:, None] * (self.G * cosd + self.B * sind)
        dQ_dU = u[:, None] * (self.G * sind - self.B * cosd)
        dP_dth = uuT * (self.G * sind - self.B * cosd)
        dQ_dth = uuT * (-self.G * cosd - self.B * sind)

        Gii = torch.diag(self.G)
        Bii = torch.diag(self.B)
        idx = torch.arange(self.n, device=self.device)

        dP_dU[idx, idx] = u * Gii + dP_dU.sum(dim=1)
        dQ_dU[idx, idx] = -u * Bii + dQ_dU.sum(dim=1)

        Pi_sum = (
            uuT * (-self.G * sind + self.B * cosd)
        ).sum(dim=1) - u ** 2 * Bii
        Qi_sum = (
            uuT * (self.G * cosd + self.B * sind)
        ).sum(dim=1) - u ** 2 * Gii
        dP_dth[idx, idx] = Pi_sum
        dQ_dth[idx, idx] = Qi_sum

        H = torch.zeros(
            3 * self.n,
            2 * self.n,
            dtype=torch.float32,
            device=self.device
        )
        H[:self.n, :self.n] = torch.eye(
            self.n, device=self.device, dtype=torch.float32
        )
        H[self.n:2 * self.n, :self.n] = dP_dU
        H[2 * self.n:, :self.n] = dQ_dU
        H[self.n:2 * self.n, self.n:] = dP_dth
        H[2 * self.n:, self.n:] = dQ_dth

        return H[self.row_mask][:, self.col_mask]

    def LSTM_predict_y(self, x_sequence):
        U = x_sequence[:self.U_dim, :]
        theta = x_sequence[self.U_dim:, :]
        Tseq = U.shape[1]

        vm_set_t = torch.as_tensor(
            bus[:, 7], dtype=torch.float32, device=self.device
        ).view(self.n, 1)

        U_full = vm_set_t.repeat(1, Tseq)
        U_full[self.predicted_indices, :] = U

        theta_full = torch.zeros((self.n, Tseq), device=self.device)
        theta_full[self.slack_bus, :] = float(np.deg2rad(30.0))
        theta_full[self.non_slack_bus, :] = theta

        mu_u = torch.as_tensor(
            self.mean_u, dtype=torch.float32, device=self.device
        ).view(self.n, 1)
        sc_u = torch.as_tensor(
            self.scale_u, dtype=torch.float32, device=self.device
        ).view(self.n, 1)
        mu_th = torch.as_tensor(
            self.mean_th, dtype=torch.float32, device=self.device
        ).view(self.n, 1)
        sc_th = torch.as_tensor(
            self.scale_th, dtype=torch.float32, device=self.device
        ).view(self.n, 1)

        U_norm = (U_full - mu_u) / sc_u
        theta_norm = (theta_full - mu_th) / sc_th
        x_tensor_input = torch.vstack(
            (U_norm, theta_norm)
        ).T.unsqueeze(0).float()

        out_U, out_theta = self.fLSTM(x_tensor_input)

        U_full2 = vm_set_t.clone()
        U_full2[self.predicted_indices, :] = out_U.T

        theta_full2 = torch.zeros((self.n, 1), device=self.device)
        theta_full2[self.slack_bus, 0] = float(np.deg2rad(30.0))
        theta_full2[self.non_slack_bus, :] = out_theta.T

        U_real = U_full2 * sc_u + mu_u
        theta_real = theta_full2 * sc_th + mu_th

        x_pred_real = torch.vstack((U_real, theta_real))
        return x_pred_real[self.feature_idx, :]

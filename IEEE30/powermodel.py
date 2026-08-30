# Power-system measurement model, Jacobian, and LSTM prior prediction.

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from Dual_LSTM import DualHeadLSTM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Powermodel(nn.Module):

      def __init__(self, n, z_dim, G, B, B_shunt, branch, T_his, predicted_indices, device: torch.device = DEVICE):
          super().__init__()

          self.device = device

          ckpt = torch.load('best_dualhead_LSTM_model.pth',
                            map_location=device,
                            weights_only = False,
                            )
          self.fLSTM = DualHeadLSTM(
              in_dim=ckpt['in_dim'],
              hidden_dim=ckpt['hidden_dim'],
          ).to(device)

          self.fLSTM.load_state_dict(ckpt['state_dict'])
          self.fLSTM.eval()  # The pretrained LSTM remains fixed during DL-AKF training.

          # Only input-state Jacobians are required during DL-AKF training.
          # Freezing the pretrained LSTM parameters avoids unnecessary parameter-gradient tracking.
          for param in self.fLSTM.parameters():
              param.requires_grad_(False)

          self.scaler_u = StandardScaler()
          self.scaler_theta = StandardScaler()
          self.scaler_u.__dict__ = ckpt['scaler_u']
          self.scaler_theta.__dict__ = ckpt['scaler_theta']

          self.n = n    # Number of system nodes

          self.G = G    # Bus admittance matrix G+j*B
          self.B = B
          self.B_shunt = B_shunt
          self.branch = branch

          self.n_branch = self.branch.shape[0]

          self.z_dim = z_dim

          self.z = None
          self.H = None
          self.T_his = T_his

          self.predicted_indices = predicted_indices
          self.U_dim = predicted_indices.shape[0]
          self.theta_dim = n - 1

          self.nonindices =  [i for i in range(self.n) if i not in set(predicted_indices)]

          self.theta_idx = list(range(self.n + 1, 2 * self.n))  # 非参考节点的相角（跳过 θ₀）
          self.feature_idx = list(self.predicted_indices) + self.theta_idx  # 拼接

          self.mean_u = torch.as_tensor(self.scaler_u.mean_, device=DEVICE)
          self.scale_u = torch.as_tensor(self.scaler_u.scale_, device=DEVICE)
          self.mean_th = torch.as_tensor(self.scaler_theta.mean_, device=DEVICE)
          self.scale_th = torch.as_tensor(self.scaler_theta.scale_, device=DEVICE)

      def measurement_function(self, x):

          x = x.detach().cpu().numpy()

          self.u_t_all = np.ones([self.n, 1])
          self.theta_t_all = np.zeros([self.n, 1])

          self.u_t_all[self.predicted_indices, :] = x[:self.U_dim, :]
          self.theta_t_all[1:, :] = x[self.U_dim:, :]

          # Voltage amplitude
          zu = x[:self.U_dim, :]

          # Bus active & reactive injection power
          zp, zq = [], []  # 节点有功 / 无功 注入列表
          for i in range(self.n):
              P_i, Q_i = 0.0, 0.0
              for j in range(self.n):
                  dth = self.theta_t_all[i] - self.theta_t_all[j]
                  uiu_j = self.u_t_all[i] * self.u_t_all[j]

                  P_i += uiu_j * (self.G[i, j] * np.cos(dth) + self.B[i, j] * np.sin(dth))
                  Q_i += uiu_j * (self.G[i, j] * np.sin(dth) - self.B[i, j] * np.cos(dth))

              zp.append(P_i)
              zq.append(Q_i)

          z = np.concatenate((zu, zp, zq), axis=0)

          z = torch.from_numpy(z).float().to(DEVICE)

          return z

      def Jacobian(self, x):

          x = x.detach().cpu().numpy()

          u_t_all = np.ones([self.n, 1])
          theta_t_all = np.zeros([self.n, 1])

          u_t_all[self.predicted_indices, :] = x[:self.U_dim].reshape(-1, 1)
          theta_t_all[1:, :] = x[self.U_dim:].reshape(-1, 1)

          H = np.zeros((3*self.n, self.n * 2))

          # Voltage amplitude
          for i in range(self.n):
              H[i, i] = 1

          # Injection active & reactive power
          # ---------- n～3n‑1 行：节点注入功率量测 ----------
          for i in range(self.n):
              u_i = u_t_all[i]
              th_i = theta_t_all[i]
              G_ii = self.G[i, i]
              B_ii = self.B[i, i]

              rowP = self.n + i  # P_i 量测所在行
              rowQ = 2 * self.n + i  # Q_i 量测所在行

              # 1) 对自身幅值 U_i 的偏导（先放自项，再累加交叉项）
              dP_du_i = 2.0 * u_i * G_ii
              dQ_du_i = -2.0 * u_i * B_ii

              # 2) 预累加求角度偏导所需的 Σ 项
              Pi_sum = 0.0
              Qi_sum = 0.0

              for j in range(self.n):
                  u_j = u_t_all[j]
                  th_j = theta_t_all[j]

                  G_ij = self.G[i, j]
                  B_ij = self.B[i, j]

                  dth = th_i - th_j
                  cosd = np.cos(dth)
                  sind = np.sin(dth)

                  if j == i:  # 自身对角元素已在 dP_du_i / dQ_du_i 中处理
                      continue

                  # ---------- 非对角元 j ≠ i ----------
                  # (a) 对 U_j 的偏导
                  H[rowP, j] = u_i * (G_ij * cosd + B_ij * sind)
                  H[rowQ, j] = u_i * (G_ij * sind - B_ij * cosd)

                  # (b) 对 θ_j 的偏导
                  H[rowP, self.n + j] = u_i * u_j * (G_ij * sind - B_ij * cosd)
                  H[rowQ, self.n + j] = u_i * u_j * (-G_ij * cosd - B_ij * sind)

                  # (c) 同时把交叉项累加到自身幅值偏导
                  dP_du_i += u_j * (G_ij * cosd + B_ij * sind)
                  dQ_du_i += u_j * (G_ij * sind - B_ij * cosd)

                  # Σ 项（角度偏导用）
                  Pi_sum += u_i * u_j * (-G_ij * sind + B_ij * cosd)
                  Qi_sum += u_i * u_j * (G_ij * cosd + B_ij * sind)

              # 3) 最后一次性写入自身列（幅值）
              H[rowP, i] = dP_du_i
              H[rowQ, i] = dQ_du_i

              # 4) 自身角度 θ_i 的偏导
              H[rowP, self.n + i] = Pi_sum
              H[rowQ, self.n + i] = Qi_sum

          # Delete the information with reference bus
          H1 = H
          # 0-row
          H1 = np.delete(H1, self.nonindices, axis=0)
          # n-col
          H1 = np.delete(H1, self.n, axis=1)
          # 0-col
          H1 = np.delete(H1, self.nonindices, axis=1)

          H1 = torch.from_numpy(H1).float().to(self.device)

          return H1

      # LSTM prediction used for both prior estimation and Jacobian evaluation.
      def LSTM_predict_y(self, x_sequence):

          # --- 拆分 ---
          U = x_sequence[: self.U_dim, :]  # [n_u, T]
          theta = x_sequence[self.U_dim:, :]  # [n_th, T]

          U_full = torch.ones((self.n, U.shape[1]), device=DEVICE)
          theta_full = torch.zeros((self.n, theta.shape[1]), device=DEVICE)
          U_full[self.predicted_indices, :] = U  # no detach!
          theta_full[1:, :] = theta

          U_norm = (U_full - self.mean_u[:, None]) / self.scale_u[:, None]
          theta_norm = (theta_full - self.mean_th[:, None]) / self.scale_th[:, None]

          x_sequence_norm = torch.vstack((U_norm, theta_norm))  # [2n, T]

          # --- LSTM 前向 ---
          x_tensor_input = x_sequence_norm.T.unsqueeze(0)  # [1, T, 2n]
          x_tensor_input = x_tensor_input.to(torch.float32)
          out_U, out_theta = self.fLSTM(x_tensor_input)  # 仍带梯度

          # --- 反归一化 ---
          U_full2 = torch.ones((self.n, 1), device=DEVICE)
          theta_full2 = torch.zeros((self.n, 1), device=DEVICE)
          U_full2[self.predicted_indices, :] = out_U.T
          theta_full2[1:, :] = out_theta.T

          U_real = U_full2 * self.scale_u[:, None] + self.mean_u[:, None]
          theta_real = theta_full2 * self.scale_th[:, None] + self.mean_th[:, None]

          x_pred_real = torch.vstack((U_real, theta_real))
          x_pred = x_pred_real[self.feature_idx, :]

          return x_pred


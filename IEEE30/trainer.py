import os
import torch
import config
import numpy as np
import scipy.io as sio

from Filter_yuan import QR_Filter
from tester_adj import Tester_QR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = config.train_config['batch_size']
alter_period = config.train_config['alter_period']
training_period = config.train_config['training_period']
lr = config.train_config['learning_rate']
wd = config.train_config['weight_decay']

print_num = 5
save_num = 5

T_his = np.load('T_his.npy')
fixed_u = np.load('fixed_u.npy')
predicted_indices = np.setdiff1d(np.arange(30), fixed_u)
n = 30

x_tr_stats = sio.loadmat('./powersystemdata3/x_tr_stats.mat')
x_tr_mean = x_tr_stats['x_tr_mean']
x_tr_std = x_tr_stats['x_tr_std']

U_mean = x_tr_mean[predicted_indices, :]
U_std = x_tr_std[predicted_indices, :]

th_mean = x_tr_mean[n + 1:, :]
th_std = x_tr_std[n + 1:, :]

U_dim = U_mean.shape[0]
th_dim = th_mean.shape[0]

U_mean = torch.from_numpy(U_mean).to(DEVICE)
th_mean = torch.from_numpy(th_mean).to(DEVICE)
U_std = torch.from_numpy(U_std).to(DEVICE)
th_std = torch.from_numpy(th_std).to(DEVICE)

U_mean = U_mean.view(1, U_dim, 1)
th_mean = th_mean.view(1, th_dim, 1)
U_std = U_std.view(1, U_dim, 1)
th_std = th_std.view(1, th_dim, 1)

U_mean = U_mean.repeat(batch_size, 1, 1)
th_mean = th_mean.repeat(batch_size, 1, 1)
U_std = U_std.repeat(batch_size, 1, 1)
th_std = th_std.repeat(batch_size, 1, 1)


def mse(target, predicted):
    """Mean Squared Error"""
    return torch.mean((target - predicted) ** 2)


def empirical_averaging_all(target, predicted_mean, predicted_cov, beta=0.05):
    L1 = mse(target, predicted_mean)

    err = target - predicted_mean

    n_batch = predicted_cov.shape[0]
    n_time = predicted_cov.shape[1]
    E_cov = 0
    for i in range(n_time):
        for j in range(n_batch):
            E_cov += torch.sum(
                torch.abs(
                    err[j, :, i:i + 1] @ err[j, :, i:i + 1].T
                    - predicted_cov[j, i, :]
                )
            )

    L2 = E_cov / (n_batch * n_time)

    return (1 - beta) * L1 + beta * L2, L1, L2


class Trainer_R():

    def __init__(
            self,
            dnn: QR_Filter,
            data_path,
            save_path,
            data_file='z.pt',
            mode=0,
            device=DEVICE
    ):

        self.save_num = save_num
        self.device = device

        self.training_period = training_period
        self.alter_period = alter_period
        self.dnn = dnn
        self.x_dim = self.dnn.x_dim
        self.z_dim = self.dnn.z_dim

        self.loss_best = 1e4

        self.data_path = data_path
        self.save_path = save_path
        self.data_file = data_file

        self.data_x = torch.load(
            os.path.join(data_path, 'x.pt'),
            map_location=self.device
        )
        self.data_z = torch.load(
            os.path.join(data_path, data_file),
            map_location=self.device
        )

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

        self.T_his = T_his
        self.T_bac = 1

        self.predicted_indices = predicted_indices
        self.n = n

        # The active power model is already owned by self.dnn.Powermodel.
        # Do not construct a second unused Powermodel/LSTM here.

        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.dnn.kf_net.parameters(),
            lr=lr,
            weight_decay=wd
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=50,
            min_lr=1e-6
        )

        cal_num_param = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(cal_num_param(self.dnn.kf_net))

        self.batch_size = batch_size
        self.alter_num = alter_period

        self.train_count = 0
        self.data_idx = 0

        self.theta_idx = list(range(self.n + 1, 2 * self.n))
        self.feature_idx = list(self.predicted_indices) + self.theta_idx

        # Validation and early stopping are configured from Main.
        self.val_data_path = None
        self.val_data_file = self.data_file
        self.val_every = None
        self.val_patience = None
        self.val_min_delta = None
        self.val_wait = 0
        self.best_rmse_sum = float('inf')
        self.stop_training = False

        self.monitor_every = print_num
        self.monitor_dir = './monitor_yuan/' + self.save_path[:-3]
        os.makedirs(self.monitor_dir, exist_ok=True)

        self.monitor_floor = 1e-5
        self.monitor_floor_tol = 1.1

    def _monitor_stats(self, tensor, floor=None):
        x = tensor.detach().flatten()
        stats = {
            'min': float(x.min().cpu()),
            'p5': float(torch.quantile(x, 0.05).cpu()),
            'median': float(torch.quantile(x, 0.50).cpu()),
            'mean': float(x.mean().cpu()),
            'p95': float(torch.quantile(x, 0.95).cpu()),
            'max': float(x.max().cpu()),
        }

        if floor is not None:
            stats['lower_ratio'] = float(
                (x < self.monitor_floor_tol * floor).float().mean().cpu()
            )

        return stats

    def _build_validation_filter(self):
        """Build an independent validation filter with the current QRNet weights."""
        val_filter = type(self.dnn)(
            self.dnn.Powermodel,
            self.n,
            self.x_dim,
            self.z_dim,
            self.predicted_indices,
            **(
                self.dnn.get_init_kwargs()
                if hasattr(self.dnn, 'get_init_kwargs')
                else {}
            ),
            device=self.device
        )

        # This is equivalent to save -> load -> reset for the trainable QRNet
        # parameters/buffers, while avoiding validation checkpoint disk I/O.
        val_filter.kf_net.load_state_dict(self.dnn.kf_net.state_dict())

        return val_filter

    def train_one_epoch(self):

        self.dnn.kf_net.train()

        if self.data_idx == 0:
            self.dnn.reset(clean_history=True)

        while self.data_idx < self.data_num:

            end_idx = min(
                self.data_idx + self.batch_size,
                len(self.data_x)
            )
            cur_bs = end_idx - self.data_idx

            batch_x = self.data_x[
                self.data_idx:end_idx,
                self.feature_idx,
                :
            ]
            batch_z = self.data_z[self.data_idx:end_idx]

            # The original code prints/saves training monitors every five batches.
            # Allocate and populate these auxiliary tensors only for those batches.
            do_monitor = (
                (self.train_count + 1) % self.monitor_every == 0
            )

            loss_yy = torch.zeros(
                cur_bs,
                self.seq_len - self.T_his,
                device=self.device
            )

            if do_monitor:
                x_hat = torch.zeros(
                    cur_bs,
                    self.x_dim,
                    self.seq_len - self.T_his,
                    device=self.device
                )

                loss_yy1 = torch.zeros(
                    cur_bs,
                    self.seq_len - self.T_his,
                    device=self.device
                )
                loss_yy2 = torch.zeros_like(loss_yy1)
                loss_yy3 = torch.zeros_like(loss_yy1)
                loss_yy4 = torch.zeros_like(loss_yy1)

                Q_diag_yy = torch.zeros(
                    cur_bs,
                    self.seq_len - self.T_his,
                    self.x_dim,
                    device=self.device
                )
                R_diag_yy = torch.zeros(
                    cur_bs,
                    self.seq_len - self.T_his,
                    self.z_dim,
                    device=self.device
                )

                Q_input_yy = torch.zeros(
                    cur_bs,
                    self.seq_len - self.T_his,
                    self.x_dim,
                    device=self.device
                )
                R_input_yy = torch.zeros(
                    cur_bs,
                    self.seq_len - self.T_his,
                    self.z_dim,
                    device=self.device
                )

                z_meas_yy = torch.zeros(
                    cur_bs,
                    self.seq_len - self.T_his,
                    self.z_dim,
                    device=self.device
                )

            for i in range(cur_bs):

                x_sequence = batch_x[i, :, :self.T_his]
                x_sequence_all = batch_x[i, :, :]

                z_past_sequence = batch_z[i, :, :self.T_his]
                x_predict = self.dnn.initial_predict(x_sequence)

                residual_q = (
                    batch_x[i, :, self.T_his].reshape((-1, 1))
                    - x_predict
                )
                residual_q = residual_q.expand(-1, self.T_bac)

                residual_z = (
                    batch_z[i, :, self.T_his].reshape((-1, 1))
                    - self.dnn.Powermodel.measurement_function(x_predict)
                )
                residual_z = residual_z.expand(-1, self.T_bac)

                for ii in range(self.T_his, self.seq_len):
                    self.dnn.filtering(
                        batch_z[i, :, ii].reshape((-1, 1)),
                        z_past_sequence,
                        x_sequence,
                        x_sequence_all[:, ii],
                        residual_q,
                        residual_z
                    )

                    t_idx = ii - self.T_his
                    loss_yy[i, t_idx] = self.dnn.loss

                    if do_monitor:
                        loss_yy1[i, t_idx] = self.dnn.quad
                        loss_yy2[i, t_idx] = self.dnn.logdet
                        loss_yy3[i, t_idx] = self.dnn.quad_q
                        loss_yy4[i, t_idx] = self.dnn.logdet_q

                        Q_diag_yy[i, t_idx] = self.dnn.monitor_q_diag
                        R_diag_yy[i, t_idx] = self.dnn.monitor_r_diag

                        Q_input_yy[i, t_idx] = self.dnn.monitor_q_input
                        R_input_yy[i, t_idx] = self.dnn.monitor_r_input

                        z_meas_yy[i, t_idx] = self.dnn.monitor_z.reshape(-1)

                if do_monitor:
                    x_hat[i] = self.dnn.state_history[
                        :,
                        -(self.seq_len - self.T_his):
                    ]

                self.dnn.detach_continuous_state()

            if do_monitor:
                x_true = batch_x[:, :, self.T_his:]
                x_pred = x_hat

                U_true = x_true[:, :U_dim, :]
                U_pred = x_pred[:, :U_dim, :]
                loss_u = mse(U_true, U_pred)

                th_true = x_true[:, U_dim:, :]
                th_pred = x_pred[:, U_dim:, :]
                loss_th = mse(th_true, th_pred)

            loss = loss_yy.mean()

            # One zero_grad is sufficient before backward.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_count += 1
            self.data_idx = end_idx

            if self.data_idx < self.data_num:
                self.dnn.reset_kf_only(clean_history=True)

            # Periodic validation and early stopping.
            if (
                    self.val_every is not None
                    and self.val_data_path is not None
                    and self.train_count % self.val_every == 0
            ):
                ckpt_path = (
                    './model_saved_yuan/'
                    + self.save_path[:-3]
                    + '_'
                    + str(self.train_count)
                    + '.pt'
                )

                val_filter = self._build_validation_filter()

                self.validate(
                    Tester_QR(
                        filter=val_filter,
                        data_path=self.val_data_path,
                        data_file=self.val_data_file,
                        model_path=ckpt_path,
                        is_validation=True,
                        is_mismatch=False,
                        device=self.device,
                        load_model=False,
                        collect_monitor=False
                    )
                )

                if self.stop_training:
                    # The original implementation saved the current periodic
                    # checkpoint before validation. Preserve that artifact at
                    # an early-stopping validation point without doing the
                    # save/load round trip for every validation.
                    if self.train_count % self.save_num == 0:
                        try:
                            torch.save(self.dnn.kf_net, ckpt_path)
                        except Exception:
                            pass
                    return

            if self.train_count % self.save_num == 0:
                try:
                    torch.save(
                        self.dnn.kf_net,
                        './model_saved_yuan/'
                        + self.save_path[:-3]
                        + '_'
                        + str(self.train_count)
                        + '.pt'
                    )
                except Exception:
                    print('here')
                    pass

            if do_monitor:
                loss_q_mean = 0.5 * (
                    loss_yy3.mean() + loss_yy4.mean()
                )
                loss_r_mean = 0.5 * (
                    loss_yy1.mean() + loss_yy2.mean()
                )

                q_stats = self._monitor_stats(
                    Q_diag_yy,
                    floor=self.monitor_floor
                )
                r_stats = self._monitor_stats(
                    R_diag_yy,
                    floor=self.monitor_floor
                )
                qin_stats = self._monitor_stats(Q_input_yy)
                rin_stats = self._monitor_stats(R_input_yy)

                print(
                    f'[Model{self.save_path}] [Train{self.train_count}] '
                    f'loss = {loss: .6f}  '
                    f'loss_q = {loss_q_mean: .6f}  '
                    f'quad_q = {loss_yy3.mean(): .6f}  '
                    f'logdet_q = {loss_yy4.mean(): .6f}  '
                    f'loss_r = {loss_r_mean: .6f}  '
                    f'quad = {loss_yy1.mean(): .6f}  '
                    f'logdet = {loss_yy2.mean(): .6f}  '
                    f'loss_u = {loss_u: .10f}  '
                    f'loss_th = {loss_th: .10f}'
                )

                print(
                    f'    Q_diag: min={q_stats["min"]:.3e}, '
                    f'med={q_stats["median"]:.3e}, '
                    f'mean={q_stats["mean"]:.3e}, '
                    f'max={q_stats["max"]:.3e}, '
                    f'lower_ratio={q_stats["lower_ratio"]:.3f}'
                )

                print(
                    f'    R_diag: min={r_stats["min"]:.3e}, '
                    f'med={r_stats["median"]:.3e}, '
                    f'mean={r_stats["mean"]:.3e}, '
                    f'max={r_stats["max"]:.3e}, '
                    f'lower_ratio={r_stats["lower_ratio"]:.3f}'
                )

                monitor_pack = {
                    'train_count': self.train_count,
                    'loss': float(loss.detach().cpu()),
                    'loss_q': float(loss_q_mean.detach().cpu()),
                    'loss_r': float(loss_r_mean.detach().cpu()),
                    'quad_q': float(loss_yy3.mean().detach().cpu()),
                    'logdet_q': float(loss_yy4.mean().detach().cpu()),
                    'quad': float(loss_yy1.mean().detach().cpu()),
                    'logdet': float(loss_yy2.mean().detach().cpu()),
                    'Q_diag': Q_diag_yy.detach().cpu(),
                    'R_diag': R_diag_yy.detach().cpu(),
                    'Q_input': Q_input_yy.detach().cpu(),
                    'R_input': R_input_yy.detach().cpu(),
                    'z_meas': z_meas_yy.detach().cpu(),
                    'Q_diag_stats': q_stats,
                    'R_diag_stats': r_stats,
                    'Q_input_stats': qin_stats,
                    'R_input_stats': rin_stats,
                }

                torch.save(
                    monitor_pack,
                    os.path.join(
                        self.monitor_dir,
                        f'monitor_train_{self.train_count}.pt'
                    )
                )

            if self.data_idx + self.batch_size >= self.data_num:
                self.data_idx = 0
                break

            self.train_loss = loss

    def validate(self, tester):

        valid_loss = tester.loss.item()
        self.scheduler.step(valid_loss)

        rmse_sum = float(tester.rmse_u) + float(tester.rmse_th)

        improved = rmse_sum < (
            self.best_rmse_sum - self.val_min_delta
        )

        if improved:
            self.best_rmse_sum = rmse_sum
            self.val_wait = 0
            try:
                torch.save(
                    tester.filter.kf_net,
                    './model_saved_yuan/'
                    + self.save_path[:-3]
                    + '_best.pt'
                )
                print(
                    f'Save best model at {self.save_path} '
                    f'& train {self.train_count} '
                    f'& rmse_sum = {rmse_sum:.10f}'
                )
            except Exception:
                pass
        else:
            self.val_wait += 1
            print(
                f'No improvement: rmse_sum = {rmse_sum:.10f}, '
                f'wait {self.val_wait}/{self.val_patience}'
            )

        self.valid_loss = valid_loss
        self.valid_x_hat = tester.x_hat

        if self.val_wait >= self.val_patience:
            self.stop_training = True

    def configure_val_early_stop(
            self,
            val_data_path,
            val_every,
            patience=3,
            min_delta=0.0,
            val_data_file=None
    ):
        self.val_data_path = val_data_path
        self.val_data_file = (
            self.data_file
            if val_data_file is None
            else val_data_file
        )
        self.val_every = val_every
        self.val_patience = patience
        self.val_min_delta = min_delta
        self.val_wait = 0
        self.best_rmse_sum = float('inf')
        self.stop_training = False

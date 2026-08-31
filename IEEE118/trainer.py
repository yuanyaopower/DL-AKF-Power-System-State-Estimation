# IEEE-118 Stage-2 trainer for DL-AKF / DL-AKF(Holt).

import os
import torch
import config

from Filter import QR_Filter
from tester import Tester_QR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = config.train_config['batch_size']
lr = config.train_config['learning_rate']
wd = config.train_config['weight_decay']

print_num = 5
save_num = 5


def mse(target, predicted):
    return torch.mean((target - predicted) ** 2)


class Trainer_R:
    def __init__(
            self,
            dnn: QR_Filter,
            data_path,
            save_path,
            data_file='z.pt',
            mode=0,
            device=DEVICE
    ):
        self.device = device
        self.dnn = dnn
        self.x_dim = self.dnn.x_dim
        self.z_dim = self.dnn.z_dim
        self.n = self.dnn.n
        self.T_his = self.dnn.T_his
        self.T_bac = 1

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

        self.predicted_indices = self.dnn.predicted_indices
        self.U_dim = len(self.predicted_indices)
        self.theta_idx = (
            self.dnn.Powermodel.theta_idx.detach().cpu().tolist()
        )
        self.feature_idx = (
            self.dnn.Powermodel.feature_idx.detach().cpu().tolist()
        )

        self.optimizer = torch.optim.Adam(
            self.dnn.kf_net.parameters(), lr=lr, weight_decay=wd
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=50,
            min_lr=1e-6
        )

        trainable_params = sum(
            p.numel() for p in self.dnn.kf_net.parameters()
            if p.requires_grad
        )
        print(trainable_params)

        self.batch_size = batch_size
        self.save_num = save_num
        self.train_count = 0
        self.data_idx = 0

        self.val_data_path = None
        self.val_data_file = self.data_file
        self.val_every = None
        self.val_patience = None
        self.val_min_delta = None
        self.val_wait = 0
        self.best_rmse_sum = float('inf')
        self.stop_training = False

        save_stem = os.path.splitext(self.save_path)[0]
        self.model_dir = './model_saved_118'
        self.monitor_dir = os.path.join('./monitor_118', save_stem)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.monitor_dir, exist_ok=True)

        self.monitor_every = print_num
        self.monitor_floor = 1e-5
        self.monitor_floor_tol = 1.1

    def _checkpoint_path(self, suffix):
        stem = os.path.splitext(self.save_path)[0]
        return os.path.join(
            self.model_dir, f'{stem}_{suffix}.pt'
        )

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
                (x < self.monitor_floor_tol * floor)
                .float().mean().cpu()
            )
        return stats

    def _build_validation_filter(self):
        val_filter = type(self.dnn)(
            self.dnn.Powermodel,
            self.n,
            self.x_dim,
            self.z_dim,
            self.predicted_indices,
            **self.dnn.get_init_kwargs(),
            device=self.device
        )
        val_filter.kf_net.load_state_dict(
            self.dnn.kf_net.state_dict()
        )
        return val_filter

    def train_one_epoch(self):
        self.dnn.kf_net.train()

        if self.data_idx == 0:
            self.dnn.reset(clean_history=True)

        while self.data_idx < self.data_num:
            end_idx = min(
                self.data_idx + self.batch_size,
                self.data_num
            )
            cur_bs = end_idx - self.data_idx

            batch_x = self.data_x[
                self.data_idx:end_idx, self.feature_idx, :
            ]
            batch_z = self.data_z[self.data_idx:end_idx]
            batch_x_all = self.data_x[self.data_idx:end_idx, :, :]

            do_monitor = (
                (self.train_count + 1) % self.monitor_every == 0
            )

            # Release gradients from the preceding batch before the new forward.
            self.optimizer.zero_grad(set_to_none=True)

            T_future = self.seq_len - self.T_his
            loss_yy = torch.zeros(
                cur_bs, T_future, device=self.device
            )

            if do_monitor:
                x_hat = torch.zeros(
                    cur_bs, self.x_dim, T_future, device=self.device
                )
                loss_yy1 = torch.zeros(
                    cur_bs, T_future, device=self.device
                )
                loss_yy2 = torch.zeros_like(loss_yy1)
                loss_yy3 = torch.zeros_like(loss_yy1)
                loss_yy4 = torch.zeros_like(loss_yy1)
                Q_diag_yy = torch.zeros(
                    cur_bs, T_future, self.x_dim, device=self.device
                )
                R_diag_yy = torch.zeros(
                    cur_bs, T_future, self.z_dim, device=self.device
                )

            for i in range(cur_bs):
                x_sequence = batch_x[i, :, :self.T_his]
                x_sequence_all = batch_x[i, :, :]
                x_all = batch_x_all[i, :, :]
                z_past_sequence = batch_z[i, :, :self.T_his]

                x_predict = self.dnn.initial_predict(x_sequence)
                residual_q = (
                    batch_x[i, :, self.T_his].reshape((-1, 1))
                    - x_predict
                ).expand(-1, self.T_bac)
                residual_z = (
                    batch_z[i, :, self.T_his].reshape((-1, 1))
                    - self.dnn.Powermodel.measurement_function(
                        x_predict, x_all[:, self.T_his]
                    )
                ).expand(-1, self.T_bac)

                for ii in range(self.T_his, self.seq_len):
                    self.dnn.filtering(
                        batch_z[i, :, ii].reshape((-1, 1)),
                        z_past_sequence,
                        x_sequence,
                        x_sequence_all[:, ii],
                        x_all[:, ii],
                        residual_q,
                        residual_z,
                        record_monitor=do_monitor
                    )

                    t_idx = ii - self.T_his
                    loss_yy[i, t_idx] = self.dnn.loss

                    if do_monitor:
                        loss_yy1[i, t_idx] = self.dnn.quad.detach()
                        loss_yy2[i, t_idx] = self.dnn.logdet.detach()
                        loss_yy3[i, t_idx] = self.dnn.quad_q.detach()
                        loss_yy4[i, t_idx] = self.dnn.logdet_q.detach()
                        Q_diag_yy[i, t_idx] = self.dnn.monitor_q_diag
                        R_diag_yy[i, t_idx] = self.dnn.monitor_r_diag

                if do_monitor:
                    x_hat[i] = self.dnn.state_history[:, -T_future:].detach()

                # Keep the within-batch continuous states but truncate BPTT.
                self.dnn.detach_continuous_state()

            if do_monitor:
                x_true = batch_x[:, :, self.T_his:]
                U_true = x_true[:, :self.U_dim, :]
                U_pred = x_hat[:, :self.U_dim, :]
                th_true = x_true[:, self.U_dim:, :]
                th_pred = x_hat[:, self.U_dim:, :]
                loss_u = mse(U_true, U_pred)
                loss_th = mse(th_true, th_pred)

            loss = loss_yy.mean()
            loss.backward()
            self.optimizer.step()

            self.train_count += 1
            self.data_idx = end_idx
            self.train_loss = loss.detach()

            # Between batches: preserve Q/R GRU hidden but restart KF recursion.
            if self.data_idx < self.data_num:
                self.dnn.reset_kf_only(clean_history=True)

            # Save recovery checkpoints at the configured interval.
            if self.train_count % self.save_num == 0:
                torch.save(
                    self.dnn.kf_net,
                    self._checkpoint_path(str(self.train_count))
                )

            # Run validation using a separate filter with the current network weights.
            if (
                    self.val_every is not None
                    and self.val_data_path is not None
                    and self.train_count % self.val_every == 0
            ):
                val_filter = self._build_validation_filter()
                tester = Tester_QR(
                    filter=val_filter,
                    data_path=self.val_data_path,
                    data_file=self.val_data_file,
                    model_path=self._checkpoint_path(str(self.train_count)),
                    result_tag='validation',
                    is_validation=True,
                    is_mismatch=False,
                    device=self.device,
                    load_model=False,
                    collect_monitor=False
                )
                self.validate(tester)

                if self.stop_training:
                    return

            if do_monitor:
                loss_q_mean = 0.5 * (
                    loss_yy3.mean() + loss_yy4.mean()
                )
                loss_r_mean = 0.5 * (
                    loss_yy1.mean() + loss_yy2.mean()
                )

                q_stats = self._monitor_stats(
                    Q_diag_yy, floor=self.monitor_floor
                )
                r_stats = self._monitor_stats(
                    R_diag_yy, floor=self.monitor_floor
                )

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
                    'Q_diag_stats': q_stats,
                    'R_diag_stats': r_stats,
                }
                torch.save(
                    monitor_pack,
                    os.path.join(
                        self.monitor_dir,
                        f'monitor_train_{self.train_count}.pt'
                    )
                )

            # End the epoch when the remaining samples are fewer than
            # one full batch.
            if self.data_idx + self.batch_size >= self.data_num:
                self.data_idx = 0
                break

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
            best_path = self._checkpoint_path('best')
            torch.save(tester.filter.kf_net, best_path)
            print(
                f'Save best model at {self.save_path} '
                f'& train {self.train_count} '
                f'& rmse_sum = {rmse_sum:.10f}'
            )
        else:
            self.val_wait += 1
            print(
                f'No improvement: rmse_sum = {rmse_sum:.10f}, '
                f'wait {self.val_wait}/{self.val_patience}'
            )

        self.valid_loss = valid_loss
        self.valid_x_hat = tester.x_hat.detach().cpu()

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
            self.data_file if val_data_file is None else val_data_file
        )
        self.val_every = val_every
        self.val_patience = patience
        self.val_min_delta = min_delta
        self.val_wait = 0
        self.best_rmse_sum = float('inf')
        self.stop_training = False

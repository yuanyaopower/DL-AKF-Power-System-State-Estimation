import os
import torch
import numpy as np
import config
import time

from scipy.io import savemat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print_num = 75

T_his = np.load('T_his.npy')
batch_size = config.train_config['batch_size']
fixed_u = np.load('fixed_u.npy')
predicted_indices = np.setdiff1d(np.arange(30), fixed_u)
n = 30


class Tester_QR():
    def __init__(
            self,
            filter,
            data_path,
            model_path,
            data_file='z.pt',
            result_tag='DL-AKF',
            is_validation=False,
            is_mismatch=False,
            device=DEVICE,
            load_model=True,
            collect_monitor=True
    ):

        self.result_path = 'R_KN'

        self.filter = filter
        self.data_path = data_path
        self.data_file = data_file
        self.result_tag = result_tag
        self.model_path = model_path
        self.is_validation = is_validation
        self.is_mismatch = is_mismatch
        self.device = device
        self.collect_monitor = collect_monitor

        # During normal testing, load the saved full QRNet object.
        # During training-time validation, Trainer_R can directly copy the
        # current state_dict into a separate validation filter and skip disk I/O.
        if load_model:
            self.filter.kf_net = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            ).to(self.device)
        else:
            self.filter.kf_net = self.filter.kf_net.to(self.device)

        self.filter.kf_net.eval()

        # Start validation/test from a clean filter state.
        self.filter.reset(clean_history=True)

        self.x_dim = self.filter.x_dim
        self.z_dim = self.filter.z_dim

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        self.data_x = torch.load(
            os.path.join(data_path, 'x.pt'),
            map_location=self.device
        ).to(self.device)

        self.data_z = torch.load(
            os.path.join(data_path, data_file),
            map_location=self.device
        ).to(self.device)

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

        self.T_his = T_his
        self.batch_size = batch_size
        self.n = n

        self.T_bac = 1

        self.predicted_indices = predicted_indices
        self.theta_idx = list(range(self.n + 1, 2 * self.n))
        self.feature_idx = list(self.predicted_indices) + self.theta_idx

        x_hat = torch.zeros(
            self.data_num,
            self.x_dim,
            self.seq_len - self.T_his,
            device=self.device
        )

        with torch.no_grad():

            T_future = self.seq_len - self.T_his

            # Q/R/NIS monitoring is useful for final testing, but it is not
            # required by training-time validation, scheduler, or early stopping.
            if self.collect_monitor:
                Q_diag_all = torch.zeros(
                    self.data_num,
                    T_future,
                    self.x_dim,
                    device=self.device
                )

                R_diag_all = torch.zeros(
                    self.data_num,
                    T_future,
                    self.z_dim,
                    device=self.device
                )

                Q_input_all = torch.zeros(
                    self.data_num,
                    T_future,
                    self.x_dim,
                    device=self.device
                )

                R_input_all = torch.zeros(
                    self.data_num,
                    T_future,
                    self.z_dim,
                    device=self.device
                )

                NIS_all = torch.zeros(
                    self.data_num,
                    1,
                    self.seq_len - self.T_his,
                    device=self.device
                )

            loss_yy = torch.zeros(
                self.data_num,
                self.seq_len - self.T_his,
                device=self.device
            )

            # =====================================================
            # 2026.8.29
            # Test-stage single prediction-filtering-step timing
            # =====================================================
            measure_time = not self.is_validation
            step_times = []
            step_count = 0
            warmup_steps = 10
            # =====================================================

            for i in range(self.data_num):

                if i % print_num == 0:
                    if self.is_validation:
                        print(
                            f'Validating {i} / {self.data_num} '
                            f'of {self.model_path}'
                        )
                    else:
                        print(
                            f'Testing {i} / {self.data_num} '
                            f'of {self.model_path}'
                        )

                x_sequence = self.data_x[
                    i,
                    self.feature_idx,
                    :self.T_his
                ]

                x_sequence_all = self.data_x[
                    i,
                    self.feature_idx,
                    :
                ]

                self.z_past_sequence = self.data_z[
                    i,
                    :,
                    :self.T_his
                ]

                self.filter.x_sequence = x_sequence

                x_predict = self.filter.initial_predict(
                    x_sequence
                )

                residual_q = (
                    self.data_x[
                        i,
                        self.feature_idx,
                        self.T_his
                    ].reshape((-1, 1))
                    - x_predict
                )

                residual_q = residual_q.expand(
                    -1,
                    self.T_bac
                )

                residual_z = (
                    self.data_z[
                        i,
                        :,
                        self.T_his
                    ].reshape((-1, 1))
                    - self.filter.Powermodel.measurement_function(
                        x_predict
                    )
                )

                residual_z = residual_z.expand(
                    -1,
                    self.T_bac
                )

                for ii in range(
                        self.T_his,
                        self.seq_len
                ):

                    r_scale = 1.0

                    # =============================================
                    # Timing start
                    # =============================================
                    if measure_time:
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()

                        t0 = time.perf_counter()
                    # =============================================

                    self.filter.filtering(
                        self.data_z[
                            i,
                            :,
                            ii
                        ].reshape((-1, 1)),
                        self.z_past_sequence,
                        x_sequence,
                        x_sequence_all[:, ii],
                        residual_q,
                        residual_z,
                        r_scale=r_scale
                    )

                    # =============================================
                    # Timing end
                    # =============================================
                    if measure_time:
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()

                        dt = time.perf_counter() - t0

                        if step_count >= warmup_steps:
                            step_times.append(dt)

                        step_count += 1
                    # =============================================

                    t_idx = ii - self.T_his

                    if self.collect_monitor:
                        Q_diag_all[
                            i,
                            t_idx
                        ] = self.filter.monitor_q_diag

                        R_diag_all[
                            i,
                            t_idx
                        ] = self.filter.monitor_r_diag

                        Q_input_all[
                            i,
                            t_idx
                        ] = self.filter.monitor_q_input

                        R_input_all[
                            i,
                            t_idx
                        ] = self.filter.monitor_r_input

                    loss_yy[
                        i,
                        t_idx
                    ] = self.filter.loss

                x_hat[i] = self.filter.state_history[
                    :,
                    -T_future:
                ]

                if self.collect_monitor:
                    NIS_all[i] = self.filter.NIS

                # Preserve continuous states while truncating
                # the computation graph.
                self.filter.detach_continuous_state()

            # =====================================================
            # Print average single-step execution time
            # =====================================================
            if measure_time and len(step_times) > 0:
                avg_step_time_ms = (
                    np.mean(step_times) * 1000.0
                )

                print(
                    f'Average DL-AKF prediction-filtering '
                    f'time per step = '
                    f'{avg_step_time_ms:.4f} ms '
                    f'({len(step_times)} timed steps, '
                    f'{warmup_steps} warm-up steps excluded)'
                )
            # =====================================================

            # Final state estimates are saved only during testing.
            if not self.is_validation:
                torch.save(
                    x_hat,
                    os.path.join(
                        self.data_path,
                        f'x_hat_{self.result_tag}.pt'
                    )
                )

            loss = loss_yy.mean()

            print(
                f'loss = {loss:.10f}'
            )

            if not self.is_validation:

                x_hat_np = (
                    x_hat
                    .cpu()
                    .detach()
                    .numpy()
                )

                x_hat_np = (
                    x_hat_np
                    .transpose(0, 2, 1)
                    .reshape(
                        -1,
                        x_hat_np.shape[1]
                    )
                    .T
                )

                savemat(
                    os.path.join(
                        self.data_path,
                        f'x_hat_{self.result_tag}.mat'
                    ),
                    {
                        'x_hat': x_hat_np
                    }
                )

            x_true = self.data_x[
                :,
                self.feature_idx,
                self.T_his:
            ]

            x_pred = x_hat

            U_true = x_true[
                :,
                :24,
                :
            ]

            U_pred = x_pred[
                :,
                :24,
                :
            ]

            th_true = x_true[
                :,
                24:,
                :
            ]

            th_pred = x_pred[
                :,
                24:,
                :
            ]

            rmse_u = torch.sqrt(
                torch.mean(
                    (U_true - U_pred) ** 2
                )
            )

            rmse_th = torch.sqrt(
                torch.mean(
                    (th_true - th_pred) ** 2
                )
            )

            print(
                f'rmse_u = {rmse_u.item():.10f}, '
                f'rmse_th = {rmse_th.item():.10f}, '
                f'rmse_sum = '
                f'{(rmse_u + rmse_th).item():.10f}'
            )

            if self.collect_monitor:

                tag = (
                    'val'
                    if self.is_validation
                    else 'test'
                )

                model_tag = os.path.splitext(
                    os.path.basename(
                        self.model_path
                    )
                )[0]

                mon_dir = os.path.join(
                    self.data_path,
                    'monitor_' + tag
                )

                os.makedirs(
                    mon_dir,
                    exist_ok=True
                )

                pack = {
                    'model_path':
                        self.model_path,

                    'data_path':
                        self.data_path,

                    'Q_diag':
                        Q_diag_all.detach().cpu(),

                    'R_diag':
                        R_diag_all.detach().cpu(),

                    'Q_input':
                        Q_input_all.detach().cpu(),

                    'R_input':
                        R_input_all.detach().cpu(),

                    'NIS':
                        NIS_all.detach().cpu(),
                }

                torch.save(
                    pack,
                    os.path.join(
                        mon_dir,
                        f'{model_tag}_monitor.pt'
                    )
                )

                savemat(
                    os.path.join(
                        mon_dir,
                        f'{model_tag}_monitor.mat'
                    ),
                    {
                        'Q_diag':
                            Q_diag_all
                            .detach()
                            .cpu()
                            .numpy(),

                        'R_diag':
                            R_diag_all
                            .detach()
                            .cpu()
                            .numpy(),

                        'Q_input':
                            Q_input_all
                            .detach()
                            .cpu()
                            .numpy(),

                        'R_input':
                            R_input_all
                            .detach()
                            .cpu()
                            .numpy(),

                        'NIS':
                            NIS_all
                            .squeeze(1)
                            .detach()
                            .cpu()
                            .numpy(),
                    }
                )

        self.loss = loss
        self.x_hat = x_hat

        self.filter.reset(
            clean_history=True
        )

        self.rmse_u = rmse_u.item()
        self.rmse_th = rmse_th.item()
# IEEE-118 validation/test runner.

import os
import time

import torch
from scipy.io import savemat

DEVICE = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print_num = 5


class Tester_QR:
    def __init__(
            self,
            filter,
            data_path,
            model_path,
            data_file='z.pt',
            result_tag='DL-AKF_case1',
            is_validation=False,
            is_mismatch=False,
            device=DEVICE,
            load_model=True,
            collect_monitor=True
    ):

        self.filter = filter
        self.data_path = data_path
        self.data_file = data_file
        self.model_path = model_path
        self.result_tag = result_tag

        self.is_validation = is_validation
        self.is_mismatch = is_mismatch

        self.device = device

        self.collect_monitor = (
            collect_monitor
            and (not is_validation)
        )

        if load_model:

            self.filter.kf_net = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            ).to(self.device)

        else:

            self.filter.kf_net = (
                self.filter.kf_net
                .to(self.device)
            )

        self.filter.kf_net.eval()

        self.filter.reset(
            clean_history=True
        )

        self.x_dim = self.filter.x_dim
        self.z_dim = self.filter.z_dim

        self.n = self.filter.n
        self.T_his = self.filter.T_his

        self.T_bac = 1

        self.predicted_indices = (
            self.filter.predicted_indices
        )

        self.U_dim = len(
            self.predicted_indices
        )

        self.feature_idx = (
            self.filter
            .Powermodel
            .feature_idx
            .detach()
            .cpu()
            .tolist()
        )

        self.data_x = torch.load(
            os.path.join(
                data_path,
                'x.pt'
            ),
            map_location=self.device
        ).to(self.device)

        self.data_z = torch.load(
            os.path.join(
                data_path,
                data_file
            ),
            map_location=self.device
        ).to(self.device)

        self.data_num = (
            self.data_x.shape[0]
        )

        self.seq_len = (
            self.data_x.shape[2]
        )

        T_future = (
            self.seq_len
            - self.T_his
        )

        x_hat = torch.zeros(
            self.data_num,
            self.x_dim,
            T_future,
            device=self.device
        )

        loss_yy = torch.zeros(
            self.data_num,
            T_future,
            device=self.device
        )

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

        # =========================================================
        # Test-stage single prediction-filtering-step timing
        # =========================================================
        measure_time = not self.is_validation
        step_times = []
        step_count = 0
        warmup_steps = 10
        # =========================================================

        with torch.no_grad():

            for i in range(
                    self.data_num
            ):

                if i % print_num == 0:

                    stage = (
                        'Validating'
                        if self.is_validation
                        else 'Testing'
                    )

                    print(
                        f'{stage} '
                        f'{i} / {self.data_num} '
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

                x_all = self.data_x[
                    i,
                    :,
                    :
                ]

                z_past_sequence = self.data_z[
                    i,
                    :,
                    :self.T_his
                ]

                x_predict = (
                    self.filter
                    .initial_predict(
                        x_sequence
                    )
                )

                residual_q = (
                    self.data_x[
                        i,
                        self.feature_idx,
                        self.T_his
                    ].reshape((-1, 1))
                    - x_predict
                ).expand(
                    -1,
                    self.T_bac
                )

                residual_z = (
                    self.data_z[
                        i,
                        :,
                        self.T_his
                    ].reshape((-1, 1))
                    -
                    self.filter
                    .Powermodel
                    .measurement_function(
                        x_predict,
                        x_all[
                            :,
                            self.T_his
                        ]
                    )
                ).expand(
                    -1,
                    self.T_bac
                )

                for ii in range(
                        self.T_his,
                        self.seq_len
                ):

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

                        z_past_sequence,

                        x_sequence,

                        x_sequence_all[
                            :,
                            ii
                        ],

                        x_all[
                            :,
                            ii
                        ],

                        residual_q,

                        residual_z,

                        r_scale=1.0,

                        record_monitor=(
                            self.collect_monitor
                        )
                    )

                    # =============================================
                    # Timing end
                    # =============================================
                    if measure_time:

                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()

                        dt = (
                            time.perf_counter()
                            - t0
                        )

                        if (
                            step_count
                            >= warmup_steps
                        ):
                            step_times.append(
                                dt
                            )

                        step_count += 1

                    # =============================================

                    t_idx = (
                        ii
                        - self.T_his
                    )

                    loss_yy[
                        i,
                        t_idx
                    ] = self.filter.loss

                    if self.collect_monitor:

                        Q_diag_all[
                            i,
                            t_idx
                        ] = (
                            self.filter
                            .monitor_q_diag
                        )

                        R_diag_all[
                            i,
                            t_idx
                        ] = (
                            self.filter
                            .monitor_r_diag
                        )

                        Q_input_all[
                            i,
                            t_idx
                        ] = (
                            self.filter
                            .monitor_q_input
                        )

                        R_input_all[
                            i,
                            t_idx
                        ] = (
                            self.filter
                            .monitor_r_input
                        )

                x_hat[i] = (
                    self.filter
                    .state_history[
                        :,
                        -T_future:
                    ]
                )

                # Preserve state continuity across samples
                # while truncating graph.
                self.filter.detach_continuous_state()

            loss = loss_yy.mean()

            x_true = self.data_x[
                :,
                self.feature_idx,
                self.T_his:
            ]

            U_true = x_true[
                :,
                :self.U_dim,
                :
            ]

            U_pred = x_hat[
                :,
                :self.U_dim,
                :
            ]

            th_true = x_true[
                :,
                self.U_dim:,
                :
            ]

            th_pred = x_hat[
                :,
                self.U_dim:,
                :
            ]

            # Per-time-step RMSE over state variables, followed by
            # averaging over all samples and prediction time steps.
            rmse_u_all = torch.sqrt(
                torch.mean(
                    (
                        U_true
                        - U_pred
                    ) ** 2,
                    dim=1
                )
            )

            rmse_th_all = torch.sqrt(
                torch.mean(
                    (
                        th_true
                        - th_pred
                    ) ** 2,
                    dim=1
                )
            )

            rmse_u = rmse_u_all.mean()
            rmse_th = rmse_th_all.mean()

        # =========================================================
        # Print average single-step execution time
        # =========================================================
        if measure_time and len(step_times) > 0:

            avg_step_time_ms = (
                sum(step_times)
                / len(step_times)
                * 1000.0
            )

            print(
                f'Average DL-AKF prediction-filtering '
                f'time per step = '
                f'{avg_step_time_ms:.4f} ms '
                f'({len(step_times)} timed steps, '
                f'{warmup_steps} warm-up steps excluded)'
            )
        # =========================================================

        print(
            f'loss = {loss:.10f}'
        )

        print(
            f'rmse_u = '
            f'{rmse_u.item():.10f}, '
            f'rmse_th = '
            f'{rmse_th.item():.10f}, '
            f'rmse_sum = '
            f'{(rmse_u + rmse_th).item():.10f}'
        )

        if not self.is_validation:

            torch.save(
                x_hat,
                os.path.join(
                    self.data_path,
                    f'x_hat_{self.result_tag}.pt'
                )
            )

            x_hat_np = (
                x_hat
                .detach()
                .cpu()
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

        if self.collect_monitor:

            mon_dir = os.path.join(
                self.data_path,
                'monitor_test'
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
                    Q_diag_all
                    .detach()
                    .cpu(),

                'R_diag':
                    R_diag_all
                    .detach()
                    .cpu(),

                'Q_input':
                    Q_input_all
                    .detach()
                    .cpu(),

                'R_input':
                    R_input_all
                    .detach()
                    .cpu(),
            }

            torch.save(
                pack,
                os.path.join(
                    mon_dir,
                    f'{self.result_tag}_monitor.pt'
                )
            )

            savemat(
                os.path.join(
                    mon_dir,
                    f'{self.result_tag}_monitor.mat'
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
                }
            )

        self.loss = loss
        self.x_hat = x_hat

        self.rmse_u = (
            rmse_u.item()
        )

        self.rmse_th = (
            rmse_th.item()
        )

        self.filter.reset(
            clean_history=True
        )
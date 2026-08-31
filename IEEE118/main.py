# IEEE-118 DL-AKF experiment entry point.

from Filter import QR_Filter, Holt_QR_Filter
from trainer import Trainer_R
from tester import Tester_QR
from powermodel import Powermodel

import os
import torch
import numpy as np
import scipy.io as sio

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================
# User settings
# ============================================================
Train = False
train_iter = 30
eps = 1e-5

# 1: Gaussian; 2: Gaussian mixture; 3: time-varying Gaussian.
case_id = 1

# 'dl_akf', 'fixed', 'holt'
method_id = 'dl_akf'

# ============================================================
# Case/method configuration
# ============================================================
case_config = {
    1: {
        'data_file': 'z.pt',
        'r_scale_bound': 1e2,
    },
    2: {
        'data_file': 'z_h.pt',
        'r_scale_bound': 1e4,
    },
    3: {
        'data_file': 'z_mix.pt',
        'r_scale_bound': 1e2,
    },
}

method_config = {
    'dl_akf': {
        'filter_class': QR_Filter,
        'covariance_mode': 'adaptive',
        'checkpoint_prefix': 'DL-AKF',
        'result_prefix': 'DL-AKF',
        'trainable': True,
    },
    'fixed': {
        # Evaluation-only ablation. It reuses the corresponding full DL-AKF
        # checkpoint and changes only the covariance path to fixed Q/R.
        'filter_class': QR_Filter,
        'covariance_mode': 'fixed',
        'checkpoint_prefix': 'DL-AKF',
        'result_prefix': 'DL-AKF_Fixed',
        'trainable': False,
    },
    'holt': {
        'filter_class': Holt_QR_Filter,
        'covariance_mode': 'adaptive',
        'checkpoint_prefix': 'DL-AKF_Holt',
        'result_prefix': 'DL-AKF_Holt',
        'trainable': True,
    },
}

if case_id not in case_config:
    raise ValueError(f'Unsupported case_id: {case_id}')
if method_id not in method_config:
    raise ValueError(f'Unsupported method_id: {method_id}')

case_cfg = case_config[case_id]
method_cfg = method_config[method_id]

data_file = case_cfg['data_file']
checkpoint_stem = (
    f'{method_cfg["checkpoint_prefix"]}_case{case_id}'
)
checkpoint_name = checkpoint_stem + '.pt'

best_model_path = os.path.join(
    './model_saved_118', checkpoint_stem + '_best.pt'
)
result_tag = f'{method_cfg["result_prefix"]}_case{case_id}'

filter_kwargs = {
    'covariance_mode': method_cfg['covariance_mode'],
    'r_scale_bound': case_cfg['r_scale_bound'],
    'jacobian_steps': 16,
}

train_data_path = './data_118/train/'
val_data_path = './data_118/val/'
test_data_path = './data_118/test/'
os.makedirs('./model_saved_118', exist_ok=True)

print(
    f'Experiment: Case {case_id}, '
    f'Method: {method_cfg["result_prefix"]}, '
    f'Data: {data_file}, '
    f'Jacobian steps: {filter_kwargs["jacobian_steps"]}'
)

if Train and not method_cfg['trainable']:
    print(
        'DL-AKF(Fixed) is evaluation-only. Training is skipped and '
        'the corresponding full DL-AKF checkpoint is used.'
    )

# ============================================================
# System and dimensions
# ============================================================
systemdata = sio.loadmat('./IEEE118_bus.mat')
bus = systemdata['bus']
branch = systemdata['branch']
Y = systemdata['Ybus']
Bsh = systemdata['Bsh']

n = bus.shape[0]
G = np.real(Y)
B = np.imag(Y)

selected_test_file = os.path.join(test_data_path, data_file)
if not os.path.isfile(selected_test_file):
    raise FileNotFoundError(
        f'Selected measurement file not found: {selected_test_file}'
    )

# Read the selected measurement tensor to obtain z_dim.
z_shape_source = torch.load(
    selected_test_file, map_location='cpu'
)
z_dim = z_shape_source.shape[1]
del z_shape_source

fixed_u = np.load('fixed_u_large.npy')
predicted_indices = np.setdiff1d(np.arange(n), fixed_u)
T_his = int(np.load('T_his.npy'))

U_dim = predicted_indices.shape[0]
theta_dim = n - 1
x_dim = U_dim + theta_dim

powermodel = Powermodel(
    n,
    z_dim,
    G,
    B,
    Bsh,
    branch,
    T_his,
    predicted_indices,
    device=DEVICE
).to(DEVICE)

filter_class = method_cfg['filter_class']

# ============================================================
# Stage-2 training
# ============================================================
if Train and method_cfg['trainable']:
    trainer = Trainer_R(
        dnn=filter_class(
            powermodel,
            n,
            x_dim,
            z_dim,
            predicted_indices,
            **filter_kwargs,
            device=DEVICE
        ),
        data_path=train_data_path,
        data_file=data_file,
        save_path=checkpoint_name,
        mode=1,
        device=DEVICE
    )

    trainer.configure_val_early_stop(
        val_data_path=val_data_path,
        val_data_file=data_file,
        val_every=15,
        patience=3,
        min_delta=eps
    )

    for _ in range(train_iter):
        trainer.train_one_epoch()

        if trainer.stop_training:
            print(
                'Early stopping triggered at '
                f'train_count = {trainer.train_count}'
            )
            break

        trainer.dnn.reset(clean_history=True)

# ============================================================
# Final test
# ============================================================
if not os.path.isfile(best_model_path):
    raise FileNotFoundError(
        f'Best checkpoint not found: {best_model_path}'
    )

tester = Tester_QR(
    filter=filter_class(
        powermodel,
        n,
        x_dim,
        z_dim,
        predicted_indices,
        **filter_kwargs,
        device=DEVICE
    ),
    data_path=test_data_path,
    data_file=data_file,
    model_path=best_model_path,
    result_tag=result_tag,
    is_validation=False,
    is_mismatch=False,
    device=DEVICE
)

print(
    f'Case {case_id}, {method_cfg["result_prefix"]}: '
    f'loss = {tester.loss.item():.10f}, '
    f'rmse_u = {tester.rmse_u:.10f}, '
    f'rmse_th = {tester.rmse_th:.10f}'
)

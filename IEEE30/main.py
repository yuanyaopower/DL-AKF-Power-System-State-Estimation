from Filter_yuan import QR_Filter, Holt_QR_Filter
from trainer_adj import Trainer_R
from tester_adj import Tester_QR
from powermodel_yuan import Powermodel

import os
import torch
import numpy as np
import scipy.io as sio

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

train_iter = 30
Train = True

eps = 1e-5

# Case 1: Gaussian; 2: Gaussian mixture; 3: time-varying Gaussian; 4: Case 3 + bad data.
case_id = 1

# Methods: 'dl_akf', 'fixed', or 'holt'.
method_id = 'dl_akf'

case_files = {
    1: 'z.pt',
    2: 'z_h.pt',
    3: 'z_mix.pt',
    4: 'z_k.pt',
}

method_config = {
    'dl_akf': {
        'filter_class': QR_Filter,
        'filter_kwargs': {
            'covariance_mode': 'adaptive',
            'r_scale_bound': 1e4,
        },
        'checkpoint_prefix': 'DL-AKF',
        'result_prefix': 'DL-AKF',
        'trainable': True,
    },
    'fixed': {
        # Evaluation-only ablation using the same trained DL-AKF checkpoint.
        'filter_class': QR_Filter,
        'filter_kwargs': {
            'covariance_mode': 'fixed',
            'r_scale_bound': 1e4,
        },

        'checkpoint_prefix': 'DL-AKF',
        'result_prefix': 'DL-AKF_Fixed',
        'trainable': False,
    },
    'holt': {
        'filter_class': Holt_QR_Filter,
        'filter_kwargs': {
            'covariance_mode': 'adaptive',
            'r_scale_bound': 1e4,
        },
        'checkpoint_prefix': 'DL-AKF_Holt',
        'result_prefix': 'DL-AKF_Holt',
        'trainable': True,
    },
}

if case_id not in case_files:
    raise ValueError(f'Unsupported case_id: {case_id}')

if method_id not in method_config:
    raise ValueError(f'Unsupported method_id: {method_id}')

z_file = case_files[case_id]
method_cfg = method_config[method_id]
filter_class = method_cfg['filter_class']
filter_kwargs = method_cfg['filter_kwargs']
checkpoint_prefix = method_cfg['checkpoint_prefix']
result_prefix = method_cfg['result_prefix']
trainable = method_cfg['trainable']

train_data_path = './data_yuan/train/'
val_data_path = './data_yuan/val/'
test_data_path = './data_yuan/test/'
model_dir = './model_saved_yuan/'

os.makedirs(model_dir, exist_ok=True)

checkpoint_name = f'{checkpoint_prefix}_case{case_id}.pt'
best_model_path = os.path.join(
    model_dir,
    f'{checkpoint_prefix}_case{case_id}_best.pt'
)
result_tag = f'{result_prefix}_case{case_id}'

print(
    f'Experiment: Case {case_id}, '
    f'Method: {result_prefix}, '
    f'Data: {z_file}'
)

if Train and not trainable:
    print(
        'DL-AKF(Fixed) is an evaluation-only ablation. '
        'Training is skipped and the corresponding DL-AKF checkpoint is used.'
    )

systemdata = sio.loadmat(
    './IEEE30_bus.mat'
)

bus = systemdata['bus']
branch = systemdata['branch']
Y = systemdata['Ybus']
Bsh = systemdata['Bsh']

n = bus.shape[0]
G = np.real(Y)
B = np.imag(Y)

z = torch.load(
    os.path.join(test_data_path, z_file),
    map_location=DEVICE
)
z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
z_dim = z.shape[1]

fixed_u = np.load('fixed_u.npy')
predicted_indices = np.setdiff1d(np.arange(30), fixed_u)
T_his = np.load('T_his.npy')

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
    predicted_indices
).to(DEVICE)

if Train and trainable:
    trainer_dl_akf = Trainer_R(
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
        data_file=z_file,
        save_path=checkpoint_name,
        mode=1,
        device=DEVICE
    )

    trainer_dl_akf.configure_val_early_stop(
        val_data_path=val_data_path,
        val_data_file=z_file,
        val_every=15,
        patience=3,
        min_delta=eps
    )

    for _ in range(train_iter):
        trainer_dl_akf.train_one_epoch()

        if getattr(trainer_dl_akf, 'stop_training', False):
            print(
                'Early stopping triggered at '
                f'train_count = {trainer_dl_akf.train_count}'
            )
            break

        trainer_dl_akf.dnn.reset(clean_history=True)

if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f'Best checkpoint not found: {best_model_path}'
    )

tester_dl_akf = Tester_QR(
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
    data_file=z_file,
    model_path=best_model_path,
    result_tag=result_tag,
    is_validation=False,
    is_mismatch=False,
    device=DEVICE
)

print(
    f'Case {case_id}, {result_prefix}: '
    f'loss = {tester_dl_akf.loss.item():.10f}, '
    f'rmse_u = {tester_dl_akf.rmse_u:.10f}, '
    f'rmse_th = {tester_dl_akf.rmse_th:.10f}'
)

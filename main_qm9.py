# Rdkit import should be first, do not move it
# Rdkit 导入必须放在最前面，不要移动它。这是为了避免与某些系统库的冲突。
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch    
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save

# 初始化参数解析器
parser = argparse.ArgumentParser(description='E3Diffusion')

# --- 实验设置参数 ---
parser.add_argument('--exp_name', type=str, default='debug_10', help='实验名称，用于保存日志和模型')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='模型类型: our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics | gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='概率模型类型，目前主要是 diffusion')

# --- 扩散模型参数 ---
# 训练复杂度是 O(1) (不受步数影响)，但采样复杂度是 O(steps)。
parser.add_argument('--diffusion_steps', type=int, default=500, help='扩散过程的步数')
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='噪声调度策略: learned, cosine, polynomial_2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5, help='扩散噪声精度')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='损失函数类型: vlb (变分下界), l2 (均方误差)')

# --- 训练超参数 ---
parser.add_argument('--n_epochs', type=int, default=200, help='训练的总轮数')
parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
parser.add_argument('--brute_force', type=eval, default=False,
                    help='是否使用暴力计算 (True | False)')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='是否使用激活归一化 (ActNorm)')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='是否在每个 epoch 中途打断训练 (用于调试)')
parser.add_argument('--dp', type=eval, default=True,
                    help='是否使用数据并行 (DataParallel)')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='模型是否以时间 t 为条件')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='是否进行梯度裁剪')
parser.add_argument('--trace', type=str, default='hutch',
                    help='迹估计方法: hutch | exact')

# --- EGNN (等变图神经网络) 参数 ---
parser.add_argument('--n_layers', type=int, default=6,
                    help='EGNN 的层数')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='不变子层的数量')
parser.add_argument('--nf', type=int, default=128,
                    help='隐藏层特征维度 (number of features)')
parser.add_argument('--tanh', type=eval, default=True,
                    help='在坐标 MLP 中是否使用 tanh 激活函数')
parser.add_argument('--attention', type=eval, default=True,
                    help='在 EGNN 中是否使用注意力机制')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='归一化常数: diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='是否使用正弦位置编码')
# <--- EGNN 参数结束

parser.add_argument('--ode_regularization', type=float, default=1e-3, help='ODE 正则化系数')

# --- 数据集参数 ---
parser.add_argument('--dataset', type=str, default='qm9',
                    help='数据集名称: qm9 | qm9_second_half (仅在训练集的后5万个样本上训练)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='QM9 数据集目录')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='如果设置为整数，QM9 将仅包含具有该原子数量的分子')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='去量化方法: uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1, help='报告步数间隔')
parser.add_argument('--wandb_usr', type=str, help='Wandb 用户名')
parser.add_argument('--no_wandb', action='store_true', help='禁用 wandb 日志')
parser.add_argument('--online', type=bool, default=True, help='True = wandb 在线模式 -- False = wandb 离线模式')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='禁用 CUDA 训练')
parser.add_argument('--save_model', type=eval, default=True,
                    help='是否保存模型')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='生成采样的 epoch 间隔 (似乎未在代码中广泛使用)')
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 的工作线程数')
parser.add_argument('--test_epochs', type=int, default=10, help='每隔多少个 epoch 进行一次测试和评估')
parser.add_argument('--data_augmentation', type=eval, default=False, help='是否使用数据增强 (旋转等)')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='条件生成的属性参数: homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='恢复训练的路径 (checkpoint 文件夹)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='开始训练的 epoch (用于恢复训练)')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='EMA (指数移动平均) 衰减率，0 表示关闭。通常 0.999 是个合理值。')
parser.add_argument('--augment_noise', type=float, default=0, help='数据增强时添加的噪声量')
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='计算稳定性指标时生成的样本数量')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='归一化因子 [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true', help='是否从分子中移除氢原子')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='是否包含原子电荷信息')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="用于在每个 epoch 中多次可视化 (设置很大则不频繁可视化)")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="归一化 EGNN 的聚合总和")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='聚合方法: "sum" 或 "mean"')

# 解析参数
args = parser.parse_args()

# 获取数据集信息 (原子编码器/解码器等)
dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# 获取 wandb 用户名
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

# 设置 CUDA 设备
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

# --- 恢复训练逻辑 ---
if args.resume is not None:
    # 如果指定了 resume，加载之前的参数配置
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    # 加载旧的参数文件
    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # 恢复关键参数
    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # 确保新添加的参数存在 (兼容旧版本 checkpoint)
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
kwargs['mode'] = 'disabled'
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    # 计算属性的均值和绝对偏差，用于归一化
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    # 准备上下文向量
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    # --- 加载模型权重 (如果恢复训练) ---
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()

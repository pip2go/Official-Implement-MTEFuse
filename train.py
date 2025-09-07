import argparse
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pytorch_msssim import ms_ssim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import trange
import pandas as pd
from LoadDataset import MSRSDataset, SARDataset
from loss import *
from model import MTEFuse_model
from utils import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # 修改环境变量进一步限制碎片化

import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def set_seed(seed=42):
    """固定随机种子，保证训练过程中各种随机操作具有确定性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_mean(x):
    if isinstance(x, list):
        A = []
        for i in range(len(x)):
            x1 = torch.mean(x[i], dim=1).unsqueeze(1)
            A.append(x1)
    else:
        A = torch.mean(x, dim=1).unsqueeze(1)
    return A


def normalize_tensor_minmax(x, eps=1e-8):
    x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + eps)


def infrared_structure_loss(fused_img, ir_img, threshold=0.5):
    """
    保留红外图像显著区域的强度信息
    :param fused_img: 融合图像，形状 [B, 1, H, W]
    :param ir_img: 红外图像，形状 [B, 1, H, W]
    :param threshold: 红外显著区域亮度阈值，建议归一化图像中 T∈[0,1]
    :return: 红外结构保持损失 (L1 范数)
    """
    ir_img_norm = normalize_tensor_minmax(ir_img)
    fused_img_norm = normalize_tensor_minmax(fused_img)
    with torch.no_grad():
        # 构造 mask（显著区域 = 1，其他区域 = 0）
        mask = (ir_img_norm > threshold).float()

    # 强调显著区域差异（元素乘法），使用 L1 损失
    loss = 1e3 * torch.abs((fused_img_norm - ir_img_norm) * mask).mean()
    return loss


def main(opt):
    set_seed(50)

    # 设备配置
    device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    dataset = MSRSDataset(
        vi_dir=os.path.join(opt.train_dataset, 'vi'),
        ir_dir=os.path.join(opt.train_dataset, 'ir'),
        train_num=opt.train_num        # -1: 表示训练整个数据集
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True,
    )

    # 模型初始化
    MTEFuse = MTEFuse_model(
        in_c=1,
        out_c=64,
    ).to(device)

    # 设置优化器和学习率
    optimizer = Adam(MTEFuse.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # 进入训练模式
    MTEFuse.train()

    # 模型再导入
    if opt.resume:
        MTEFuse.load_state_dict(torch.load(opt.resume))

    # 开始训练
    loss_total_all = []
    loss_in_all = []
    loss_grad_all = []
    loss_ir_focus_all = []
    loss_ssim_all = []
    loss_fuse_all = []

    epochs_bar = trange(opt.epochs, ncols=120)
    for epoch in epochs_bar:
        for batch_id, (vi, ir) in enumerate(dataloader):

            vi = vi.to(device).float()
            ir = ir.to(device).float()

            # 模型的梯度清零
            MTEFuse.zero_grad()
            optimizer.zero_grad()

            # 训练模型-Encoder
            pre_img_vi = MTEFuse.encoder(vi)
            pre_img_ir = MTEFuse.encoder(ir)

            # 训练模型-Infuse
            pre_img21 = MTEFuse.fuses1(torch.cat((pre_img_vi[0], pre_img_ir[0]), dim=1))
            pre_img22 = MTEFuse.fuses2(torch.cat((pre_img_vi[1], pre_img_ir[1]), dim=1))
            pre_img23 = MTEFuse.fuses3(torch.cat((pre_img_vi[2], pre_img_ir[2]), dim=1))
            pre_img24 = MTEFuse.fuses4(torch.cat((pre_img_vi[3], pre_img_ir[3]), dim=1))

            # 训练模型-decoder
            pre_img3 = MTEFuse.decoder([pre_img21, pre_img22, pre_img23, pre_img24])

            laplacian = Laplacian()
            image_y = vi[:, :1, :, :]
            ir_y = ir[:, :1, :, :]  # 注意，取红外的单通道

            # 1. 原有基本loss
            fusion_max = torch.max(image_y, ir_y)  # max(vi, ir)
            loss_in = F.l1_loss(fusion_max, pre_img3)

            # 2. 基础梯度loss（保持细节清晰）
            y_grad = laplacian(image_y)
            ir_grad = laplacian(ir_y)
            generate_img_grad = laplacian(pre_img3)
            grad_joint = torch.max(y_grad, ir_grad)
            loss_grad = (
                    F.l1_loss(grad_joint, generate_img_grad) +
                    F.l1_loss(generate_img_grad, grad_joint) +  # 逆向梯度损失，减缓图像模糊
                    F.l1_loss(y_grad, generate_img_grad)
            )

            # 3. 红外显著区域loss（让IR特征更突出）
            mask = torch.sigmoid((torch.max(image_y, ir_y) - 0.5) * 10)  # 只对亮目标部分施加约束，太过于机械，容易大致图像大规模模糊，降低图像的SSIM
            loss_ir_focus = (
                    2*F.l1_loss(pre_img3 * mask, ir_y * mask) +   # 强化亮度靠近，太大容易导致大面积发黑
                    2*F.mse_loss(pre_img3 * mask, ir_y * mask) -   # 细腻逼近
                    3*torch.mean(pre_img3 * mask)  # 直接奖励高亮（负向loss）
            )

            # 4. SSIM loss（保持结构感）
            loss_ssim = (
                    (1 - ssim(pre_img3, fusion_max, data_range=1.0)) +
                    (1 - ms_ssim(pre_img3, fusion_max, data_range=1.0, size_average=True))
            )

            # 5. Layer loss（保持结构感）
            loss_fuse_layers = (
                # Layer 1
                gradient_loss_max(pre_img21, pre_img_vi[0], pre_img_ir[0]) +

                # Layer 2
                gradient_loss_max(pre_img22, pre_img_vi[1], pre_img_ir[1]) +
                F.l1_loss(pre_img22, pre_img_vi[1]) +
                2*F.mse_loss(pre_img22, pre_img_ir[1]) +

                # Layer 3
                F.l1_loss(pre_img23, pre_img_vi[2]) +
                2*F.mse_loss(pre_img23, pre_img_ir[2]) +

                # Layer 4
                F.l1_loss(pre_img24, pre_img_vi[3]) +
                2*F.mse_loss(pre_img24, pre_img_ir[3])
            )

            # 6. 最终总loss（加权）
            totalloss = (
                    0.8 * loss_in +
                    1.6 * loss_grad +
                    1.2 * loss_ir_focus +
                    0.8 * loss_ssim +
                    1.2 * loss_fuse_layers
            )

            loss_total_all.append(totalloss.item())
            loss_in_all.append(loss_in.item())
            loss_grad_all.append(loss_grad.item())
            loss_ir_focus_all.append(loss_ir_focus.item())
            loss_ssim_all.append(loss_ssim.item())
            loss_fuse_all.append(loss_fuse_layers.item())

            # 反向传播
            totalloss.backward()
            nn.utils.clip_grad_norm_(MTEFuse.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            torch.cuda.empty_cache()

            # 计算已处理图像数量
            current_images = (batch_id + 1) * opt.batch_size  # 当前图像数
            total_images = len(dataloader.dataset)  # 总图像数

            # 添加显示信息
            mesg = "{}\tEpoch {}/{}: \t[{}/{}]\t total loss: {:.6f}\t".format(
                time.ctime(), epoch, opt.epochs, current_images // opt.batch_size, total_images // opt.batch_size,
                totalloss)
            epochs_bar.set_description(mesg)

        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

    # 保存模型
    os.makedirs(opt.save_model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = os.path.join(opt.save_model_dir, f'checkpoint_MTEFuse_epoch-{timestamp}.pt')
    save_checkpoint(MTEFuse, save_path)

    # 绘制Loss曲线
    plt.figure(figsize=(12, 6))
    plt.plot(loss_total_all, label='Total Loss')
    plt.plot(loss_in_all, label='Loss_in')
    plt.plot(loss_grad_all, label='Loss_grad')
    plt.plot(loss_ir_focus_all, label='Loss_ir_focus')
    plt.plot(loss_ssim_all, label='Loss_ssim')
    plt.plot(loss_fuse_all, label='Loss_fuse_layers')

    # 创建一个字典，键是列名，值是对应的损失列表
    losses_dict = {
        'Total_Loss': loss_total_all,
        'Loss_in': loss_in_all,
        'Loss_grad': loss_grad_all,
        'Loss_ir_focus': loss_ir_focus_all,
        'Loss_ssim': loss_ssim_all,
        'Loss_fuse_layers': loss_fuse_all
    }

    # 创建DataFrame
    df = pd.DataFrame(losses_dict)

    # 保存为CSV文件
    df.to_csv('losses.csv', index=False)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve per Batch')
    plt.legend()
    plt.grid(True)
    # 强制x轴只显示整数刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    current_time = time.strftime("%H:%M:%S", time.localtime())
    plt.savefig("outputs1/" + "loss_curve_per_batch" + str(current_time) + ".png", dpi=300)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='tranning epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--train_num', type=int, default=300, help='train numbers;-1表示训练整个数据集')
    parser.add_argument('--test_num', type=int, default=None, help='test numbers;-1表示测试整个数据集')
    parser.add_argument('--train_dataset', type=str, default=r'/home/self/MTEFuse/MSRS-main/MSRS-main/train',
                        help='path of training dataset')
    parser.add_argument('--test_dataset', type=str, default=r'/home/self/MTEFuse/TNO',
                        help='path of dataset')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--save_model_dir', type=str, default='models_mri', help='folder name to save model')
    parser.add_argument('--save_log_dir', type=str, default='logs', help='folder name to save log')
    parser.add_argument('--save_loss_dir', type=str, default='models/loss',
                        help='folder name to save loss result of model')
    parser.add_argument('--device', type=int, default=0, help='set it to 1 for running on GPU, 0 for CPU')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training model')
    parser.add_argument('--resume', type=str, default=None, help='resume the saved model for training')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='path for saving result images and model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
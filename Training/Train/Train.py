"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""

# python imports
import os, shutil, time, csv, sys, json
import warnings
from argparse import ArgumentParser
import glob
import io
from tqdm import tqdm
import argparse
import inspect
import importlib
import yaml
# 忽略部分警告
warnings.filterwarnings("ignore")
'''将当前目录加入搜索路径'''
# external imports
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn  as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as nnf
# from torch.utils.tensorboard import SummaryWriter


# 加载目录
current_work_dir = os.path.abspath(os.path.dirname(__file__))
superior_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_work_dir)
sys.path.append(superior_dir)

from Losses import *
from MyDataset import *

def import_module_from_file(module_name, net_name):
    module_name = 'Models.' + module_name
    module = importlib.import_module(module_name)  # 动态导入模块
    net = getattr(module, net_name)  # 获取函数对象
    return net

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mov_image, flow, mode='bilinear'):
        device = flow.device
        flow = flow.permute(0, 2, 3, 4, 1)
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda(device=device).float()
        grid_d = grid_d.cuda(device=device).float()
        grid_w = grid_w.cuda(device=device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]

        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode=mode, align_corners=True)

        return warped
    
def dice(pred1, truth1):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32,
                34, 36]
    dice_35 = np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        # print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0

        dice_35[index] = intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_35)

def train(config_path):    
    # 加载配置文件
    with open(os.path.join(config_path), 'r') as f:
        Configs = yaml.load(f, Loader=yaml.FullLoader)

    model_name = Configs['model_name']
    device = 'cuda:' + Configs['gpu']
    data_path = Configs['data_path']
    # 导入网络
    Net = import_module_from_file(model_name, 'Net')

    '''记录当前时间'''
    currTime = time.localtime()
    stratTimeStr = str(currTime.tm_year) + '年' + str(currTime.tm_mon) + '月' + str(currTime.tm_mday) + '日' + str(
        currTime.tm_hour) + '时' + str(currTime.tm_min) + "分"
    print("开始时间为：", stratTimeStr)

    '''构建日志文件夹'''
    # 检测日志文件夹是否存在，不存在则创建
    log_path = os.path.join(current_work_dir, 'LogFiles')
    if not os.path.exists(log_path):
         os.makedirs(log_path)
    # 确定文件夹名称: 模型名称+模型超参数+训练超参数+数据集
    log_name_dir = Configs['log_file_name']
    # 获取超参数
    model_hyperparameters = Configs['model_hyperparameters']
    training_hyperparameters = Configs['training_hyperparameters']
    # 构建日志文件夹命名字符串
    log_folder_name = model_name + '_' + Configs['dataset_name'] + '_B-' + str(training_hyperparameters['batch_size']) + \
        '_Sim-' + training_hyperparameters['similarity_loss'] + '_LR-'+ str(training_hyperparameters['learning_rate']) + '_Lam-' + str(training_hyperparameters['loss_weights'][-1])
    if log_name_dir is not None:
        for key, value in log_name_dir.items():
            log_folder_name += '_' + key + '-' + str(value)
    # 检查该文件夹是否存在
    current_model_log_path = os.path.join(log_path, log_folder_name)
    if not os.path.exists(current_model_log_path):
         os.makedirs(current_model_log_path)
    # 检查模型权重文件目录是否存在
    current_model_save_path = os.path.join(current_model_log_path, 'ModelWeights')
    if not os.path.exists(current_model_save_path):
         os.makedirs(current_model_save_path)
    # 将模型文件和超参数文件保存到日志文件夹中
    model_path = os.path.join(superior_dir, 'Models', model_name+'.py')
    shutil.copy(model_path, os.path.join(current_model_log_path, model_name + '.py'))
    shutil.copy(config_path, os.path.join(current_model_log_path, model_name + '.yaml'))
    # 生成日志csv文件
    training_log_csv = os.path.join(current_model_log_path, 'training_log.csv')
    val_log_csv = os.path.join(current_model_log_path, 'val_log.csv')
    f = open(training_log_csv, 'w')
    with f:
        fnames = ['Epoch', 'Train_Loss', 'Train_Loss_Similarity', 'Train_Loss_Smooth', 'Best_Dice', 'Best_Epoch']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    f = open(val_log_csv, 'w')
    with f:
        fnames = ['Epoch', 'Val_Loss', 'Val_Loss_Similarity', 'Val_Loss_Smooth', 'Val_Dice', 'Best_Dice', 'Best_Epoch']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    '''模型初始化'''
    pretrain_model_path = os.path.join(current_model_save_path, training_hyperparameters["pretrain_model_weights_name"])
    # 获取参数列表
    init_signature = inspect.signature(Net.__init__)
    parameters_dir = init_signature.parameters
    net_parameters = {}
    for p in parameters_dir:
        if p != 'self':
            net_parameters[p] = model_hyperparameters[p]
    
    model = Net(**net_parameters).cuda(device=device)
    # 判断是否从断点加载模型重新训练
    if training_hyperparameters['is_breakpoint']:
        if os.path.exists(pretrain_model_path):
            model.load_state_dict(torch.load(pretrain_model_path))
            print("模型加载成功")
            start_epoch = training_hyperparameters['start_epoch']
        else:
            print("模型文件不存在")
            raise
    else:
        start_epoch = 0
    # 加载STN网络
    transform = SpatialTransformer().cuda(device=device)
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True    
    # 设置优化函数和损失函数 Set optimizer and losses
    learning_rate = training_hyperparameters['learning_rate']
    similarity_loss = training_hyperparameters['similarity_loss']
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if similarity_loss == "SAD":
        loss_similarity = SAD().loss
    elif similarity_loss == "MSE":
        loss_similarity = MSE().loss
    elif similarity_loss == "NCC":
        loss_similarity = NCC()
    else:
        print("similarity_loss参数必须为{SAD, MSE, NCC}其中一个")
    loss_smooth = smoothloss
    # 加载训练和验证数据集
    batch_size = training_hyperparameters['batch_size']
    train_transform = None
    train_dataset = Train_Dataset(data_path=data_path, transform=train_transform)
    train_datanum = train_dataset.__len__()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                pin_memory=True)
    val_dataset = Val_Dataset(data_path=data_path)
    val_datanum = val_dataset.__len__()
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True,
                            drop_last=True)

    '''训练'''
    best_dice = 0 # 记录最佳dice分数
    best_epoch = 0 # 记录最佳dice分数对应的epoch
    loss_weights = training_hyperparameters['loss_weights'] # 不同损失的权重参数，是一个列表
    num_accumulation = training_hyperparameters['num_accumulation']
    train_log_loss_all, train_log_loss_sim, train_log_loss_smooth =[],[],[]
    val_log_loss_all, val_log_loss_sim, val_log_loss_smooth, val_dice =[],[],[],[]
    for epoch in range(training_hyperparameters['num_epoch']):
        model.train()
        # 如果从断点开始训练，需要重新计算当前epoch
        current_epoch = start_epoch + epoch + 1
        log_loss_all, log_loss_sim, log_loss_smooth = 0., 0., 0.
        num_batch = 0
        with tqdm(total=train_datanum) as pbar:
            pbar.set_description('TrainingProcessing:')
            for _, data in enumerate(train_loader):
                num_batch += 1
                # 计算形变场和形变图像
                input_moving = data[0].cuda(device=device).float()
                input_fixed = data[1].cuda(device=device).float()
                dvf = model(input_moving, input_fixed)
                warpped_moving = transform(input_moving, dvf)
                # 计算损失
                loss1 = loss_similarity(input_fixed, warpped_moving)  * loss_weights[0]
                loss2 = loss_smooth(dvf)  * loss_weights[1]  
                loss = loss1  + loss2 
                # 梯度反传
                loss.backward()
                if num_accumulation > 1 :
                    if num_batch%num_accumulation==0 :
                      optimizer.step() # 更新网络参数
                      optimizer.zero_grad() # 清空过往梯度
                else:
                    optimizer.step() # 更新网络参数
                    optimizer.zero_grad() # 清空过往梯度
                
                # 记录损失值
                log_loss_all += loss.item()
                log_loss_sim += loss1.item()
                log_loss_smooth += loss2.item()
                train_log_loss_all.append(loss.item())
                train_log_loss_sim.append(loss1.item())
                train_log_loss_smooth .append(loss2.item())
                # 更新进度条
                time.sleep(0.1)
                pbar.update(batch_size)
        # 打印训练信息
        print('\n迭代索引为： {} 训练集总损失为： {:.4f}; 相似测度损失为：{:.4f}; 形变场平滑损失为：{:.4f};'
            .format(current_epoch, log_loss_all / num_batch, log_loss_sim / num_batch, log_loss_smooth / num_batch))
        training_save_log_value = [current_epoch, log_loss_all / num_batch, log_loss_sim / num_batch, log_loss_smooth / num_batch]

        '''验证'''
        val_save_log_value = []
        if (current_epoch % Configs['training_hyperparameters']['save_epoch'] == 0 or current_epoch==1):
            model.eval()
            log_loss_all, log_loss_sim, log_loss_smooth = 0., 0., 0.
            with torch.no_grad():
                dsc_mean = 0
                for _, data in enumerate(val_loader):
                    # 计算形变场和形变图像
                    input_moving = data[0].cuda(device=device).float()
                    input_fixed = data[1].cuda(device=device).float()
                    moving_lab = data[2].cuda(device=device).float()
                    fixed_lab = data[3].cuda(device=device).float()
                    dvf = model(input_moving, input_fixed)
                    warpped_moving = transform(input_moving, dvf)
                    # 计算损失
                    loss1 = loss_similarity(input_fixed, warpped_moving)  
                    loss2 = loss_smooth(dvf)  
                    loss = loss1 * loss_weights[0] + loss2 * loss_weights[1] 
                    # 记录损失值
                    log_loss_all += loss.item()
                    log_loss_sim += loss1.item()
                    log_loss_smooth += loss2.item()                    
                    # 计算dice
                    warp_moving_lab = transform(moving_lab.float(), dvf, mode='nearest')
                    dsc_mean += dice(warp_moving_lab.data.cpu().numpy().squeeze().copy(), fixed_lab.data.cpu().numpy().squeeze().copy())
            val_log_loss_all.append(log_loss_all/val_datanum)
            val_log_loss_sim.append(log_loss_sim/val_datanum)
            val_log_loss_smooth .append(log_loss_smooth/val_datanum)
            print('迭代索引为： {} 验证集总损失为： {:.4f}; 相似测度损失为：{:.4f}; 形变场平滑损失为：{:.4f}; Dice为：{:.4f};'
            .format(current_epoch,
                log_loss_all / val_datanum,
                log_loss_sim / val_datanum,
                log_loss_smooth / val_datanum,
                dsc_mean / val_datanum))
            val_dice.append(dsc_mean/val_datanum)
            val_save_log_value.extend([current_epoch, log_loss_all / val_datanum, log_loss_sim / val_datanum, log_loss_smooth / val_datanum])
            f = open(val_log_csv, 'a')
            with f:
                writer = csv.writer(f)
                val_save_log_value.extend([dsc_mean/val_datanum, best_dice, best_epoch])
                writer.writerow(val_save_log_value)
        '''记录结果'''
        
        # 更新最优Dice
        if len(val_dice)>0 and  val_dice[-1] > best_dice:
            best_dice = val_dice[-1]
            best_epoch = current_epoch
            # 保存模型参数
            torch.save(model.state_dict(), os.path.join(current_model_save_path, model_name + '_E_' + str(current_epoch) + '.pkl'))
        print(model_name + '最优验证集Dice指标为: {:.4f} 来自迭代：{}'.format(best_dice, best_epoch))
        # 保存训练和验证的损失
        f = open(training_log_csv, 'a')
        with f:
            writer = csv.writer(f)
            training_save_log_value.extend([best_dice, best_epoch])
            writer.writerow(training_save_log_value)
            
        """记录当前时间"""
        currTime = time.localtime()
        currTimeStr = str(currTime.tm_year) + '年' + str(currTime.tm_mon) + '月' + str(
            currTime.tm_mday) + '日' + str(
            currTime.tm_hour) + '时' + str(currTime.tm_min) + "分"
        print("开始时间为：", stratTimeStr)
        print("当前时间为：", currTimeStr)
        print('*' * 100)
        print('\n')

        
if __name__ == "__main__":
    '''获取超参数'''
    parser = ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        help="the path of config file")
    train(**vars(parser.parse_args()))
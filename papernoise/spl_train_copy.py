import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from tqdm import tqdm

# 导入数据集和模型
from noisedata_copy import SPLdata  # 假设数据集类在noisedata.py中
from soundmodel_copy import SPLModel  # 假设多分枝模型在model.py中

# 设置训练参数
class TrainingConfig:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 100
        self.nc=800
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.val_split = 0.2
        self.random_seed = 42
        self.data_dir = "../csvdata333"  # 数据集路径
        self.log_dir = f"runs/spl/exp_nc{self.nc}_epochs{self.epochs}_batchsize{self.batch_size}_lr{self.learning_rate}_wd{self.weight_decay}" + time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"{self.log_dir}/checkpoint"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 混合损失函数：MSE + 余弦相似度
# class HybridLoss(nn.Module):
#     def __init__(self, mse_weight=1.0, cosine_weight=0.5):
#         super(HybridLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.cosine_loss = nn.CosineEmbeddingLoss()
#         self.mse_weight = mse_weight
#         self.cosine_weight = cosine_weight
        
#     def forward(self, pred, target):
#         # MSE损失
#         mse = self.mse_loss(pred, target)
        
#         # 余弦相似度损失 (需要创建标签为1的目标向量，表示相似)
#         target_cosine = torch.ones(pred.size(0), device=pred.device)
#         cosine = self.cosine_loss(pred, target, target_cosine)
        
#         # 加权求和
#         total_loss = self.mse_weight * mse + self.cosine_weight * cosine
#         return total_loss, mse, cosine

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # 数据移至设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 压缩输出维度，从(batch, 1)变为(batch)
        outputs = outputs.squeeze()
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累计损失
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        
        # 更新进度条
        pbar.set_postfix({"batch_loss": round(loss.item(), 4)})  # 使用round函数格式化到4位小数
        
        # 记录训练批次数据到TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("train/batch_loss", loss.item(), global_step)

    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader.dataset)
    
    # 记录到TensorBoard
    writer.add_scalar("train/epoch_loss", avg_loss, epoch)
    
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    # 收集所有批次的目标和输出，用于绘制PR曲线
    all_targets = []
    all_outputs = []

    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # 数据移至设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            # batch_size = torch.arange(inputs.size(0), device=device)
            outputs = outputs.squeeze()
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 累计损失
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            # 收集所有批次的数据
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader.dataset)

    
    # 记录到TensorBoard
    writer.add_scalar("val/epoch_loss", avg_loss, epoch)

    
    # 记录示例预测
    if epoch % 5 == 0:  # 每5个epoch记录一次
        # writer.add_pr_curve("val/predictions", targets.cpu().numpy(), outputs.cpu().numpy(), epoch)
        # writer.add_histogram("val/prediction_dist", outputs.cpu().numpy(), epoch)
        # 合并所有批次的数据
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        # 确保PR曲线输入形状匹配
        writer.add_pr_curve("val/predictions", all_targets, all_outputs, epoch)
        writer.add_histogram("val/prediction_dist", all_outputs, epoch)
    
    
    return avg_loss

def main(config):
    # 设置随机种子
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(config.log_dir)
    print(f"TensorBoard日志保存至: {config.log_dir}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = SPLdata(
        directory_path=config.data_dir,
        val_split=config.val_split,
        random_seed=config.random_seed
    )
    val_dataset = train_dataset.get_validation_dataset()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    
    # 初始化模型
    model = SPLModel(nc=config.nc).to(config.device)
    
    # 初始化损失函数和优化器
    '''在PyTorch中，nn.MSELoss()默认计算的是批次中所有样本的平均损失（即批次损失），而不是单个样本的损失。这是PyTorch中大多数损失函数的默认行为。
    具体来说：

    如果你传入一个批次的预测值和目标值（比如形状为[batch_size, *]的张量）
    nn.MSELoss()会先计算批次中每个样本的均方误差
    然后默认对这些误差求平均（reduction='mean'）
    你可以通过设置reduction参数来改变这个行为：

    'mean'（默认）：返回批次的平均损失
    'sum'：返回批次的总损失
    'none'：返回每个样本的单独损失'''
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # # 记录模型图到TensorBoard
    # dummy_input = torch.randn(1, 3, device=config.device)
    # dummy_type = torch.tensor([0], device=config.device)
    # dummy_mode = torch.tensor([0], device=config.device)
    # writer.add_graph(model, (dummy_input, dummy_type, dummy_mode))
    
    # 训练循环
    best_val_loss = float('inf')
    
    print("开始训练...")
    for epoch in range(config.epochs):
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch, writer
        )
        
        # 在验证集上评估
        val_loss = validate(
            model, val_loader, criterion, config.device, epoch, writer
        )
        
        # 计算平均损失
        train_loss_avg = train_loss
        val_loss_avg = val_loss
        
        # 打印 epoch 统计信息
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss_avg:.8f}")
        print(f"Val Loss:   {val_loss_avg:.8f}")
        
        # 记录到TensorBoard
        writer.add_scalars("loss/comparison", {
            "train": train_loss_avg,
            "val": val_loss_avg
        }, epoch)
        
        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(config.checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_avg,
            }, checkpoint_path)
            print(f"最佳模型保存至: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"最终模型保存至: {final_path}")
    
    # 关闭TensorBoard
    writer.close()
    print("训练完成!")

if __name__ == "__main__":
    config = TrainingConfig()
    print(f"使用设备: {config.device}")
    main(config)
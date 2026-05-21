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
from noisedata_copy import Octave_1_3_data  # 假设数据集类在noisedata.py中
from soundmodel_copy import Octave_1_3_Model  # 假设多分枝模型在model.py中

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
        self.model_name = "Octave_1_3"
        self.data_dir = "../csvdata333"  # 数据集路径
        self.log_dir = f"runs/Octave_1_3/exp_nc{self.nc}_epochs{self.epochs}_batchsize{self.batch_size}_lr{self.learning_rate}_wd{self.weight_decay}" + time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"{self.log_dir}/checkpoint"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## 前10000Hz的线谱模型保存在checkpoint
## 前20000Hz的线谱模型保存在checkpoints
# 混合损失函数：MSE + 余弦相似度 + 总声压级损失
class HybridLoss(nn.Module):
    def __init__(self, mse_weight=1.0, cosine_weight=50, lp_weight=1):
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.lp_weight = lp_weight  # 总声压级损失权重
        
    def forward(self, pred, target):
        # 1. MSE损失（频谱形状相似性）
        mse = self.mse_loss(pred, target)
        
        # 2. 余弦相似度损失（频谱整体趋势）
        target_cosine = torch.ones(pred.size(0), device=pred.device)
        cosine = self.cosine_loss(pred, target, target_cosine)
        
        # 3. 总声压级损失（能量叠加特性）
        # 将dB转换为线性声压（能量）
        pred_lin = torch.pow(10.0, pred / 10.0)
        target_lin = torch.pow(10.0, target / 10.0)
        
        # 计算总声压级（添加极小值避免log(0)）
        pred_total_lp = 10.0 * torch.log10(torch.sum(pred_lin, dim=1) + 1e-12)
        target_total_lp = 10.0 * torch.log10(torch.sum(target_lin, dim=1) + 1e-12)
        lp_loss = self.mse_loss(pred_total_lp, target_total_lp)
        
        # 加权求和
        total_loss = (self.mse_weight * mse + 
                      self.cosine_weight * cosine + 
                      self.lp_weight * lp_loss)
        return total_loss, mse, cosine, lp_loss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_lp_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # print(targets)
        # 数据移至设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播：多分枝输出
        outputs = model(inputs)
        batch_size = torch.arange(inputs.size(0), device=device)
        # outputs = outputs[batch_size, :].squeeze()
        
        # 计算损失
        loss, mse, cosine, lp_loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累计损失
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_cosine += cosine.item() * batch_size
        total_lp_loss += lp_loss.item() * batch_size
        
        # 更新进度条
        pbar.set_postfix({"batch_loss": round(loss.item(), 4)})  # 使用round函数格式化到4位小数
        
        # 记录训练批次数据到TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("train/batch_loss", loss.item(), global_step)
        writer.add_scalar("train/batch_mse", mse.item(), global_step)
        writer.add_scalar("train/batch_cosine", cosine.item(), global_step)
        writer.add_scalar("train/batch_lp_loss", lp_loss.item(), global_step)
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_cosine = total_cosine / len(dataloader.dataset)
    avg_lp_loss = total_lp_loss / len(dataloader.dataset)
    
    # 记录到TensorBoard
    writer.add_scalar("train/epoch_loss", avg_loss, epoch)
    writer.add_scalar("train/epoch_mse", avg_mse, epoch)
    writer.add_scalar("train/epoch_cosine", avg_cosine, epoch)
    writer.add_scalar("train/epoch_lp_loss", avg_lp_loss, epoch)
    
    return avg_loss, avg_mse, avg_cosine, avg_lp_loss

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_lp_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # 数据移至设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 前向传播
            outputs = model(inputs)
            batch_size = torch.arange(inputs.size(0), device=device)
            # outputs = outputs[batch_size, :].squeeze()
            
            # 计算损失
            loss, mse, cosine, lp_loss = criterion(outputs, targets)
            
            # 累计损失
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            total_cosine += cosine.item() * batch_size
            total_lp_loss += lp_loss.item() * batch_size
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_cosine = total_cosine / len(dataloader.dataset)
    avg_lp_loss = total_lp_loss / len(dataloader.dataset)
    
    # 记录到TensorBoard
    writer.add_scalar("val/epoch_loss", avg_loss, epoch)
    writer.add_scalar("val/epoch_mse", avg_mse, epoch)
    writer.add_scalar("val/epoch_cosine", avg_cosine, epoch)
    writer.add_scalar("val/epoch_lp_loss", avg_lp_loss, epoch)
    
    # 记录示例预测
    if epoch % 5 == 0:  # 每5个epoch记录一次
        writer.add_pr_curve("val/predictions", targets.cpu().numpy(), outputs.cpu().numpy(), epoch)
        writer.add_histogram("val/prediction_dist", outputs.cpu().numpy(), epoch)
    
    return avg_loss, avg_mse, avg_cosine, avg_lp_loss

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
    train_dataset = Octave_1_3_data(
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
    model = Octave_1_3_Model(nc=config.nc).to(config.device)
    
    # 初始化损失函数和优化器
    criterion = HybridLoss(mse_weight=1.0, cosine_weight=50, lp_weight=1)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # # 记录模型图到TensorBoard
    # dummy_input = torch.randn(1, 7, device=config.device)
    # # dummy_bowl = torch.tensor([0], device=config.device)
    # # dummy_mode = torch.tensor([0], device=config.device)
    # writer.add_graph(model, (dummy_input))
    
    # 训练循环
    best_val_loss = float('inf')
    
    print("开始训练...")
    for epoch in range(config.epochs):
        # 训练一个epoch
        train_loss, train_mse, train_cosine, train_lp_loss = train_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch, writer
        )
        
        # 在验证集上评估
        val_loss, val_mse, val_cosine, val_lp_loss = validate(
            model, val_loader, criterion, config.device, epoch, writer
        )
        
        # 计算平均损失
        train_loss_avg = train_loss
        val_loss_avg = val_loss
        
        # 打印 epoch 统计信息
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss_avg:.4f} (MSE: {train_mse:.4f}, Cosine: {train_cosine:.4f}, LP Loss: {train_lp_loss:.4f})")
        print(f"Val Loss:   {val_loss_avg:.4f} (MSE: {val_mse:.4f}, Cosine: {val_cosine:.4f}, LP Loss: {val_lp_loss:.4f})")
        
        # 记录到TensorBoard
        writer.add_scalars("loss/comparison", {
            "train": train_loss_avg,
            "val": val_loss_avg
        }, epoch)
        
        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(config.checkpoint_dir, f"octave_1_3_best_model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_avg,
            }, checkpoint_path)
            print(f"最佳模型保存至: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, "_final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"最终模型保存至: {final_path}")
    
    # 关闭TensorBoard
    writer.close()
    print("训练完成!")

if __name__ == "__main__":
    config = TrainingConfig()
    print(f"使用设备: {config.device}")
    main(config)
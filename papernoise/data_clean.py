import pandas as pd
import numpy as np
import glob
import os
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats

class MultiCSVModelWithThreshold:
    def __init__(self, feature_columns, target_column, threshold=0.1):
        """
        初始化多CSV文件处理模型
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.threshold = threshold
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def load_and_combine_csv(self, folder_path, file_pattern="*.csv"):
        """
        加载并合并多个CSV文件
        """
        file_paths = glob.glob(os.path.join(folder_path, file_pattern))
        print(f"找到 {len(file_paths)} 个CSV文件")
        
        all_data = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
                print(f"成功加载: {os.path.basename(file_path)}, 形状: {df.shape}")
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
        
        if not all_data:
            raise ValueError("未成功加载任何CSV文件")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"合并后数据总形状: {combined_df.shape}")
        return combined_df
    
    def preprocess_data(self, df):
        """
        数据预处理和清洗
        """
        required_columns = self.feature_columns + [self.target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        processed_df = df.copy()
        
        # 处理缺失值
        print("处理前数据形状:", processed_df.shape)
        processed_df = processed_df.dropna(subset=required_columns)
        print("处理缺失值后数据形状:", processed_df.shape)
        
        return processed_df
    
    def train_model(self, df):
        """
        训练线性回归模型
        """
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"模型训练完成")
        print(f"测试集MAE: {mae:.4f}")
        print(f"测试集MSE: {mse:.4f}")
        
        return X_scaled, y
    
    def apply_threshold_filter(self, df, X_scaled, y):
        """
        应用阈值筛选
        """
        y_pred = self.model.predict(X_scaled)
        
        absolute_deviation = np.abs(y - y_pred)
        relative_deviation = absolute_deviation / np.abs(y)
        
        result_df = df.copy()
        result_df['predicted_value'] = y_pred
        result_df['absolute_deviation'] = absolute_deviation
        result_df['relative_deviation'] = relative_deviation
        result_df['exceeds_threshold'] = relative_deviation > self.threshold
        
        exceeded_threshold_df = result_df[result_df['exceeds_threshold']].copy()
        within_threshold_df = result_df[~result_df['exceeds_threshold']].copy()
        
        print(f"总数据点: {len(result_df)}")
        print(f"超出阈值的数据点: {len(exceeded_threshold_df)} ({len(exceeded_threshold_df)/len(result_df)*100:.2f}%)")
        print(f"在阈值内的数据点: {len(within_threshold_df)} ({len(within_threshold_df)/len(result_df)*100:.2f}%)")
        
        return exceeded_threshold_df, within_threshold_df, result_df

    def analyze_feature_distributions(self, exceeded_threshold_df, within_threshold_df, output_folder):
        """
        分析并打印阈值内外输入指标的分布情况
        """
        print("\n" + "="*60)
        print("阈值内外输入指标分布分析")
        print("="*60)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 创建分布分析文件夹
        dist_folder = os.path.join(output_folder, 'feature_distributions')
        if not os.path.exists(dist_folder):
            os.makedirs(dist_folder)
        
        distribution_results = {}
        
        for feature in self.feature_columns:
            print(f"\n--- 特征 '{feature}' 分布分析 ---")
            
            # 提取两组数据
            within_values = within_threshold_df[feature].dropna()
            exceeded_values = exceeded_threshold_df[feature].dropna()
            
            # 基本统计信息
            within_stats = within_values.describe()
            exceeded_stats = exceeded_values.describe()
            
            print(f"阈值内数据统计:")
            print(f"  数量: {len(within_values):,}")
            print(f"  均值: {within_stats['mean']:.4f}")
            print(f"  标准差: {within_stats['std']:.4f}")
            print(f"  最小值: {within_stats['min']:.4f}")
            print(f"  25%分位数: {within_stats['25%']:.4f}")
            print(f"  中位数: {within_stats['50%']:.4f}")
            print(f"  75%分位数: {within_stats['75%']:.4f}")
            print(f"  最大值: {within_stats['max']:.4f}")
            
            print(f"\n阈值外数据统计:")
            print(f"  数量: {len(exceeded_values):,}")
            print(f"  均值: {exceeded_stats['mean']:.4f}")
            print(f"  标准差: {exceeded_stats['std']:.4f}")
            print(f"  最小值: {exceeded_stats['min']:.4f}")
            print(f"  25%分位数: {exceeded_stats['25%']:.4f}")
            print(f"  中位数: {exceeded_stats['50%']:.4f}")
            print(f"  75%分位数: {exceeded_stats['75%']:.4f}")
            print(f"  最大值: {exceeded_stats['max']:.4f}")
            
            # 统计检验（如果样本量足够）
            if len(within_values) > 30 and len(exceeded_values) > 30:
                try:
                    t_stat, p_value = stats.ttest_ind(within_values, exceeded_values, equal_var=False)
                    print(f"\n统计检验 (t检验):")
                    print(f"  t统计量: {t_stat:.4f}")
                    print(f"  p值: {p_value:.4f}")
                    if p_value < 0.05:
                        print("  ★ 两组数据分布有显著差异 (p < 0.05)")
                    else:
                        print("  ○ 两组数据分布无显著差异")
                except:
                    print("  统计检验无法计算")
            
            # 保存分布对比图
            self.plot_feature_distribution_comparison(
                within_values, exceeded_values, feature, dist_folder
            )
            
            distribution_results[feature] = {
                'within_stats': within_stats,
                'exceeded_stats': exceeded_stats,
                'within_count': len(within_values),
                'exceeded_count': len(exceeded_values)
            }
        
        # 保存统计结果到CSV
        self.save_distribution_stats(distribution_results, dist_folder)
        
        return distribution_results
    
    def plot_feature_distribution_comparison(self, within_values, exceeded_values, feature_name, output_folder):
        """
        绘制特征分布对比图
        """
        plt.rcParams['font.family'] = 'SimHei'        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"特征 '{feature_name}' 在阈值内外的分布对比", fontsize=16, fontweight='bold')
        
        # 1. 直方图对比
        axes[0, 0].hist(within_values, bins=30, alpha=0.7, label='阈值内', color='blue', edgecolor='black')
        axes[0, 0].hist(exceeded_values, bins=30, alpha=0.7, label='阈值外', color='red', edgecolor='black')
        axes[0, 0].set_xlabel(feature_name)
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('直方图对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 箱线图对比
        data_to_plot = [within_values, exceeded_values]
        box_plot = axes[0, 1].boxplot(data_to_plot, labels=['阈值内', '阈值外'], patch_artist=True)
        # 设置箱线图颜色
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        axes[0, 1].set_ylabel(feature_name)
        axes[0, 1].set_title('箱线图对比')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 密度图对比
        within_values.plot.density(ax=axes[1, 0], label='阈值内', color='blue', linewidth=2)
        exceeded_values.plot.density(ax=axes[1, 0], label='阈值外', color='red', linewidth=2)
        axes[1, 0].set_xlabel(feature_name)
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('概率密度函数对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q图（正态性检验）
        stats.probplot(within_values, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('阈值内数据Q-Q图（正态性检验）')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 清理特征名称，确保文件名安全
        safe_feature_name = self.sanitize_filename(feature_name)
        plt.savefig(os.path.join(output_folder, f'distribution_comparison_{safe_feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  分布对比图已保存: distribution_comparison_{safe_feature_name}.png")
    
    def save_distribution_stats(self, distribution_results, output_folder):
        """
        保存分布统计结果到CSV文件
        """
        stats_data = []
        
        for feature, results in distribution_results.items():
            within_stats = results['within_stats']
            exceeded_stats = results['exceeded_stats']
            
            stats_data.append({
                'feature': feature,
                'group': '阈值内',
                'count': results['within_count'],
                'mean': within_stats['mean'],
                'std': within_stats['std'],
                'min': within_stats['min'],
                '25%': within_stats['25%'],
                'median': within_stats['50%'],
                '75%': within_stats['75%'],
                'max': within_stats['max']
            })
            
            stats_data.append({
                'feature': feature,
                'group': '阈值外',
                'count': results['exceeded_count'],
                'mean': exceeded_stats['mean'],
                'std': exceeded_stats['std'],
                'min': exceeded_stats['min'],
                '25%': exceeded_stats['25%'],
                'median': exceeded_stats['50%'],
                '75%': exceeded_stats['75%'],
                'max': exceeded_stats['max']
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(output_folder, 'feature_distribution_statistics.csv'), index=False)
        print(f"\n分布统计结果已保存: feature_distribution_statistics.csv")
    
    def save_results(self, exceeded_threshold_df, result_df, output_folder, within_threshold_df):
        """
        保存结果到CSV文件，并将超出阈值的数据按源文件分开保存
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 1. 保存所有超出阈值的数据到一个汇总文件
        exceeded_threshold_df.to_csv(
            os.path.join(output_folder, 'all_exceeded_threshold_data.csv'), 
            index=False
        )

        # 2. 保存所有未超出阈值的数据到一个汇总文件
        within_threshold_df.to_csv(
            os.path.join(output_folder, 'all_within_threshold_data.csv'), 
            index=False
        )
        
        # 3. 按来源文件分开保存超出阈值的数据
        exceeded_groups = exceeded_threshold_df.groupby('source_file')
        
        for source_file, group_data in exceeded_groups:
            safe_filename = f"exceeded_data_from_{os.path.splitext(source_file)[0]}.csv"
            file_path = os.path.join(output_folder, safe_filename)
            group_data.to_csv(file_path, index=False)
            print(f"文件 {source_file} 中超出阈值的数据已保存至: {safe_filename}")

        # 4. 按来源文件分开保存未超出阈值的数据
        within_groups = within_threshold_df.groupby('source_file')
        
        for source_file, group_data in within_groups:
            safe_filename = f"within_data_from_{os.path.splitext(source_file)[0]}.csv"
            file_path = os.path.join(output_folder, safe_filename)
            group_data.to_csv(file_path, index=False)
            print(f"文件 {source_file} 中未超出阈值的数据已保存至: {safe_filename}")
            

        # 5. 保存完整的预测结果
        result_df.to_csv(
            os.path.join(output_folder, 'all_predictions_with_deviation.csv'), 
            index=False
        )
        
        # 6. 保存文件统计信息
        file_stats = result_df.groupby('source_file').agg({
            'exceeds_threshold': ['count', 'sum', 'mean']
        }).round(4)
        file_stats.columns = ['total_count', 'exceeded_count', 'exceeded_ratio']
        file_stats.to_csv(os.path.join(output_folder, 'file_statistics.csv'))
        
        print(f"所有结果已保存到文件夹: {output_folder}")
    
    def visualize_results(self, result_df, output_folder):
        """
        结果可视化
        """
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(result_df['relative_deviation'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(self.threshold, color='red', linestyle='--', label=f'阈值 ({self.threshold})')
        plt.xlabel('相对偏差')
        plt.ylabel('频数')
        plt.title('相对偏差分布')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.scatter(result_df[self.target_column], result_df['predicted_value'], 
                   alpha=0.6, c=result_df['exceeds_threshold'], cmap='viridis')
        plt.plot([result_df[self.target_column].min(), result_df[self.target_column].max()], 
                [result_df[self.target_column].min(), result_df[self.target_column].max()], 
                'r--', alpha=0.8)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('实际值 vs 预测值')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'results_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()

#       import re
#       import os

    def sanitize_filename(self, filename):
        """
        清理文件名，移除或替换Windows系统中不允许的字符[9,10](@ref)。
        
        参数:
        filename: str, 需要清理的原始文件名
        
        返回:
        str, 清理后的安全文件名
        """
        # 定义Windows系统中不允许的字符模式[8,9](@ref)
        # 包括：<, >, :, ", /, \, |, ?, *, 换行符(\n), 回车符(\r)等
        illegal_chars_pattern = r'[<>:"/\\|?*\n\r]'
        
        # 使用下划线替换所有非法字符[10](@ref)
        safe_name = re.sub(illegal_chars_pattern, '_', filename)
        
        # 可选：移除文件名首尾可能产生的多余空格和点号（某些系统不允许文件名以点号结尾）
        safe_name = safe_name.strip().rstrip('.')
        
        # 确保文件名长度合理，避免过长[8](@ref)
        if len(safe_name) > 200:
            name_part, ext_part = os.path.splitext(safe_name)
            safe_name = name_part[:200] + ext_part
        
        return safe_name    
    
    def run_pipeline(self, folder_path, output_folder="./results"):
        """
        运行完整流程
        """
        print("=" * 60)
        print("开始多CSV文件建模与阈值筛选流程")
        print("=" * 60)
        
        # 1. 加载和合并数据
        print("\n1. 加载和合并CSV文件...")
        combined_df = self.load_and_combine_csv(folder_path)
        
        # 2. 数据预处理
        print("\n2. 数据预处理...")
        processed_df = self.preprocess_data(combined_df)
        
        # 3. 训练模型
        print("\n3. 训练线性回归模型...")
        X_scaled, y = self.train_model(processed_df)
        
        # 4. 应用阈值筛选
        print("\n4. 应用阈值筛选...")
        exceeded_threshold_df, within_threshold_df, result_df = self.apply_threshold_filter(
            processed_df, X_scaled, y
        )
        
        # 5. 分析特征分布（新增功能）
        print("\n5. 分析阈值内外特征分布...")
        distribution_results = self.analyze_feature_distributions(
            exceeded_threshold_df, within_threshold_df, output_folder
        )
        
        # 6. 保存结果
        print("\n6. 保存结果...")
        self.save_results(exceeded_threshold_df, result_df, output_folder, within_threshold_df)
        
        # 7. 可视化结果
        print("\n7. 生成可视化...")
        self.visualize_results(result_df, output_folder)
        
        print("\n" + "=" * 60)
        print("流程完成！")
        print("=" * 60)
        
        return result_df, exceeded_threshold_df, distribution_results

# 使用示例
if __name__ == "__main__":
    # 配置参数
    FEATURE_COLUMNS = ['Qv 体积流量\n(m3/h)', 'DP\nHVAC inlet\n(Pa)', 'N 鼓风机转速\n(rpm)', 'Diameter', '流速v1', '流速v2', '流速v3']  # 替换为实际的特征列名
    TARGET_COLUMN = 'Lp M1 麦克风总计值\n (dBA)'  # 替换为实际的目标列名
    THRESHOLD = 0.1  # 10%的偏差阈值
    CSV_FOLDER_PATH = "../csvdata444_yuanshi"  # 替换为CSV文件所在文件夹路径
    OUTPUT_FOLDER = "./model_results444"
    
    # 创建并运行模型
    model = MultiCSVModelWithThreshold(
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        threshold=THRESHOLD
    )
    
    # 运行完整流程
    results, exceeded_data, dist_results = model.run_pipeline(CSV_FOLDER_PATH, OUTPUT_FOLDER)
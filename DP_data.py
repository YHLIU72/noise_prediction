import pandas as pd
import numpy as np

def process_excel_polynomial_fit(excel_file):
    sheets = pd.read_excel(excel_file, sheet_name=None)
    # print(sheets)
    # print (df)

    polynomial_results={}
    # 新增：处理每个工作表的数据
    for sheet_name, sheet_data in sheets.items():
        print(f"\n处理工作表: {sheet_name}")
        polynomial_results[sheet_name]=[]
        # 跳过前3行，只读取指定列(3,5,12列对应索引2,4,11)
        # 注意：这里假设数据从第4行开始是有效数据行
        processed_data = sheet_data.iloc[2:, [2, 4, 11]]
        
        # 循环13次，每次读取9行数据
        for cycle in range(13):
            start_row = cycle * 9
            end_row = start_row + 9
            
            # 截取当前批次的9行数据
            batch_data = processed_data.iloc[start_row:end_row]
            
            # 移除空值行并转换为列表
            clean_data = batch_data.dropna()
            col3 = clean_data.iloc[:, 0].tolist()   # 第3列数据列表
            col5 = clean_data.iloc[:, 1].tolist()   # 第5列数据列表
            col12 = clean_data.iloc[:, 2].tolist()  # 第12列数据列表
            print(f"第{cycle+1}次数据：")
            print(f"  第3列（RPM）: {col3}")
            print(f"  第5列（V5）: {col5}")
            print(f"  第12列（V12）: {col12}")
            # 计算三列列表的最小长度
            min_length = min(len(col3), len(col5), len(col12))
            # 截断所有列表到最小长度，确保长度一致
            col3 = col3[:min_length]

            col5 = col5[:min_length]
            col12 = col12[:min_length]
            # 三阶多项式拟合需要至少4个数据点（次数+1）
            if min_length >= 4:
                maxrpm=max(col3)
                minrpm=min(col3)
                # 转换为numpy数组用于拟合计算
                x = np.array(col3, dtype=np.float64)  # 第三列作为x轴（自变量）
                y5 = np.array(col5, dtype=np.float64)  # 第五列作为y轴（因变量1）
                y12 = np.array(col12, dtype=np.float64)  # 第十二列作为y轴（因变量2）
                
                # 执行三阶多项式拟合（返回系数：[a, b, c, d] 对应 ax³+bx²+cx+d）
                coeffs_3v5 = np.polyfit(x, y5, 3)  # 第三列 vs 第五列拟合
                coeffs_3v12 = np.polyfit(x, y12, 3)  # 第三列 vs 第十二列拟合
                
                # 保存拟合结果（包含工作表名、周期和系数）
            
                polynomial_results[sheet_name].append({
                    'sheet': sheet_name,
                    'cycle': cycle + 1,
                    'maxrpm': maxrpm,
                    'minrpm': minrpm,
                    '3v5_coeffs': coeffs_3v5.tolist(),  # 转换为列表便于存储
                    '3v12_coeffs': coeffs_3v12.tolist()
                })
                print(f"第{cycle+1}次拟合完成：已保存三阶多项式系数")
            else:
                print(f"第{cycle+1}次数据不足（需要至少4个点，当前{min_length}个），跳过拟合")
    # print(polynomial_results[sheet_name])
    return polynomial_results

def predict_pressure(polynomial_results, model, real_speed, real_power):
    """
    根据真实转速、功率和型号预测压力值
    
    参数:
        polynomial_results: process_excel_polynomial_fit返回的拟合结果字典
        model: 型号（对应工作表名称）
        real_speed: 真实转速（对应第三列数据）
        real_power: 真实功率（对应第五列数据）
    
    返回:
        预测的压力值（对应第十二列数据）
    """
    hole_diameter={
        '1': 35,
        '2': 45,
        '3': 58,
        '4': 70,
        '5': 82,
        '6': 90,
        '7': 95,
        '8': 100,
        '9': 105,
        '10': 110,
        '11': 120,
        '12': 135,
        '13': 150,
    }


    # 检查型号是否存在于拟合结果中
    if model not in polynomial_results:
        raise ValueError(f"型号 '{model}' 不存在于拟合结果中")
    
    model_cycles = polynomial_results[model]
    if not model_cycles:
        raise ValueError(f"型号 '{model}' 没有可用的拟合循环数据")
    
    min_error = float('inf')
    best_cycle = None
    
    # 遍历所有循环，找到功率预测误差最小的循环
    for cycle_data in model_cycles:
        if real_speed>=cycle_data['minrpm'] and real_speed<=cycle_data['maxrpm']:
            cycle = cycle_data['cycle']
            # 获取3v5拟合系数并创建多项式
            coeffs_3v5 = cycle_data['3v5_coeffs']
            poly_3v5 = np.poly1d(coeffs_3v5)  # 创建三阶多项式函数
            
            # 预测功率并计算与真实功率的绝对误差
            predicted_power = poly_3v5(real_speed)
            current_error = abs(predicted_power - real_power)
            
            # 跟踪最小误差对应的循环
            if current_error < min_error:
                min_error = current_error
                best_cycle = cycle_data
        else:
            print(f"转速 {real_speed} 不在型号{model} 孔径{hole_diameter[str(cycle_data['cycle'])]} 的有效范围内")
    
    if best_cycle is None:
        raise ValueError(f"型号 '{model}' 没有找到转速 {real_speed} 有效的拟合循环")
    
    # 使用最优循环的3v12拟合系数预测压力
    coeffs_3v12 = best_cycle['3v12_coeffs']
    poly_3v12 = np.poly1d(coeffs_3v12)  # 创建三阶多项式函数
    predicted_pressure = poly_3v12(real_speed)
    
    print(f"最佳孔径: {hole_diameter[str(best_cycle['cycle'])]},真实转速: {real_speed}, 真实功率: {real_power},预测功率: {predicted_power:.4f}, 最小功率误差: {min_error:.4f}，误差占比: {min_error/real_power:.4f}")
    return predicted_pressure

# # 使用示例（需先运行拟合函数获取结果）
# if __name__ == "__main__":
#     # 1. 首先获取多项式拟合结果
#     fit_results = process_excel_polynomial_fit('blower1.xlsx')
#     speed=[500,1000,1500,2000,2500,3000,3500]
#     power=[20,50,100,150,200,250,300]
#     for sd,pw in zip(speed,power):
#         # 2. 调用预测函数（示例参数）
#         try:
#             pressure = predict_pressure(
#                 polynomial_results=fit_results,
#                 model='SRH',  # 替换为实际工作表名称（型号）
#                 real_speed=sd,  # 替换为实际转速值
#                 real_power=pw   # 替换为实际功率值
#             )
#             print(f"预测压力值: {pressure:.4f}")
#         except ValueError as e:
#             print(f"预测失败: {e}")

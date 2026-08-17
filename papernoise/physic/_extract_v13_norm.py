"""临时脚本：提取 V13 训练时的归一化参数"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'f:\lyh\paddlespeech\papernoise\physic')
os.chdir(r'f:\lyh\paddlespeech\papernoise\physic')

from PIMBCN_data_0614_v13 import PIMBCNDataset

ds = PIMBCNDataset(
    directory_path=r'F:\lyh\paddlespeech\csvdata333',
    input_cols=[4, 5, 6], oaspl_col=11, octave_col=12, spectrum_col=13,
    type_col=3, mode_col=2, val_split=0.2, is_validation=False, augment=True)

print('=== V13 训练归一化参数 ===')
print(f'input_mean = {ds.input_mean.tolist()}')
print(f'input_std  = {ds.input_std.tolist()}')
print(f'训练集: {len(ds.train_indices)} 条 | 验证集: {len(ds.val_indices)} 条 | 总: {len(ds.data)} 条')

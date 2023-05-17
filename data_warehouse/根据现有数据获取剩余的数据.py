import os,shutil

copy_dir = r'./train_raw'

dealed_dir = r'E:\datasets_2022_12_7\jianpu_transformer_samples_2023_5_16\train-checked'
for i in os.listdir(dealed_dir):
    if i.endswith('.xml'):
        src_path = os.path.join(copy_dir, i.replace('.xml','.txt'))
        dst_path = os.path.join(dealed_dir, i.replace('.xml','.txt'))

        shutil.copy(src_path, dst_path)








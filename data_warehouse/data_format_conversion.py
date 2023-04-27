from gc import collect
import os
from textwrap import indent
import numpy as np
import json
import shutil
"""
该脚本的主要功能:  1. 把文件夹中成对的图片和标签======>模型所需要的格式
                  2. 把vocab.txt转换成json格式
"""
collect_data_together = True # 是否把各个文件夹下的图片收集下来
collect_data_dir = r'./formula_images_processed'
if os.path.exists(collect_data_dir):
    shutil.rmtree(collect_data_dir)
os.mkdir(collect_data_dir)

with open('./vocab.txt', 'r', encoding='utf-8') as reader:
    vocab_con = reader.readlines()
vocab_dict = {
"<PAD>": 0, 
"<SOS>": 1, 
"<EOS>": 2, 
"<UNK>": 3, 
    
}

for no, vocab_str in enumerate(vocab_con):
    vocab_str = vocab_str.strip()
    vocab_dict[vocab_str] = no + 4
with open('./vocab.json', 'w', encoding='utf-8') as f:   
    json.dump(vocab_dict, f, ensure_ascii=False, indent=4)



data_dict = {
    'train': './train_raw',
    'val': './val_raw',
    # 'test': './test_raw'
}
support_img_format = ['.png', '.jpg']



def get_txtfile_con(file_path):
    with open(file_path, 'r', encoding='utf-8') as reader:
        con = reader.readline()
    return con


tot_file_count = 0
def record_info(label_path, writer_tot, writer_type):
    '''
    label_path: txt文件路径
    writer_tot: 记录 总的标签文件的 句柄
    writer_type: 记录 ['train','val'] 文件的 句柄
    '''
    # 先写 训练集/验证集(也就是具体数据集的内容)
    global tot_file_count
    this_img_label = get_txtfile_con(label_path)
    tot_file_count += 1
    want_write_con = str(tot_file_count) + ' ' + file_full_name + ' ' + 'basic\n' 
    writer_type.write(want_write_con)
    
    # 在写总的 标签本
    want_write_con = this_img_label.strip() + '\n'
    writer_tot.write(want_write_con)




with open('./im2latex_formulas.lst', 'w', encoding='utf-8') as writer_tot_label:
    for data_type, data_dir in data_dict.items():
        if data_type=='train':
            with open('./im2latex_train.lst', 'w', encoding='utf-8') as writer_train:
                file_list = os.listdir(data_dir)
                np.random.shuffle(file_list)
                for file_full_name in file_list:
                    file_name, extension_name = os.path.splitext(file_full_name)
                    if extension_name in support_img_format:                      
                        src_img_path = os.path.join(data_dir, file_full_name)
                        src_label_path = os.path.join(data_dir, file_name + '.txt')
                        if os.path.exists(src_label_path):
                            record_info(src_label_path, writer_tot_label, writer_train)
                      
        elif data_type=='val':
            with open('./im2latex_val.lst', 'w', encoding='utf-8') as writer_val:
                file_list = os.listdir(data_dir)
                np.random.shuffle(file_list)
                for file_full_name in file_list:
                    file_name, extension_name = os.path.splitext(file_full_name)
                    if extension_name in support_img_format:                      
                        src_img_path = os.path.join(data_dir, file_full_name)
                        src_label_path = os.path.join(data_dir, file_name + '.txt')
                        if os.path.exists(src_label_path):
                            record_info(src_label_path, writer_tot_label,writer_val)
                                     
        elif data_type=='test':
            with open('./im2latex_test.lst', 'w', encoding='utf-8') as writer_test:
                file_list = os.listdir(data_dir)
                np.random.shuffle(file_list)
                for file_full_name in file_list:
                    file_name, extension_name = os.path.splitext(file_full_name)
                    if extension_name in support_img_format:                      
                        src_img_path = os.path.join(data_dir, file_full_name)
                        src_label_path = os.path.join(data_dir, file_name + '.txt')
                        if os.path.exists(src_label_path):
                            record_info(src_label_path, writer_tot_label, writer_test)
        else:
            raise 'found unknown data type....'
if collect_data_together:
    for data_type, data_dir in data_dict.items():
        for i in os.listdir(data_dir):
            src_path = os.path.join(data_dir, i)
            dst_path = os.path.join(collect_data_dir, i)
            shutil.copy(src_path, dst_path)
        
        
    # collect_data_dir
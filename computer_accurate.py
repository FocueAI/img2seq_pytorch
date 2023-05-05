import os,shutil
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import difflib
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
# from image_to_latex.lit_models import LitResNetTransformer
from models.resnet_transformer import ResNetTransformer
import torch, os
from create_model import Img2Seq
# 加载模型
weight_path = r'./checkpoint.pt'
img2seq_model, loss_fn, tokenizer = Img2Seq(d_model=128*2, dim_feedforward=256*2, nhead=8,
                                          dropout=0.3, num_decoder_layers=6,
                                          max_output_len=150,
                                          vocab_path='data_warehouse/vocab.json'
                                          )
img2seq_model.eval()
img2seq_model.load_state_dict(torch.load(weight_path))


def get_edit_distance(str1, str2) -> int:
    """
    计算两个串的编辑距离，支持str和list类型   
    str1: label
    str2: predict
    """
    leven_cost = 0
    sequence_match = difflib.SequenceMatcher(None, str2, str1)    
    for tag, index_1, index_2, index_j1, index_j2 in sequence_match.get_opcodes():
        if tag == 'replace':
            leven_cost += max(index_2-index_1, index_j2-index_j1)
        elif tag == 'insert':
            leven_cost += (index_j2-index_j1)
        elif tag == 'delete':
            leven_cost += (index_2-index_1)
    return leven_cost

result_analysis = False # 是否保留标签与预测值的对比结果?  True: 保留    False: 不保留
if result_analysis:
    save_err_dirs = r'./result/label_pre_compare_raw'
    if os.path.exists(save_err_dirs):
        shutil.rmtree(save_err_dirs)
    os.makedirs(save_err_dirs)


support_img_format = ['.png','.jpg']

words_num, word_error_num = 0, 0
right_count, tot_count = 0, 0
transform = ToTensorV2()
file_dir = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/test_raw'
for i in os.listdir(file_dir):
    file_name, extend_name = os.path.splitext(i)
    if extend_name in support_img_format:
        file_path = os.path.join(file_dir, i)
        lable_path = file_path.replace(extend_name,'.txt')
        with open(lable_path, 'r', encoding='utf-8') as reader:
            label_con = reader.readline()
        

        image = Image.open(file_path).convert("RGB")
        image_tensor = transform(image=np.array(image))["image"]  # type: ignore

        pred = img2seq_model.predict(image_tensor.unsqueeze(0).float())[0]
        # print(f'len:{len(pred)},raw-pred:{pred}')
        decoded = tokenizer.decode(pred.tolist())
        pre_str = " ".join(decoded)
        # print(f'pre_str:{pre_str}') # 字符串--->   < 6 4 > < 6 1 ' > 7 . < 6 >
        print('-='*5)
        new_predi_val   = pre_str.strip().replace("(","").replace(")","").split(' ')
        new_label_val = label_con.strip().replace("(","").replace(")","").split(' ')
        print(f'new_predi_val:{new_predi_val}')
        print(f'new_label_val:{new_label_val}')
        # --------------------- 统计准确率 ------------------- # 
        #  step1: 统计字符级别的准确率
        words_n = len(new_label_val)  # 获取每个句子的字数
        words_num += words_n          # 把句子的总字数加上
        edit_distance = get_edit_distance(new_label_val,new_predi_val)
        if edit_distance <= words_n:  # 当编辑距离小于等于句子字数时
            word_error_num += edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            word_error_num += words_n  # 就直接加句子本来的总字数就好了
        # 字符集准确率         
        print(f"word_right_num:{words_num-word_error_num},words_num:{words_num},word_right_rotio: {1-word_error_num/words_num}")
        
        # step2: 按整个条目统计准确率
        if pre_str.strip().replace("(","").replace(")","").replace(" ","") == label_con.strip().replace("(","").replace(")","").replace(" ",""): 
            right_count += 1
        else:
            if result_analysis: # 是否保留标签与预测值的对比结果
                src_img = os.path.join(file_dir,i)
                dst_img = os.path.join(save_err_dirs, i)
                err_compare_txt_path = os.path.join(save_err_dirs, file_name + '.txt')
                with open(err_compare_txt_path,'w',encoding='utf-8') as writer:
                    #writer.write(pre_val.strip())
                    want_write_con = 'preval: %s'% pre_str.strip() + '\n' +\
                                        '-label: %s'% label_con.strip()
                    writer.write(want_write_con)
                shutil.copy(src_img, dst_img)
            
        tot_count += 1
        # 整个条目的准确率
        print(f'right:{right_count},tot:{tot_count},acc:{right_count/tot_count}')
print('-'*10,'整体的结果','-'*10)
print(f'right:{right_count},tot:{tot_count},acc:{right_count/tot_count}')
print(f"word_right_num:{words_num-word_error_num},words_num:{words_num},word_right_rotio: {1-word_error_num/words_num}")




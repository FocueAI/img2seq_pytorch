# img2seq_pytorch
## 一.功能:
    将图片翻译成一系列字符串(可用于OCR,公式识别,乐谱识别)  
## 二.效果:(使用的是音乐小节数据集)
   ***字符级别准确率:*** word_right_num:17521,words_num:17664,word_right_rotio: 0.9919044384057971  
   ***条目级别准确率:*** right:1493,tot:1547,acc:0.9650937297996122  
## 三.缺点:
    **目前模型比较大,大概有3.9G左右**  
<font color=red>todo-list:</font> 1. *自己手写transformer替换nn.transformer(方便修改transformer的结构,利用交互注意力画注意力热图,以及确定bbox)*   
       2. *想法减少模型的容量*

## 四. 该工程的使用步骤  :
   step1: 按照data_warehouse/readme.md中的要求准备数据   
   step2: 执行 '''python train.py''' 开始训练数据  
   step3: 执行 '''python computer_accurate.py''' 就可以是使用训练的权重文件对指定的数据做准确率的评估  
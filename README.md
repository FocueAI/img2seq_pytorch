# img2seq_pytorch
将图片翻译成一系列字符串(可用于OCR,公式识别,乐谱识别) 
***字符级别准确率:*** word_right_num:17521,words_num:17664,word_right_rotio: 0.9919044384057971  
***条目级别准确率:*** right:1493,tot:1547,acc:0.9650937297996122  
*缺点:目前模型比较大,大概有3.9G左右*  
TODO: 1. *自己手写transformer替换nn.transformer(方便修改transformer的结构,利用交互注意力画注意力热图,以及确定bbox)*   
      2. *想法减少模型的容量*


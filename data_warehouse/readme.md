==================================模型训练准备数据=====================================
一. 需要准备的
    1. train_raw/  val_raw/  训练数据包和验证数据包 每个包中包含 <图片文件>和对应<标签文件>
    2. vocab.txt   每一行就是一个token
二. 手动执行脚本->生成模型训练所需的文件
    '''python data_format_conversion.py'''


三. 上一步骤中生成文件的格式解释

    1. 在该工程的根目录 data_warehouse/  放置字典文件 vocab.json 具体格式在说{'token1':index1, 'token2':index2, ... ,'tokenn':'indexn'}
    2. 在该工程的根目录 data_warehouse/  放置train/ val/ test/ 数据的总标签  im2latex_formulas.lst
        eg:     label1_string1 label1_string2 label1_string3 label1_string4  //一个图片的标签
                label2_string1 label2_string2 label2_string3 label2_string4  //一个图片的标签
    3. 在该工程的根目录 data/  放置 train/ val/ test/ 各自图像的名称,已经对应标签在2.中的序号
        训练集:   im2latex_train.lst 具体格式
                  标签行号 图片名称 类型(一个无用的信息)
        验证集/测试集 与训练集格式相同     z
    4. 在该工程的根目录 data/formula_images_processed 放置好图片数据




			



import os,shutil

has_question_file_l=[277, 375, 400, 434, 455, 459, 475, 883, 959, 975, 1003, 1019, 1084, 1152, 1250, 1470, 1634, 1746, 1751, 2104, 2190, 2233, 2261, 2271, 2335, 2366, 2368, 2403, 2617, 2662, 2671, 2897, 2925, 2974, 3028, 3062, 3081, 3196, 3243, 3304, 3482, 3500, 3642]

with open('./im2latex_train.lst','r',encoding='utf-8') as reader:
    contents = reader.readlines()

dst_dir = r'need-checked'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

src_dir = r'./train-checked'
for content in contents:
    # print(f'content:{content}')
    index, file_name, _ = content.split(' ')
    # print(f'index:{index}, file_name:{file_name}')
    if int(index) in has_question_file_l:
        real_file_name = os.path.splitext(file_name)[0]
        src_img_path = os.path.join(src_dir, file_name)
        src_txt_path = os.path.join(src_dir, real_file_name+'.txt')
        src_xml_path = os.path.join(src_dir, real_file_name + '.xml')

        dst_img_path = os.path.join(dst_dir, file_name)
        dst_txt_path = os.path.join(dst_dir, real_file_name+'.txt')
        dst_xml_path = os.path.join(dst_dir, real_file_name + '.xml')

        shutil.move(src_img_path, dst_img_path)
        shutil.move(src_txt_path, dst_txt_path)
        shutil.move(src_xml_path, dst_xml_path)
        print('正在移动....')




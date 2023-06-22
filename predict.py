import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Resize
import albumentations as A

from PIL import Image, ImageDraw
# from image_to_latex.lit_models import LitResNetTransformer
from models.resnet_transformer import ResNetTransformer
import torch
from create_model import Img2Seq
import time

# 加载模型
# weight_path = r'/usr/hsc_project/latexOCR/image-to-latex/outputs/2023-04-26/02-11-57/image-to-latex/whz39oao/checkpoints/epoch=13-val/loss=0.19-val/cer=0.08.ckpt'
weight_path = r'./checkpoint_6_14.pt'
img2seq_model, loss_fn, tokenizer = Img2Seq(d_model=128*2, dim_feedforward=256*2, nhead=8,
                                          dropout=0.3, num_decoder_layers=6,
                                          max_output_len=150,
                                          vocab_path='data_warehouse/vocab.json'
                                          )
img2seq_model.eval()
img2seq_model.load_state_dict(torch.load(weight_path))
print('model load over .....')

# print(f'img2seq_model:{img2seq_model}')
transform = A.Compose([
    # Resize(int(50), int(275)),  # 这行代码让模型一下子好了起来
    ToTensorV2()
])
# file_path = r'./data_warehouse/train_raw/2012020616404276_system_index_0_staff_0_measure_0.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012020616404276_system_index_0_staff_0_measure_0.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012070617295804_system_index_4_staff_0_measure_1.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012020616404276_system_index_4_staff_0_measure_2.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012070617301276_system_index_0_staff_0_measure_0.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012070617301276_system_index_1_staff_0_measure_3.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012020616404966_system_index_8_staff_0_measure_0.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012070617300846_system_index_4_staff_0_measure_2.jpg'
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012020616463864_system_index_4_staff_0_measure_2.jpg' # 效果还可以
# file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked/2012020616444127_system_index_0_staff_0_measure_2.jpg'

def letterbox_image(image, size=(275,50)):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (255,255,255))
    # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    new_image.paste(image, (0, 0))
    return new_image

res_show_img = r'./res_show'
if os.path.exists(res_show_img):
    shutil.rmtree(res_show_img)
os.mkdir(res_show_img)

img_dir = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train-checked'
# img_dir = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/test_raw'
for i in os.listdir(img_dir):
    if i.endswith('.jpg'):
        file_path = os.path.join(img_dir, i)
        save_path = os.path.join(res_show_img, i)
        save_txt_path = os.path.join(res_show_img, i.replace('.jpg','.txt'))
        image = Image.open(file_path).convert("RGB")
        img_w, img_h = image.size
        ########## resize image -begin #####
        # --------------- 贴图之前先resize高度到额定的高度,宽度在补充 ---------- #
        new_w, new_h = int(275), int(50)
        image = letterbox_image(image, (new_w, new_h))
        img_w, img_h = image.size
        
        

        # if img_w > new_w or img_h > new_h:
        #     new_w, new_h = img_w, img_h
        
        # pil_img_bj = Image.new("RGB",(new_w,new_h),(255, 255, 255))
        # pil_img_bj.paste(image,(0,0)) # 对齐左上角粘贴
        # image = pil_img_bj
        # img_w, img_h = image.size
 
        
        ########## resize image -end #####
        image_tensor = transform(image=np.array(image))["image"]  # type: ignore
        # pred = img2seq_model(image_tensor.unsqueeze(0).float())[0]  # type: ignore
        # with torch.no_grad(): # 暂时先不加
        print('='*5)
        begin_time = time.time()
        pred, boxes = img2seq_model.predict(image_tensor.unsqueeze(0).float())
        print(f'model-inference-time:{time.time()-begin_time}')
        # print(f'len:{len(pred)},raw-pred:{pred}')
        decoded = tokenizer.decode(pred.tolist())
        print(f'len:{len(pred)},raw-pred:{decoded}')
        boxes_l = boxes.tolist()
        ################# transform 的逆操作 begin################### 不加这个代码块,问题也不大
        # img_numpy = image_tensor.numpy().transpose(1,2,0)
        # image = Image.fromarray(img_numpy)
        # img_w, img_h = image.size
        ################# transform 的逆操作 end###################
        draw = ImageDraw.Draw(image)
        for box in boxes_l:
            center_x, center_y, box_w, box_h = box
            if box_w < 1e-3 or box_h < 1e-3:
                continue
            # box的左上角坐标
            x0 = (center_x - box_w/2) * img_w
            y0 = (center_y - box_h/2) * img_h
            # box的右下角坐标
            x1 = (center_x + box_w/2) * img_w
            y1 = (center_y + box_h/2) * img_h
            print(f'rect:{[x0,y0,x1,y1]}')
            draw.rectangle((x0, y0, x1, y1), outline='red', width=1)
        image.save(save_path)   
        decoded_str = " ".join(decoded)
        with open(save_txt_path,'w',encoding='utf-8') as writer:
            writer.write(decoded_str)
         
            
            
        # decoded_str = " ".join(decoded)
        # print(f'decoded_str:{decoded_str}')





# def load_model():
#     global transform
#     global weight_path
#     lit_model = ResNetTransformer.load_from_checkpoint(weight_path)
#     lit_model.freeze()
#     transform = ToTensorV2()
# # 模型预测
# def predict(file_path):
#     image = Image.open(file_path).convert("RGB")
#     image_tensor = transform(image=np.array(image))["image"]  # type: ignore
#     pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0]  # type: ignore
#     decoded = lit_model.tokenizer.decode(pred.tolist())  # type: ignore
#     decoded_str = " ".join(decoded)
#     return decoded_str
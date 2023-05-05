import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
# from image_to_latex.lit_models import LitResNetTransformer
from models.resnet_transformer import ResNetTransformer
import torch
from create_model import Img2Seq

# 加载模型
# weight_path = r'/usr/hsc_project/latexOCR/image-to-latex/outputs/2023-04-26/02-11-57/image-to-latex/whz39oao/checkpoints/epoch=13-val/loss=0.19-val/cer=0.08.ckpt'
weight_path = r'./checkpoint.pt'
img2seq_model, loss_fn, tokenizer = Img2Seq(d_model=128, dim_feedforward=256, nhead=8,
                                          dropout=0.3, num_decoder_layers=3,
                                          max_output_len=150,
                                          vocab_path='data_warehouse/vocab.json'
                                          )
img2seq_model.eval()
img2seq_model.load_state_dict(torch.load(weight_path))


# print(f'img2seq_model:{img2seq_model}')
transform = ToTensorV2()
# file_path = r'./data_warehouse/train_raw/2012020616404276_system_index_0_staff_0_measure_0.jpg'
file_path = r'/usr/hsc_project/latexOCR/img2seq_pytorch/data_warehouse/train_raw/2012020616404276_system_index_2_staff_0_measure_6.jpg'
image = Image.open(file_path).convert("RGB")
image_tensor = transform(image=np.array(image))["image"]  # type: ignore
# pred = img2seq_model(image_tensor.unsqueeze(0).float())[0]  # type: ignore
# with torch.no_grad(): # 暂时先不加
pred = img2seq_model.predict(image_tensor.unsqueeze(0).float())[0]
print(f'len:{len(pred)},raw-pred:{pred}')
decoded = tokenizer.decode(pred.tolist())
decoded_str = " ".join(decoded)
print(f'decoded_str:{decoded_str}')





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
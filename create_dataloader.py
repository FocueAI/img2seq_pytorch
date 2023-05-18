import os, random
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from utils.data_deal.dataset import BaseDataset, Tokenizer, get_all_formulas, get_split
from torch.utils.data import DataLoader
# 暂时先这样用
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CreateDataloader:
    def __init__(self,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 dir_path: str = 'None',
                 train_dir_name: str = 'train',
                 val_dir_name: str = 'val'
                 ):
        self.val_dataset = None
        self.train_dataset = None
        self.train_dir = os.path.join(dir_path, train_dir_name)
        self.val_dir = os.path.join(dir_path, val_dir_name)
        self.__dict__.update(locals())
        self.vocab_file = os.path.join(self.dir_path, "vocab.json")  # 数据的字符字典
        formula_file = os.path.join(self.dir_path, "im2latex_formulas.lst")  # 总标签文件
        if not os.path.exists(formula_file):
            raise FileExistsError(f"can't find {formula_file}")
        self.all_formulas, self.all_bboxes = get_all_formulas(formula_file)  # 获取所有数据的标签列表
        # 为图片的数据增强做准备
        self.transform = {
            "train": A.Compose(  # a. 训练数据的处理方式(包含数据增强)
                [
                    #A.Affine(scale=(0.6, 1.0), rotate=(-1, 1), cval=255, p=0.5), TODO:由于现在要加上bbox的回归,这里先不对原图做扭曲操作!!!
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(1, 1), p=0.5),
                    ToTensorV2(),
                ]
            ),
            "val/test": ToTensorV2(),  # b. 测试与验证数据的处理方式
        }

    def collate_fn(self, batch):
        images, formulas, boxes = zip(*batch)
        B = len(images)
        max_H = max(image.shape[1] for image in images)  # 找出该批次中图像最大---高度值
        max_W = max(image.shape[2] for image in images)  # 找出该批次中图像最大---宽度值
        max_length = max(len(formula) for formula in formulas)  # 找出该批次中标签最长---长度值
        padded_images = torch.zeros((B, 3, max_H, max_W))
        batched_indices = torch.zeros((B, max_length + 2), dtype=torch.long)
        boxes_tmp = torch.zeros((B,max_length+2,4),dtype=torch.float32)
        for i in range(B):
            H, W = images[i].shape[1], images[i].shape[2]
            y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
            padded_images[i, :, y: y + H, x: x + W] = images[i]  # 在原图随机完整的放在整块白色画布上---->该批次最大宽高的尺寸
            this_img_bbox_tmp = []
            for this_box in boxes[i]: # boxes[i]-->表示该批次中第i张图像的所有所有符号的bbox(当然(0,0,0,0)表示占位box,其本身没有实用意义)
                if this_box[0] == this_box[1] == this_box[2] == this_box[3]:
                    this_img_bbox_tmp.append([this_box[0], this_box[1], this_box[2], this_box[3]])
                else: # 根据原图的位置,平移box.并将其宽高坐标缩放至[0,1]范围
                    box_x0, box_y0, box_x1, box_y1 = (this_box[0]+x)/max_W, (this_box[1]+y)/max_H, (this_box[2]+x)/max_W, (this_box[3]+y)/max_H
                    assert box_x0 < box_x1
                    assert box_y0 < box_y1
                    center_x = (box_x0+box_x1)/2
                    center_y = (box_y0+box_y1)/2
                    box_w = box_x1 - box_x0
                    box_h = box_y1 - box_y0
                    this_img_bbox_tmp.append([center_x, center_y, box_w, box_h])
            # boxes[i] = this_img_bbox_tmp

            indices = self.tokenizer.encode(formulas[i])
            boxes_tmp[i, 1:len(indices)-1] = torch.from_numpy(np.array(this_img_bbox_tmp)).float() # 因为在字符的开头插入了<bos>
            batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded_images, batched_indices, boxes_tmp

    def do(self):
        """
        :return:train_dataloader, val_dataloader
        """
        self.tokenizer = Tokenizer.load(self.vocab_file)
        # ------------------------------  训练集处理 -------------------------------------#
        train_image_names, train_formulas, train_bboxes = get_split(
            self.all_formulas,
            self.all_bboxes,
            os.path.join(self.dir_path, "im2latex_train.lst")  # 该文件的格式"序号 图片名称 无用的信息"
        )
        self.train_dataset = BaseDataset(  # 最基本的dataset类
            root_dir=self.train_dir,
            img_filenames_l=train_image_names,
            formulas_l=train_formulas,
            bboxes_l=train_bboxes,
            transform=self.transform["train"],  # 数据处理（增强）相关的参数
        )
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        # ------------------------------ 验证集处理 ---------------------------------------#
        val_image_names, val_formulas, val_bboxes = get_split(
            self.all_formulas,
            self.all_bboxes,
            os.path.join(self.dir_path, "im2latex_val.lst")  # 该文件的格式"序号 图片名称 无用的信息"
        )
        self.val_dataset = BaseDataset(  # 最基本的dataset类
            root_dir=self.val_dir,
            img_filenames_l=val_image_names,
            formulas_l=val_formulas,
            bboxes_l=val_bboxes,
            transform=self.transform["val/test"],  # 数据处理（增强）相关的参数
        )
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

        return train_dataloader, val_dataloader

if __name__ == '__main__':
    dataloader_creator = CreateDataloader(batch_size=2, num_workers=0, pin_memory=False, dir_path='data_warehouse',
                                          train_dir_name='train-checked', val_dir_name='train-checked')
    train_dataloader, val_dataloader = dataloader_creator.do()
    for epoch in train_dataloader:
        x, y, box = epoch
        print(f'x-shape:{x.shape}')
        print(f'y-shape:{y.shape}')
        print(f'box-shape:{box.shape}')
        print(f'box:{box}')
        print(f'y:{y}')

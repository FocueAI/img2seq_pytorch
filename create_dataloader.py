import os, random
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
        self.all_formulas = get_all_formulas(formula_file)  # 获取所有数据的标签列表
        # 为图片的数据增强做准备
        self.transform = {
            "train": A.Compose(  # a. 训练数据的处理方式(包含数据增强)
                [
                    A.Affine(scale=(0.6, 1.0), rotate=(-1, 1), cval=255, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(1, 1), p=0.5),
                    ToTensorV2(),
                ]
            ),
            "val/test": ToTensorV2(),  # b. 测试与验证数据的处理方式
        }

    def collate_fn(self, batch):
        images, formulas = zip(*batch)
        B = len(images)
        max_H = max(image.shape[1] for image in images)  # 找出该批次中图像最大---高度值
        max_W = max(image.shape[2] for image in images)  # 找出该批次中图像最大---宽度值
        max_length = max(len(formula) for formula in formulas)  # 找出该批次中标签最长---长度值
        padded_images = torch.zeros((B, 3, max_H, max_W))   # TODO: 我人为3通道的图片比单通道的好
        batched_indices = torch.zeros((B, max_length + 2), dtype=torch.long)
        for i in range(B):
            H, W = images[i].shape[1], images[i].shape[2]
            y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
            padded_images[i, :, y: y + H, x: x + W] = images[i]  # 在原图随机完整的放在整块白色画布上---->该批次最大宽高的尺寸
            indices = self.tokenizer.encode(formulas[i])
            batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded_images, batched_indices

    def do(self):
        """
        :return:train_dataloader, val_dataloader
        """
        self.tokenizer = Tokenizer.load(self.vocab_file)
        # ------------------------------  训练集处理 -------------------------------------#
        train_image_names, train_formulas = get_split(
            self.all_formulas,
            os.path.join(self.dir_path, "im2latex_train.lst")  # 该文件的格式"序号 图片名称 无用的信息"
        )
        self.train_dataset = BaseDataset(  # 最基本的dataset类
            root_dir=self.train_dir,
            img_filenames_l=train_image_names,
            formulas_l=train_formulas,
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
        val_image_names, val_formulas = get_split(
            self.all_formulas,
            os.path.join(self.dir_path, "im2latex_val.lst")  # 该文件的格式"序号 图片名称 无用的信息"
        )
        self.val_dataset = BaseDataset(  # 最基本的dataset类
            root_dir=self.val_dir,
            img_filenames_l=val_image_names,
            formulas_l=val_formulas,
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
                                          train_dir_name='train_raw', val_dir_name='val_raw')
    train_dataloader, val_dataloader = dataloader_creator.do()
    for epoch in train_dataloader:
        x, y = epoch
        print(f'x-shape:{x.shape}')
        print(f'y-shape:{y.shape}')

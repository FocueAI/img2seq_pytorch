import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import json

def letterbox_image(image, boxes, size=(275,50)):
    '''resize image with unchanged aspect ratio using padding
    image: 原pil图像
    boxes: 该图像对应的box列表 [[x0,y0,x1,y1],[x0,y0,x1,y1],...] --->box的格式为左上角坐标与右下角坐标
    size:  (处理后的图像的_w, 处理后的图像的_h)
    '''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih) # 想要的尺寸/图像原本的尺寸
    nw = int(iw*scale)      # 新的图像的尺寸_w
    nh = int(ih*scale)      # 新的图像的尺寸_h

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (255,255,255))
    # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    new_image.paste(image, (0, 0))
    ########### 由于图像的位置发生变化,对应的box也要产生相应的变换
    arr_boxes = np.array(boxes)
    # arr_boxes[:,[0,2]] = arr_boxes[:,[0,2]] * scale # +dx
    # arr_boxes[:,[1,3]] = arr_boxes[:,[1,3]] * scale # +dy 为了以后的放置位置随机做准备(现在默认是放在左上角)
    # scale=1
    boxes_x = arr_boxes[:,[0,2]].astype(np.float) * scale
    boxes_y = arr_boxes[:,[1,3]].astype(np.float) * scale
    lables = arr_boxes[:,-1]
    
    boxes_x_l = boxes_x.tolist()
    boxes_y_l = boxes_y.tolist()
    lables_l = lables.tolist()
    
    boxes = [[x[0],y[0],x[1],y[1],l] for x,y,l in zip(boxes_x_l, boxes_y_l, lables_l)]
    ###########
    
    return new_image,boxes

class BaseDataset(Dataset):
    """
    获取图像数据+标签的基础类
    """

    def __init__(self,
                 root_dir: str,  # 图像数据(包含txt标签)存放的根文件地址
                 img_filenames_l: List[str],  # ['**1.jpg','**2.jpg',...'**n.png']
                 formulas_l: List[List[str]],  # [['<','2','>' ],["3", "'", "."],...[]]
                 bboxes_l: List,  # [ [[box0_x0,box0_y0,box0_x1,box0_y1],[],[] ],
                 transform: Optional[Callable] = None,  # 图像数据增强相关的
                 ) -> None:
        super(BaseDataset, self).__init__()
        assert len(img_filenames_l) == len(formulas_l)
        self.__dict__.update(locals())

    def __len__(self) -> int:
        return len(self.img_filenames_l)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset at the given idx."""
        image_filename, formula, bbox = self.img_filenames_l[idx], self.formulas_l[idx], self.bboxes_l[idx]
        image_filepath = os.path.join(self.root_dir, image_filename)
        if os.path.exists(image_filepath):
            image = Image.open(image_filepath).convert('RGB')
            image, bbox = letterbox_image(image, bbox)

        else:
            # Returns a blank image if cannot find the image
            image = Image.fromarray(np.full((64, 128), 255, dtype=np.uint8))
            formula = []
            bbox = []
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
            # TODO: bbox可能会因为图像做完数据增强后, box的位置和形状也会随着改变
        return image, formula, bbox


class Tokenizer:
    """
    将标签中的字符串转换成计算机能理解的数字(索引-->相当于是类别),以后做每个字符的交叉熵损失
    """

    def __init__(self, token_to_index: Optional[Dict[str, int]] = None) -> None:
        """
        :param token_to_index: {'<':0, '>':1, '<PAD>':2, .......}
        """
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]
        self.token_to_index_init_full = False  #
        if token_to_index:
            self.token_to_index_init_full = True
            self.token_to_index = token_to_index
            self.index_to_token = {index: token for token, index in self.token_to_index.items()}
            # 取出特殊字符的索引
            self.pad_index = self.token_to_index[self.pad_token]
            self.sos_index = self.token_to_index[self.sos_token]
            self.eos_index = self.token_to_index[self.eos_token]
            self.unk_index = self.token_to_index[self.unk_token]
        else:
            self.token_to_index = {}
            self.index_to_token = {}
            # 取出特殊字符的索引
            self.pad_index = self._add_token(self.pad_token)
            self.sos_index = self._add_token(self.sos_token)
            self.eos_index = self._add_token(self.eos_token)
            self.unk_index = self._add_token(self.unk_token)

        self.ignore_indices = {self.pad_index, self.sos_index, self.eos_index, self.unk_index}

    def _add_token(self, token: str) -> int:
        """Add one token to the vocabulary.

        Args:
            token: The token to be added.

        Returns:
            The index of the input token.
        """
        if token in self.token_to_index:
            return self.token_to_index[token]
        index = len(self)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
        return index

    def __len__(self):
        return len(self.token_to_index)

    def train(self, formulas: List[List[str]], min_count: int = 2) -> None:
        """Create a mapping from tokens to indices and vice versa.

        Args:
            formulas: Lists of tokens.
            min_count: Tokens that appear fewer than `min_count` will not be
                included in the mapping.
        """
        # 该方法执行的前提是 该类在初始化的时候 参数token_to_index=None, 否则就会self.index_to_token, self.token_to_index就会出问题
        if not self.token_to_index_init_full:
            # 统计所有样本中的每一个token的数量
            counter: Dict[str, int] = {}
            for formula in formulas:  # [['1','<','_'],['avg','|'.'2'],...,[]]
                for token in formula:
                    counter[token] = counter.get(token, 0) + 1

            for token, count in counter.items():
                # 删除数量少于min_count的token
                if count < min_count:
                    continue
                index = len(self)
                self.index_to_token[index] = token
                self.token_to_index[token] = index
        else:
            print(f'index_to_token or index_to_token is not empty........')

    def encode(self, formula: List[str]) -> List[int]:
        """
        获取一个样本的label中的tokens 对应的索引
        :param formula: 一个样本的标签内容,格式: ['1','<','_','dsds']
        :return: [0, ]
        """
        indices = [self.sos_index]  # 先加上 'sos' 头
        for token in formula:  # 送入的相当于 一个图像 送进来的标签, 暂时推断其格式: ['<', 'format1', 'dfc', 'd', 'e', 'f']
            index = self.token_to_index.get(token, self.unk_index)
            indices.append(index)
        indices.append(self.eos_index)  # 在内容后面跟上'eos'尾
        return indices  # [0,22334,112223,1234,7867,..,1]

    def decode(self, indices: List[int], inference: bool = True) -> List[str]:
        tokens = []
        for index in indices:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found an unknown index {index}")
            if index == self.eos_index:
                break
            if inference and index in self.ignore_indices:
                continue
            token = self.index_to_token[index]
            tokens.append(token)
        return tokens

    def save(self, filename: str):
        """Save token-to-index mapping to a json file."""
        with open(filename, "w") as f:
            json.dump(self.token_to_index, f)

    @classmethod
    def load(cls, filename: str) -> "Tokenizer":
        """Create a `Tokenizer` from a mapping file outputted by `save`.

        Args:
            filename: Path to the file to read from.

        Returns:
            A `Tokenizer` object.
        """
        with open(filename) as f:
            token_to_index = json.load(f)
        return cls(token_to_index)


def judge_invaild(formula: List, boxes: List) -> bool:
    need_box_num = len([i for i in formula if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'x', 'X', '-']])
    real_box_num = len(boxes)
    if need_box_num != real_box_num:
        return False
    return True


has_question_file_l = []


def get_all_formulas(filename: str) -> List[List[str]]:
    """Returns all the formulas in the formula file."""
    with open(filename, 'r', encoding='utf-8') as f:
        contents = f.readlines()
        # all_formulas = [formula.strip("\n").split() for formula in contents]  # 原始的
        all_formulas, all_bboxes = [], []
        for line_no, formula in enumerate(contents):
            real_formula, real_bbox = formula.split('\t')
            real_formula_l = real_formula.strip('\n').split()
            real_bbox_l = json.loads(real_bbox.strip("\n"))
            real_bbox_l = sorted(real_bbox_l,
                                 key=lambda x: (x[0] + x[2]) / 2)  # 按照x轴上的中心坐标x做升序排序 TODO: 先不考虑想分数场景下的box排列
            judge_pass = judge_invaild(real_formula_l, real_bbox_l)
            # assert judge_pass==True
            if judge_pass:  # 合法性校验通过
                all_formulas.append(real_formula_l)
                all_bboxes.append(real_bbox_l)
            else:
                has_question_file_l.append(line_no + 1)
    print(f'has_question_file_l:{has_question_file_l}')
    return all_formulas, all_bboxes


def get_split(
        all_formulas: List[List[str]],
        all_bboxes: List,
        filename: str,
) -> Tuple[List[str], List[List[str]], List]:
    image_names = []
    formulas = []
    bboxes = []
    has_bbox_str_l = list(map(str, range(0, 10))) + ['x', 'X', '-']
    with open(filename) as f:  # 对应的文件名为 im2latex_[train/test/val].lst ------ 其中对应的内容看下面for循环中的内容分解
        for line in f:
            formula_idx, img_name, _ = line.strip("\n").split()  # formula_idx image_name render_type\n
            image_names.append(img_name)
            try:
                this_formulas = all_formulas[int(formula_idx) - 1]  # 这张图片真实的标签 ['<', '1', '^', 'dy' '>', '2', '3']
                this_bboxes = all_bboxes[
                    int(formula_idx) - 1]  # 这张图像上要做bbox回归的boxes [[20,0, 22.0, 36.0, 41.0, '6'],[40,0, 22.0, 46.0, 41.0, '5'],[60,0, 22.0, 66.0, 41.0, 'x']]
                this_bboxes_tmp = []
                # TODO:这里给不需要bboxing的区域可以也加上[0,0,0,0,'null']的伪bboxing
                this_img_has_bbox_count = 0
                for this_str in this_formulas:
                    if this_str in has_bbox_str_l:
                        this_bboxes_tmp.append(this_bboxes[this_img_has_bbox_count])
                        this_img_has_bbox_count += 1
                    else:
                        this_bboxes_tmp.append([0, 0, 0, 0, 'null'])  # 表示此位置上是一个无效的bbox

                formulas.append(this_formulas)  # 真实的内容列表
                bboxes.append(this_bboxes_tmp)
            except Exception as e:
                print('=' * 10)
                print(f'find a err:{e}')
                print(f'formula_idx:{formula_idx}')
    return image_names, formulas, bboxes  # [img_name1, img_name2, ... img_namen], [label1, label2, ... labeln] 此时的label1还是一个字符串

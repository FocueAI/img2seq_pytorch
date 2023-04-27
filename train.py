import torch
import numpy as np
from create_dataloader import CreateDataloader
from create_model import Img2Seq, Accuracy, CharacterErrorRate
from utils.train_strategy.common_strategy import configure_optimizers
from toras.summary import summary
from toras.kerasmodel import KerasModel
from toras.kerascallbacks import TensorBoardCallback
from utils.metrics_zoo.char_metric import CharacterErrorRate
VIEW_MODEL_STRUCTURE = False
# step0: 读取配置文件
# TODO: 读取yaml配置文件,方便在灵活配置训练中的变量
# step1: 开启tensorboard记录器
tensorboard_record = TensorBoardCallback(save_dir='runs',model_name='img2seq', log_weight=True,log_weight_freq=5)

# step2: 创建训练/验证数据迭代器
dataloader_creator = CreateDataloader(batch_size=8,num_workers=0,pin_memory=False,dir_path='data_warehouse',
                                      train_dir_name='train_raw',val_dir_name='val_raw')
train_dataloader, val_dataloader = dataloader_creator.do()

# step3: 创建模型/损失函数/评价指标
img2seq_model, loss_fn, ignore_indices = Img2Seq(d_model=128, dim_feedforward=256, nhead=4,
                                          dropout=0.3, num_decoder_layers=3,
                                          max_output_len=150,
                                          vocab_path='data_warehouse/vocab.json'
                                          )
ignore_indices = torch.from_numpy(np.array(list(ignore_indices)))
# step4: 创建优化器和lr策略器
optimizer, scheduler = configure_optimizers(model=img2seq_model, init_lr=0.001, weight_decay=0.0001, milestones=[10], gamma=0.001)

# step5: 将模型封装成keras格式
model = KerasModel(img2seq_model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   lr_scheduler=scheduler,)
                   #metrics_dict={"acc": Accuracy()} ) # Accuracy
                   # metrics_dict={"acc": CharacterErrorRate(ignore_indices=ignore_indices)} ) # Accuracy
                   # metrics_dict={"acc": val_cer()}, ) # Accuracy
# step6: 是否像只管的查看下模型结构
if VIEW_MODEL_STRUCTURE:
    input_feature = torch.zeros(32, 1, 28, 28)
    summary(model, input_data=input_feature)

# step7: 模型训练
dfhistory = model.fit(train_data=train_dataloader,
                      val_data=val_dataloader,
                      epochs=20,
                      patience=3,
                      monitor="train_loss",
                      mode="max",
                      ckpt_path='checkpoint.pt',
                      plot=True,
                      quiet=False,
                      # callbacks=[tensorboard_record]
                      )
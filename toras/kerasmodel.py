import sys, datetime
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator


def colorful(obj, color="red", display_type="plain"):
    color_dict = {"black": "30", "red": "31", "green": "32", "yellow": "33",
                  "blue": "34", "purple": "35", "cyan": "36", "white": "37"}
    display_type_dict = {"plain": "0", "highlight": "1", "underline": "4",
                         "shine": "5", "inverse": "7", "invisible": "8"}
    s = str(obj)
    color_code = color_dict.get(color, "")
    display = display_type_dict.get(display_type, "")
    out = '\033[{};{}m'.format(display, color_code) + s + '\033[0m'
    return out


class StepRunner:  # 数据跑一步的逻辑
    def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator

    def __call__(self, batch):
        features, labels = batch

        # loss
        ##----------------- 原始的写法 --------------------##
        # preds = self.net(features)
        ##----------------- 修改后的写法 ------------------##
        preds = self.net(features, labels)
        # output = self.net.decode(labels,encoded_x)
        # preds = output.permute(1, 2, 0)
        ## ----------------------------------------------##
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # losses
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()  # 这里主要就是算准确率, 使用的很巧妙...
                        for name, metric_fn in self.metrics_dict.items()}  # self.metrics_dict.items() 像是操作普通字典一样...

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']  # 这里
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics  # 形式 {'train_loss':2.32777}, {'train_acc':0.15625, 'lr':0.001}


class EpochRunner:  # 数据跑一个epoch的逻辑(要调用数据跑一步的逻辑)
    def __init__(self, steprunner, quiet=False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()
        self.accelerator = self.steprunner.accelerator
        self.quiet = quiet

    def __call__(self, dataloader):
        loop = tqdm(enumerate(dataloader, start=1),
                    total=len(dataloader),
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100,
                    desc=f'{self.stage.upper()}-INFO==>'
                    )

        epoch_losses = {}
        for step, batch in loop:
            if self.stage == "train":
                step_losses, step_metrics = self.steprunner(batch)  # 其实,这个才是核心代码
            else:
                with torch.no_grad():
                    step_losses, step_metrics = self.steprunner(batch)
            # {'train_loss':2.32777}, {'train_acc':0.15625, 'lr':0.001}
            step_log = dict(step_losses,
                            **step_metrics)  # 把上述字典整合一起 {'train_loss':2.32777, 'train_acc':0.15625, 'lr':0.001 }
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v  # 把每一步的loss加起来, 也就是求出每一个epoch的损失

            if step != len(dataloader):
                loop.set_postfix(**step_log)  # 将训练过程中的log放置在tq进度条的后面展示出来
            else:  # 每一轮结束时候,
                epoch_metrics = step_metrics
                epoch_metrics.update({self.stage + "_" + name: metric_fn.compute().item()  # 这一轮总体的准确率
                                      for name, metric_fn in self.steprunner.metrics_dict.items()})
                epoch_losses = {k: v / step for k, v in epoch_losses.items()}  # 这一轮所有样本的损失平均值
                epoch_log = dict(epoch_losses, **epoch_metrics)
                loop.set_postfix(**epoch_log)
                for name, metric_fn in self.steprunner.metrics_dict.items():  # 将准确率积累的给清除掉
                    metric_fn.reset()
        return epoch_log


class KerasModel(torch.nn.Module):
    StepRunner, EpochRunner = StepRunner, EpochRunner

    def __init__(self, net, loss_fn, metrics_dict=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.net, self.loss_fn, self.metrics_dict = net, loss_fn, torch.nn.ModuleDict(metrics_dict)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=1e-3)
        self.lr_scheduler = lr_scheduler
        self.from_scratch = True

    def load_ckpt(self, ckpt_path='checkpoint.pt'):
        self.net.load_state_dict(torch.load(ckpt_path))
        self.from_scratch = False

    def forward(self, x):
        return self.net.forward(x)

    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min",
            mixed_precision='no', callbacks=None, plot=False, quiet=False):

        self.__dict__.update(locals())  # 该方法中的所有参数都变为对应类的属性, 该方法太隐蔽等,不建议在使用
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        device = str(self.accelerator.device)  # TODO: 在cpu环境中device='cpu', 下次尝试下在GPU环境中的状态．．．．．
        device_type = '🐌' if 'cpu' in device else '⚡️'
        self.accelerator.print(  # 这样可以避免在分布式训练中产生重复的输出...
            colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>"))

        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler)

        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)

        self.history = {}
        callbacks = callbacks if callbacks is not None else []

        if plot == True:
            from .kerascallbacks import VisProgress
            callbacks.append(VisProgress(self))

        self.callbacks = self.accelerator.prepare(callbacks)

        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_fit_start(model=self)  # TODO: 等赋值钩子函数之后在做研究

        start_epoch = 1 if self.from_scratch else 0  # TODO:  start_epoch = 或许配置中的开始值 if not self.from_scratch else 0 ======> 这样子更合理一点, 我的修改意见
        for epoch in range(start_epoch, epochs + 1):
            should_quiet = False if quiet == False else (quiet == True or epoch > quiet)

            if not should_quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n" + "==========" * 8 + "%s" % nowtime)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs) + "\n")

            # 1，train -------------------------------------------------
            train_step_runner = self.StepRunner(
                net=self.net,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),
                optimizer=self.optimizer if epoch > 0 else None,
                lr_scheduler=self.lr_scheduler if epoch > 0 else None
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, should_quiet)
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_train_epoch_end(model=self)

            # 2，validate -------------------------------------------------
            if val_dataloader:
                val_step_runner = self.StepRunner(
                    net=self.net,
                    loss_fn=self.loss_fn,
                    accelerator=self.accelerator,
                    stage="val",
                    metrics_dict=deepcopy(self.metrics_dict)
                )
                val_epoch_runner = self.EpochRunner(val_step_runner, should_quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]

                if self.accelerator.is_local_main_process:
                    for callback_obj in self.callbacks:
                        callback_obj.on_validation_epoch_end(model=self)

            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

            if best_score_idx == len(arr_scores) - 1:
                net_dict = self.accelerator.get_state_dict(self.net)
                self.accelerator.save(net_dict, ckpt_path)
                if not should_quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor, arr_scores[best_score_idx])))

            if len(arr_scores) - best_score_idx > patience:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>>"
                ).format(monitor, patience))
                break

        if self.accelerator.is_local_main_process:
            dfhistory = pd.DataFrame(self.history)
            self.accelerator.print(dfhistory)

            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model=self)

            self.net = self.accelerator.unwrap_model(self.net)
            self.net.load_state_dict(torch.load(ckpt_path))
            return dfhistory

    @torch.no_grad()
    def evaluate(self, val_data):
        accelerator = Accelerator()
        self.net, self.loss_fn, self.metrics_dict = accelerator.prepare(self.net, self.loss_fn, self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        val_step_runner = self.StepRunner(net=self.net, stage="val",
                                          loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                          accelerator=accelerator)
        val_epoch_runner = self.EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics

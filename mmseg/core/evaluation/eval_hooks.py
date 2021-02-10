import os.path as osp
import cv2

import numpy as np
import torch

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        #eval_res = self.dataloader.dataset.evaluate(
        #    results, logger=runner.logger, **self.eval_kwargs)

        # for name, val in eval_res.items():
        #    runner.log_buffer.output[name] = val

        images_to_visualize = []
        
        for i in range(len(self.dataloader.dataset)):
            if i < 12:

                img_info = self.dataloader.dataset.img_infos[i]

                img = cv2.imread(osp.join(self.dataloader.dataset.img_dir, img_info['filename']))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                gt_mask = cv2.imread(osp.join(self.dataloader.dataset.ann_dir, img_info['ann']['seg_map']))
                pred = results[i]

                diff = gt_mask.astype(np.float32)[:, :, 0] / 255. - pred[0, :, :]
                diff_colored = np.zeros(diff.shape + (3,))
                diff_colored[:, :, 0] = np.clip(diff, 0, 1)
                diff_colored[:, :, 1] = np.clip(-diff, 0, 1)
                if self.eval_kwargs.get("mask_norm_type", "by_max") == "by_max":
                    diff_colored = diff_colored / diff_colored.max()
                else:
                    mask_norm_min = self.eval_kwargs.get('mask_norm_min', 0.)
                    mask_norm_max = self.eval_kwargs.get('mask_norm_max', 1.)
                    diff_colored = diff_colored.clip(mask_norm_min, mask_norm_max)
                    diff_colored = (diff_colored - mask_norm_min) / (mask_norm_max - mask_norm_min)
                diff_colored = (diff_colored * 255.).astype(np.uint8)

                pred = (pred * 255.).astype(np.uint8)
                pred = np.repeat(pred.reshape(pred.shape[1], pred.shape[2], 1), 3, axis=2)

                images = np.stack([img, gt_mask, pred, diff_colored])
                img_batch = torch.from_numpy(images).permute(0, 3, 1, 2)
                images_to_visualize.append(img_batch)

        runner.log_buffer.output['images/val'] = images_to_visualize

        runner.log_buffer.ready = True

class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
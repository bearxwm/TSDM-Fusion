"""
-----------------------------------------------
A Simply Training Implementation of TeacherNet
--------------By XWM---------------------------
"""
import os
import yaml
import time
import torch
import argparse
import warnings
import pytorch_msssim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

from Train.DataLoader import TeacherData
from Model.TeacherNet import TeacherNet
from Model.loss_function import Gradloss

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    # ---------- Configuration ----------
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--sys_config', type=str, default='./sys_configs_teacher.yaml')
    args_cfg = parser.parse_args()
    FLAGS = yaml.safe_load(open(args_cfg.sys_config, 'r'))
    LOSS = {}
    epoch = 1

    print('-' * 35)
    print('-----------Teacher Net Train Start!!!---------------')

    # ---------- Dataloader ----------
    train_data = TeacherData(directory=FLAGS['Root']['data_dictionary_train'], FLAGS=FLAGS)
    train_loader = DataLoader(train_data, batch_size=FLAGS['Train']['train_batch_size'], shuffle=True, pin_memory=False,
                              persistent_workers=False, num_workers=4, drop_last=True)

    print('-' * 35)
    print('---Load Train_data Done!')
    print('   Train_data_num: %s' % len(train_data))

    # ---------- Loss ----------
    print('-' * 35)
    LOSS['pixel_criterion'] = torch.nn.L1Loss(reduction='mean').to(FLAGS['System_Parameters']['device'])
    LOSS['GradLoss'] = Gradloss(in_dim=1).to(FLAGS['System_Parameters']['device'])
    LOSS['SSIM_Teacher'] = pytorch_msssim.SSIM(channel=1, data_range=1.0).to(FLAGS['System_Parameters']['device'])
    print('---Build Loss Done!')

    # ---------- Model ----------
    print('-' * 35)
    netTeacher = TeacherNet(nf=32).to(FLAGS['System_Parameters']['device'])
    print('---Build NetWork Done!')

    # ---------- Optimizer ----------
    print('-' * 35)
    optimizerTeacher = torch.optim.AdamW(netTeacher.parameters(), lr=FLAGS['Train']['learning_rate_TeacherNet'])
    schedulerTeacher = torch.optim.lr_scheduler.StepLR(optimizerTeacher,
                                                       step_size=FLAGS['Train']['decay_step_TeacherNet'],
                                                       gamma=FLAGS['Train']['decay_rate_TeacherNet'])

    print('   TeacherNet_learning_rate:%s\n   TeacherNet_decay_step:%s\n   TeacherNet_decay_rate:%s' %
          (FLAGS['Train']['learning_rate_TeacherNet'], FLAGS['Train']['decay_step_TeacherNet'],
           FLAGS['Train']['decay_rate_TeacherNet']))
    print('---Build Optimizer Done!')

    # ---------- Train Loop ----------
    print('-' * 35)
    print('---Start Training')
    time.sleep(0.5)
    for epoch in range(1, FLAGS['Train']['epoch'] + 1):
        netTeacher.train()
        with tqdm(total=len(train_data), ncols=100, ascii=True) as t:

            for i, TRAIN_DATA in enumerate(train_loader, 0):
                train_ir = TRAIN_DATA['IR'].to(FLAGS['System_Parameters']['device'])
                train_vi_gray = TRAIN_DATA['VI_GRAY'].to(FLAGS['System_Parameters']['device'])

                netTeacher.zero_grad(set_to_none=True)
                train_pre = netTeacher(train_ir, train_vi_gray)

                gs_loss_ssim_ir = LOSS['SSIM_Teacher'](train_pre, train_ir) * 0.5
                gs_loss_ssim_vi = LOSS['SSIM_Teacher'](train_pre, train_vi_gray) * 0.5
                gs_loss_gc_ir = LOSS['GradLoss'](train_pre, train_ir) * 0.5
                gs_loss_gc_vi = LOSS['GradLoss'](train_pre, train_vi_gray) * 0.5
                gs_loss_pixel_criterion_ir = LOSS['pixel_criterion'](train_pre, train_ir) * 0.5
                gs_loss_pixel_criterion_vi = LOSS['pixel_criterion'](train_pre, train_vi_gray) * 0.5

                gs_loss_pixel_criterion = (gs_loss_pixel_criterion_ir + gs_loss_pixel_criterion_vi)
                gs_loss_ssim = (gs_loss_ssim_ir + gs_loss_ssim_vi)
                gs_loss_gc_ir = (gs_loss_gc_ir + gs_loss_gc_vi)

                gs_loss = 0.001 * gs_loss_ssim + gs_loss_gc_ir + gs_loss_pixel_criterion
                gs_loss.backward()
                optimizerTeacher.step()

                t.update(FLAGS['Train']['train_batch_size'])

            schedulerTeacher.step()
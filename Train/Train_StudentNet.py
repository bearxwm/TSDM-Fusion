"""
-----------------------------------------------
A Simply Training Implementation of StudentNet
--------------By XWM---------------------------
"""
import argparse
import os
import time
import warnings

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model.Discriminator import Discriminator
from Model.StudentNet.SwinIR import SwinIR
from Model.loss_function import ContentLoss, CharbonnierLoss
from Train.DataLoader import StudentData
from Train.toolbox import rgb_to_ycbcr, ycbcr_to_rgb

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    # ---------- Configuration ----------
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--sys_config', type=str, default='./sys_configs_student.yaml')
    args_cfg = parser.parse_args()
    FLAGS = yaml.safe_load(open(args_cfg.sys_config, 'r'))
    LOSS = {}
    epoch = 1

    print('-' * 35)
    print('-----------Teacher Net Train Start!!!---------------')

    # ---------- Dataloader ----------
    print('-' * 35)
    train_data = StudentData(directory=FLAGS['Root']['data_dictionary_train'], FLAGS=FLAGS)
    train_loader = DataLoader(train_data, batch_size=FLAGS['Train']['train_batch_size'], shuffle=True, pin_memory=True,
                              persistent_workers=True, num_workers=8, drop_last=True)
    print('---Load Train_data Done!')
    print('   Train_data_num: %s' % len(train_data))

    # ---------- Loss ----------
    print('-' * 35)
    LOSS['content_criterion'] = ContentLoss(
        feature_model_extractor_node="features.35",
        feature_model_normalize_mean=[0.485, 0.456, 0.406],
        feature_model_normalize_std=[0.229, 0.224, 0.225]).to(FLAGS['System_Parameters']['device'])
    LOSS['adversarial_criterion'] = torch.nn.BCEWithLogitsLoss().to(FLAGS['System_Parameters']['device'])
    LOSS['pixel_criterion'] = torch.nn.L1Loss(reduction='mean').to(FLAGS['System_Parameters']['device'])
    LOSS['CharbonnierLoss'] = CharbonnierLoss().to(FLAGS['System_Parameters']['device'])
    print('---Build Loss Done!')

    # ---------- Model ----------
    print('-' * 35)
    netStudent = SwinIR(out_chans=FLAGS['Model']['image_dim'],
                        img_size=FLAGS['Data']['image_size'],
                        window_size=8, img_range=1.,
                        embed_dim=FLAGS['Model']['embed_dim'],
                        depths=FLAGS['Model']['depths'],
                        num_heads=FLAGS['Model']['num_heads']).to(FLAGS['System_Parameters']['device'])

    netD = Discriminator(in_dim=FLAGS['Model']['image_dim']).to(FLAGS['System_Parameters']['device'])

    modelTeacher = torch.load(os.path.abspath('../Model/Model_TeacherNet_epoch_50.pth'))
    netTeacher = modelTeacher['model'].to(FLAGS['System_Parameters']['device'])
    netTeacher.load_state_dict(modelTeacher['state_dict'])
    netTeacher.eval()
    print('---Build NetWork Done!')

    # ---------- Optimizer ----------
    print('-' * 35)
    optimizerStudent = torch.optim.AdamW(netStudent.parameters(), lr=FLAGS['Train']['learning_rate_Student'])
    schedulerStudent = torch.optim.lr_scheduler.StepLR(optimizerStudent, step_size=FLAGS['Train']['decay_step_Student'],
                                                       gamma=FLAGS['Train']['decay_rate_Student'])

    print('   Student_learning_rate:%s\n   Student_decay_step:%s\n   Student_decay_rate:%s' %
          (FLAGS['Train']['learning_rate_Student'], FLAGS['Train']['decay_step_Student'],
           FLAGS['Train']['decay_rate_Student']))

    optimizerD = torch.optim.AdamW(netD.parameters(), lr=FLAGS['Train']['learning_rate_D'])
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=FLAGS['Train']['decay_step_D'],
                                                 gamma=FLAGS['Train']['decay_rate_D'])

    print('   D_learning_rate:%s\n   D_decay_step:%s\n   D_decay_rate:%s' %
          (FLAGS['Train']['learning_rate_D'], FLAGS['Train']['decay_step_D'], FLAGS['Train']['decay_rate_D']))

    print('---Build Optimizer Done!')

    # ---------- Train Loop ----------
    print('-' * 35)
    print('---Start Training')
    time.sleep(0.5)
    for epoch in range(1, FLAGS['Train']['epoch'] + 1):
        netTeacher.train()
        with tqdm(total=len(train_data), ncols=100, ascii=True) as t:

            for i, TRAIN_DATA in enumerate(train_loader, 0):
                train_hq_ir = TRAIN_DATA['image_HQ_IR'].to(FLAGS['System_Parameters']['device'])
                train_hq_vi = TRAIN_DATA['image_HQ_VI'].to(FLAGS['System_Parameters']['device'])

                train_lq_ir = TRAIN_DATA['image_LQ_IR'].to(FLAGS['System_Parameters']['device'])
                train_lq_vi = TRAIN_DATA['image_LQ_VI'].to(FLAGS['System_Parameters']['device'])

                Cy, Cb, Cr = rgb_to_ycbcr(train_hq_vi)
                Cy = netTeacher(train_hq_ir, Cy)
                train_gt = ycbcr_to_rgb(Cy, Cb, Cr)
                train_gt = train_gt.detach()

                if epoch >= FLAGS['Train']['GAN_start_epoch']:

                    for g_parameters in netStudent.parameters():
                        g_parameters.requires_grad = True

                    batch_size, _, height, width = train_gt.shape
                    real_label = torch.full([batch_size, 1], 1.0,
                                            dtype=train_gt.dtype, device=FLAGS['System_Parameters']['device'])
                    fake_label = torch.full([batch_size, 1], 0.0,
                                            dtype=train_gt.dtype, device=FLAGS['System_Parameters']['device'])

                    for d_parameters in netD.parameters():
                        d_parameters.requires_grad = True

                    netD.zero_grad()
                    netStudent_pre = netStudent(train_lq_ir, train_lq_vi)
                    hr_output = netD(train_gt)
                    sr_output = netD(netStudent_pre.detach().clone())
                    d_loss_real = LOSS['adversarial_criterion'](hr_output - torch.mean(sr_output), real_label)
                    d_loss_real.backward(retain_graph=True)

                    sr_output = netD(netStudent_pre.detach().clone())
                    d_loss_fake = LOSS['adversarial_criterion'](sr_output - torch.mean(hr_output), fake_label)
                    d_loss_fake.backward()
                    d_loss = d_loss_real + d_loss_fake
                    optimizerD.step()

                    for d_parameters in netD.parameters():
                        d_parameters.requires_grad = False

                    netStudent.zero_grad(set_to_none=True)
                    hr_output = netD(train_gt.detach().clone())
                    sr_output = netD(netStudent_pre)
                    pixel_loss = 10 * LOSS['CharbonnierLoss'](netStudent_pre, train_gt)

                    if FLAGS['Model']['image_dim'] == 1:
                        content_loss = 1 * LOSS['content_criterion'](
                            netStudent_pre.repeat(1, 3, 1, 1),
                            train_gt.repeat(1, 3, 1, 1)
                        )
                    else:
                        content_loss = 1 * LOSS['content_criterion'](netStudent_pre, train_gt)

                    d_loss_real = LOSS['adversarial_criterion'](hr_output - torch.mean(sr_output), real_label) * 0.5
                    d_loss_fake = LOSS['adversarial_criterion'](sr_output - torch.mean(hr_output), fake_label) * 0.5
                    adversarial_loss = 0.005 * (d_loss_real + d_loss_fake)
                    g_loss = pixel_loss + content_loss + adversarial_loss
                    g_loss.backward()
                    optimizerStudent.step()

                    for g_parameters in netStudent.parameters():
                        g_parameters.requires_grad = False

                    d_gt_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
                    d_Student_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

                else:

                    netStudent.zero_grad(set_to_none=True)
                    netStudent_pre = netStudent(train_lq_ir, train_lq_vi)
                    Student2Teacher_pixel_loss = 1 * LOSS['pixel_criterion'](netStudent_pre, train_gt)
                    total_loss = Student2Teacher_pixel_loss
                    total_loss.backward()
                    optimizerStudent.step()

                t.update(FLAGS['Train']['train_batch_size'])

        schedulerStudent.step()
        if epoch >= FLAGS['Train']['GAN_start_epoch']:
            schedulerD.step()
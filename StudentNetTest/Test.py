import argparse
import os
import time
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from StudentNetTest.TestData import TestMotherData
from StudentNetTest.testtoolbox import rgb_to_grayscale


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--sys_config', type=str,
                        default='./sys_configs_test.yaml')
    args_cfg = parser.parse_args()
    config_sys = yaml.safe_load(open(args_cfg.sys_config, 'r'))
    return config_sys


class TestRunner(object):
    def __init__(self, args):
        super(TestRunner, self).__init__()

        self.FLAGS = args

    def generate(self, net, test_data, dataset):
        Save_Path = os.path.abspath(os.path.join(self.FLAGS['Save_ROOT'], dataset))
        if not os.path.exists(Save_Path):
            os.makedirs(Save_Path)

        with torch.no_grad():
            with tqdm(total=len(test_data), ncols=80, ascii=True) as t:
                for i, TEST_DATA in enumerate(test_data):
                    t.set_description('|| Image %s' % (i + 1))

                    IR = TEST_DATA['IR'].to(self.FLAGS['device'])
                    VI = TEST_DATA['VI'].to(self.FLAGS['device'])
                    h = TEST_DATA['H']
                    w = TEST_DATA['W']

                    Image_name = TEST_DATA['Image_name']
                    outputs = net(IR, VI)
                    outputs = torch.clamp(outputs, 0, 1)
                    outputs = outputs[:, :, 0:h, 0:w]

                    if dataset == 'TNO':
                        outputs = rgb_to_grayscale(outputs)

                    outputs = ToPILImage()(torch.squeeze(outputs.data.cpu()))
                    final_save_path = os.path.join(Save_Path, Image_name[0] + '.png')
                    outputs.save(final_save_path, quality=100)

                    t.update(1)

    def gen_dataset(self, dataset):
        Data_Path = os.path.abspath(os.path.join(self.FLAGS['Data_Root'], dataset))
        data = TestMotherData(Data_Path, dataset)
        test_data = DataLoader(dataset=data, batch_size=1, pin_memory=True)
        return test_data

    def run_test(self):

        model = torch.load(os.path.abspath(self.FLAGS['pre_trained_model']))
        net = model['model'].to(self.FLAGS['device'])
        net.load_state_dict(model['state_dict'])
        net.eval()
        print('||---Load PreTrain Model Done')

        for dataset in self.FLAGS['dateset_list']:
            print('||', dataset)
            time.sleep(1)
            test_data = self.gen_dataset(dataset)
            self.generate(net, test_data, dataset)


TestRunner(parse_args()).run_test()

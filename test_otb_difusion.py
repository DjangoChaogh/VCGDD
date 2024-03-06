# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python
import os
import argparse, cv2, torch, json
import numpy as np
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists

from net import SiamRPNotb
from run_attack import SiamRPN_init, SiamRPN_track, SiamRPN_defense, SiamRPN_purify_clean, defense_adaptive_attack
from utils1 import rect_2_cxy_wh, cxy_wh_2_rect
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB100', help='datasets')
parser.add_argument('--sequence', dest='sequence', default=None, help='sequence')
parser.add_argument('-v', '--visualization', dest='visualization', default=False, action='store_true',
                    help='whether visualize result')
# new added 5.8
import yaml
os.sys.path.append('/data/Disk_D/shaochuan/Project/GuidedDiffusionPur-main')
from runners import *
parser.add_argument('--log', default='imgs', help='Output path, including images and logs')
parser.add_argument('--diffuison_config', type=str, default='ImageNet.yml',  help='Path for saving running related data.')  # cifar10_noguide  ImageNet
parser.add_argument('--seed', type=int, default=1234, help='Random seed')

parser.add_argument('--exp_mode', type=str, default='Full', help='Available: [Full, Partial, One]')
parser.add_argument('--runner', type=str, default='Tracker_denoiser', help='Available: [Empirical, Certified, Deploy]')  # Empirical  Demo
parser.add_argument('--purify_adv', dest='purify_adv', default=True, help='purify adv img or clean img')
model_name = 'test'
# Arguments not to be touched
parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
# new added end


def track_video(model, video, denoiser, new_config):
    image_save = 0
    toc, regions = 0, []

    # statistic the convergence of the loss 10.1
    video_loss = []
    # statistic the convergence of the loss end

    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            state, z_feat = SiamRPN_init(im, target_pos, target_sz, model, gpu_id=new_config.device.clf_device) # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(gt[f])
            att_per = 0  # adversarial perturbation in attack
            def_per = 0  # adversarial perturbation in defense
        elif f > 0:  # tracking
            if args.purify_adv:
                if f % 30 == 1:  # clean the perturbation from last frame
                    att_per = 0
                    def_per = 0
                    # state, att_per, def_per = SiamRPN_track(state, z_feat, im, f, regions[f-1], att_per, def_per, image_save, iter=10)
                    # statistic the convergence of the loss 10.1
                    state, att_per, def_per, attack_loss = SiamRPN_defense(state, z_feat, im, f, regions[f - 1],
                                                                           att_per, def_per,
                                                                           image_save, denoiser, new_config, iter=10)

                    # new added 12.2 adaptive attack eval
                    # state, att_per, def_per, attack_loss = defense_adaptive_attack(state, z_feat, im, f, regions[f - 1],
                    #                                                        att_per, def_per,
                    #                                                        image_save, denoiser, new_config, iter=10)
                    # new added end

                    video_loss.append(attack_loss)
                    # statistic the convergence of the loss end
                    location = cxy_wh_2_rect(state['target_pos'] + 1, state['target_sz'])
                    regions.append(location)
                else:
                    # state, att_per, def_per = SiamRPN_track(state, z_feat, im, f, regions[f-1], att_per, def_per, image_save, iter=5)
                    # statistic the convergence of the loss 10.1
                    state, att_per, def_per, attack_loss = SiamRPN_defense(state, z_feat, im, f, regions[f - 1],
                                                                           att_per,
                                                                           def_per,
                                                                           image_save, denoiser, new_config, iter=5)

                    # new added 12.2 adaptive attack eval
                    # state, att_per, def_per, attack_loss = defense_adaptive_attack(state, z_feat, im, f, regions[f - 1],
                    #                                                        att_per,
                    #                                                        def_per,
                    #                                                        image_save, denoiser, new_config, iter=5)
                    # new added end

                    video_loss.append(attack_loss)
                    # statistic the convergence of the loss end
                    location = cxy_wh_2_rect(state['target_pos'] + 1, state['target_sz'])
                    regions.append(location)
            else:
                state = SiamRPN_purify_clean(state, im, f, regions[f - 1], denoiser, new_config)
                location = cxy_wh_2_rect(state['target_pos'] + 1, state['target_sz'])
                regions.append(location)
        toc += cv2.getTickCount() - tic

        # statistic the convergence of the loss 10.1
        # if f == 1:
        #     return def_per
        # statistic the convergence of the loss end

        if args.visualization and f >= 0:  # visualization
            if f == 0: cv2.destroyAllWindows()
            if len(gt[f]) == 8:
                cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            else:
                cv2.rectangle(im, (int(gt[f, 0]), int(gt[f, 1])), (int(gt[f, 0]) + int(gt[f, 2]), int(gt[f, 1]) + int(gt[f, 3])), (0, 255, 0), 2)
            if len(location) == 8:
                cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 2)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 2)
            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(video['name'], im)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    video_path = join('test', 'defense', args.dataset, model_name)
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write(','.join([str(i) for i in x])+'\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f / toc))
    return f / toc


def load_dataset(dataset):
    base_path = '/data/Disk_A/datasets/tracking/dataset_OTB/'
    # base_path = '/data/Disk_A/datasets/tracking/dataset_UAV/UAV123/data_seq/UAV123/'
    # base_path = '/data/Disk_B/Shaochuan/datasets/dataset_LaSOT/'
    if not exists(base_path):
        print("Please download OTB dataset into `data` folder!")
        exit()
    json_path = '/data/Disk_A/datasets/tracking/dataset_OTB/OTB100.json'
    # json_path = '/data/Disk_A/datasets/tracking/dataset_UAV/UAV123/UAV123.json'
    # json_path = '/data/Disk_B/Shaochuan/datasets/dataset_LaSOT/LaSOT.json'
    info = json.load(open(json_path, 'r'))
    if args.dataset == 'LaSOT':
        f = open('/data/Disk_B/Shaochuan/datasets/dataset_LaSOT/testing_set.txt')
        test_set = f.read().splitlines()
        f.close()
        for v in list(info.keys()):
            if v in test_set:
                info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
                info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]  # our tracker is 0-index
                info[v]['name'] = v
            else:
                del info[v]
    else:
        for v in info.keys():
            # path_name = info[v]['name']
            # info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            # info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
            # info[v]['name'] = v
            info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]  # our tracker is 0-index
            info[v]['name'] = v


    return info


def main():
    global args, v_id
    args = parser.parse_args()

    # new added 5.8
    diffusion_project_path_ = '/data/Disk_D/shaochuan/Project/GuidedDiffusionPur-main'
    with open(os.path.join(diffusion_project_path_, 'configs', args.diffuison_config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)

    denoiser = eval(args.runner)(args, new_config)
    denoiser.deploy()
    # new added end

    net = SiamRPNotb()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNOTB.model')))
    #net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
    net.to(new_config.device.clf_device).eval()

    dataset = load_dataset(args.dataset)
    fps_list = []

    if args.sequence is not None:
        v_id=0
        video = args.sequence
        track_video(net, dataset[video], denoiser, new_config)
    else:
        for v_id, video in enumerate(dataset.keys()):
            if v_id > -1:
                fps_list.append(track_video(net, dataset[video], denoiser, new_config))
        print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))
        # print(np.mean(np.array(fps_list)))


if __name__ == '__main__':
    main()

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
from run_attack import SiamRPN_init, SiamRPN_track
from utils1 import rect_2_cxy_wh, cxy_wh_2_rect
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB100', help='datasets')
parser.add_argument('--sequence', dest='sequence', default=None, help='sequence')
parser.add_argument('-v', '--visualization', dest='visualization', default=False, action='store_true',
                    help='whether visualize result')


def track_video(model, video):
    image_save = 0
    toc, regions = 0, []

    # statistic the convergence of the loss 10.1
    video_loss = []
    imgs_resp_mean = []
    imgs_resp_var = []
    # statistic the convergence of the loss end

    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            state, z_feat = SiamRPN_init(im, target_pos, target_sz, model) # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(gt[f])
            att_per = 0  # adversarial perturbation in attack
            def_per = 0  # adversarial perturbation in defense
        elif f > 0:  # tracking
            if f % 30 == 1:  # clean the perturbation from last frame
                att_per = 0
                def_per = 0
                # state, att_per, def_per = SiamRPN_track(state, z_feat, im, f, regions[f-1], att_per, def_per, image_save, iter=10)
                # statistic the convergence of the loss 10.1
                state, att_per, def_per, attack_loss, img_resp_mean, img_resp_var = SiamRPN_track(state, z_feat, im, f, regions[f - 1], att_per, def_per,
                                                        image_save, iter=10)
                video_loss.append(attack_loss)
                imgs_resp_mean.append(img_resp_mean)
                imgs_resp_var.append(img_resp_var)
                # statistic the convergence of the loss end
                location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
                regions.append(location)
            else:
                # state, att_per, def_per = SiamRPN_track(state, z_feat, im, f, regions[f-1], att_per, def_per, image_save, iter=5)
                # statistic the convergence of the loss 10.1
                state, att_per, def_per, attack_loss, img_resp_mean, img_resp_var = SiamRPN_track(state, z_feat, im, f, regions[f - 1], att_per,
                                                                     def_per,
                                                                     image_save, iter=5)
                video_loss.append(attack_loss)
                imgs_resp_mean.append(img_resp_mean)
                imgs_resp_var.append(img_resp_var)
                # statistic the convergence of the loss end
                location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
                regions.append(location)
        toc += cv2.getTickCount() - tic

        # statistic the convergence of the loss 10.1
        # if f == 10:
        #     return video_loss
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
            # save img
            # main_path = '/data/Disk_A/shaochuan/Projects/RTAA-main/DaSiamRPN/failure_cases/Mhyang_ori/'
            # save_path = main_path + str(f) + '.jpg'
            # cv2.imwrite(save_path, im)
    toc /= cv2.getTickFrequency()

    # save result
    # video_loss = np.array(video_loss)
    # np.save(file=os.path.join(main_path, 'loss'), arr=video_loss)

    video_path = join('test', args.dataset, 'RTAA')
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write(','.join([str(i) for i in x])+'\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f / toc))

    return f / toc

    # new added 23.8.5  The mean and variance of the response
    # imgs_resp_mean = np.array(imgs_resp_mean)
    # imgs_resp_var = np.array(imgs_resp_var)
    # video_resp_mean = np.mean(imgs_resp_mean)
    # video_resp_var = np.mean(imgs_resp_var)
    # return video_resp_mean, video_resp_var
    # new added end


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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    global args, v_id
    args = parser.parse_args()

    net = SiamRPNotb()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNOTB.model')))
    #net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
    net.eval().cuda()

    dataset = load_dataset(args.dataset)
    fps_list = []

    if args.sequence is not None:
        video = args.sequence
        track_video(net, dataset[video])
    else:
        for v_id, video in enumerate(dataset.keys()):
            if v_id > -1:
                fps_list.append(track_video(net, dataset[video]))
        print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))

    # statistic the convergence of the loss 10.1
    # for v_id, video in enumerate(dataset.keys()):
    #     if v_id == 0:
    #         aver_loss_list = track_video(net, dataset[video])
    #     else:
    #         new_loss_list = track_video(net, dataset[video])
    #         for i in range(len(new_loss_list)):
    #             aver_loss_list[i] = np.mean([aver_loss_list[i], new_loss_list[i]], axis=0).tolist()
    # save_path = '/data/Disk_A/shaochuan/Projects/RTAA-main/DaSiamRPN/code/test/frame_transferability/ensemble'
    # mode = 'active_channel'  # afterbefore, var, mean, active_channel, response_attack
    # for i in range(len(aver_loss_list)):
    #     x_axis = np.arange(0, len(aver_loss_list[i]))
    #     plt.plot(x_axis, aver_loss_list[i])
    # plt.savefig(join(save_path, mode + '.jpg'))
    # print('5555')
    # statistic the convergence of the loss end

    # new added 23.8.5  The mean and variance of the response
    # dataset_resp_mean = []
    # dataset_resp_var = []
    # for v_id, video in enumerate(dataset.keys()):
    #     video_resp_mean, video_resp_var = track_video(net, dataset[video])
    #     print(video_resp_mean, video_resp_var)
    #     dataset_resp_mean.append(video_resp_mean)
    #     dataset_resp_var.append(video_resp_var)
    # overall_mean = np.mean(np.array(dataset_resp_mean))
    # overall_var = np.mean(np.array(dataset_resp_var))
    # print("dataset response mean and var:")
    # print(overall_mean, overall_var)
    # new added end


if __name__ == '__main__':
    main()

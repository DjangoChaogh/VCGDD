from torchvision import models
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
# from torchstat import stat

resnet = models.resnet50(pretrained=True)
# stat(resnet, (3, 224, 224))
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])

def resnet_forward(img_arr):
    img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')
    img_t = preprocess(img)

    cam_extractor = GradCAM(resnet)
    batch_t = torch.unsqueeze(img_t, 0)
    resnet.eval()
    out = resnet(batch_t)
    # activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    target = out.squeeze(0).topk(3)[-1][1].item()
    activation_map = cam_extractor(target, out)

    idx = out.squeeze().cpu().detach().numpy()
    # max_score = np.max(idx)
    top3_idx = idx.argsort()[-3:][::-1]
    top3_score = idx[top3_idx]
    return top3_idx, top3_score, activation_map

def vis(img_arr, name):
    # img_vis = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, img_arr)
    cv2.waitKey(1)

img = cv2.imread("/data/Disk_A/datasets/tracking/dataset_OTB/Biker/img/0001.jpg")
top_class, scores, activation_map = resnet_forward(img)

# CAM show
# plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

vis(img, 'basketball_game')
print('666')

# img = cv2.imread("/data/Disk_A/shaochuan/Projects/SimTrack-main/tracking_results/attack_vis/Basketball/active_channel.jpg")
# clean_patch = img[:, :img.shape[0], :]
# # vis(clean_patch, 'clean')
# adv_patch = img[:, img.shape[0]:img.shape[0]*2, :]
# # vis(adv_patch, 'adv')
# per_patch = img[:, img.shape[0]*2:img.shape[0]*3, :]
# # vis(per_patch, 'perturbation')
#
# clean_idx, clean_score = resnet_forward(clean_patch)
# adv_idx, adv_score = resnet_forward(adv_patch)
# per_idx, per_score = resnet_forward(per_patch)
# print('666')


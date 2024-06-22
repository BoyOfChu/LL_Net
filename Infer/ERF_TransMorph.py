import torch
import os
import glob
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch import optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageEnhance


from data import datasets, trans

def get_input_grad(model, x, y, central_point):
    x_in = torch.cat((x, y), dim=1)
    _, outputs = model(x_in)
    target = torch.zeros(outputs.shape, dtype=torch.float, requires_grad=True).cuda()
    target[:, :, central_point[0], central_point[1], central_point[2]] = 1.0
    target = torch.mul(outputs, target)
    central_point_map = torch.sum(target[:, :, central_point[0], central_point[1], central_point[2]],dim=1)
    grad = torch.autograd.grad(central_point_map, x)
    grad_map = grad[0]
    grad_map = grad_map.cpu().numpy().squeeze()
    return grad_map

def save_erf(ERF_map, model_name):
    sns.heatmap(data=ERF_map, cbar=False)
    plt.axis('off')
    plt.savefig(os.path.join('./Results', model_name+'.png'), dpi=200, bbox_inches='tight', pad_inches = -0.1)
    plt.close()

    # 对比度增强
    img = Image.open(os.path.join('./Results', model_name+'.png'))
    enh_bright = ImageEnhance.Brightness(img)
    brightness = 2
    img_contrasted = enh_bright.enhance(brightness)
    img_contrasted.save(os.path.join('./Results', model_name+'.png'))

def get_erf(test_path, atlas_path, model, model_path, model_name,img_shape):
    best_model = torch.load(model_path, map_location='cuda:0')['state_dict']
    model.load_state_dict(best_model)


    central_point = [img_shape[0]//2, img_shape[1]//2, img_shape[2]//2]

    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_path + '*.pkl'), atlas_path, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    ERF_map = np.zeros(img_shape, dtype=float)

    study_idx = 0
    model.cuda()
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    optimizer.zero_grad()
    for data in test_loader:
        data = [t.cuda() for t in data]
        x = data[0].float().cuda(non_blocking=True)
        y = data[1].float().cuda(non_blocking=True)
        x.requires_grad = True
        y.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, x, y, central_point)

        ERF_map += np.abs(contribution_scores)

        study_idx += 1
        print(study_idx)
    ERF_map = ERF_map/study_idx
    ERF_map = (ERF_map - ERF_map.min()) / (ERF_map.max() - ERF_map.min())
    ERF_map = np.squeeze(ERF_map[img_shape[0]//2,:, :])
    save_erf(ERF_map, model_name)

if __name__ == '__main__':
    atlas_path = '../../../Datasets/IXI_data/atlas.pkl'
    test_path = '../../../Datasets/IXI_data/Test/'
    img_shape = (160, 192, 224)

    from RD_LKA_Weiwei3090.Models_IXI.Infer.Models.TransMorph import TransMorph
    from RD_LKA_Weiwei3090.Models_IXI.Infer.Models.TransMorph import CONFIGS as CONFIGS_TM
    model_path = "Models/TransMorph_Validation_dsc0.744.pth.tar"
    model_name = "TransMorph"
    config = CONFIGS_TM['TransMorph']
    model = TransMorph(config)

    get_erf(test_path, atlas_path, model, model_path, model_name,  img_shape)

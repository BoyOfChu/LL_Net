import glob
import os, utils, utils_metrics
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from data import datasets, trans
from torchvision import transforms
import torch.nn.functional as nnf
from surface_distance import *
import time
import SimpleITK as sitk

# 加载目录
current_work_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_work_dir)


class SpatialTransform1(nn.Module):
    def __init__(self):
        super(SpatialTransform1, self).__init__()

    def forward(self, mov_image, flow, mod='bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]

        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode=mod, align_corners=True)

        return warped

class SpatialTransformer_VM(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

def dice_simple(pred, truth):
    smooth = 1e-5
    pred = pred.flatten().astype(int)
    truth = truth.flatten().astype(int)
    intersection = np.sum(pred * truth) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(truth) + smooth)
    return dice

def get_same_element_index(ob_list, word):
    return [i for (i, v) in enumerate(ob_list) if v == word]

def main(model, model_path, model_name):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = SpatialTransform1().cuda()
    transform_vm = SpatialTransformer_VM((160,192,224)).cuda()
    # diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    atlas_dir = '../../../Datasets/IXI_data/atlas.pkl'
    test_dir = '../../../Datasets/IXI_data/Test/'

    tar_labels = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    coor_name = ['Cerebral-White-Matter', 'Cerebral-Cortex', 'Lateral-Ventricle', 'Cerebellum-White-Matter',
                 'Cerebellum-Cortex', 'Thalamus', 'Caudate', 'Putamen', 'Pallidum', '3rd-Ventricle', '4th-Ventricle',
                 'Brain-Stem', 'Hippocampus', 'Amygdala', 'CSF', 'VentralDC', 'choroid-plexus', 'Cerebral-White-Matter',
                 'Cerebral-Cortex', 'Lateral-Ventricle', 'Cerebellum-White-Matter', 'Cerebellum-Cortex', 'Thalamus',
                 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala', 'VentralDC', 'choroid-plexus']
    tar_structs = ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter',
                   'Putamen', 'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
                   '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']

    if not os.path.exists('./Results/' + model_name):
        os.makedirs('./Results/' + model_name)
        save_path = './Results/' + model_name
    else:
        save_path = './Results/' + model_name

    with torch.no_grad():
        best_model = torch.load(model_path,map_location='cuda:0')#['state_dict']
        model.load_state_dict(best_model)
        model.cuda()

        test_composed = transforms.Compose([trans.Seg_norm(),
                                            trans.NumpyType((np.float32, np.int16)),
                                            ])
        test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)


        total_dice = []
        total_hd = []
        total_hd_95 = []
        total_time = []
        total_jac = []
        with torch.no_grad():
            stdy_idx = 0
            for data in test_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0].float().to(device)
                y = data[1].float().to(device)
                x_seg = data[2]
                y_seg = data[3]

                sum_time = 0
                start_moment = time.process_time_ns()
                D_f_xy = model(x, y)
                end_moment = time.process_time_ns()
                sum_time += (end_moment - start_moment)/1e+6

                with torch.no_grad():
                    dd, hh, ww = D_f_xy.shape[-3:]
                    flow = D_f_xy.clone()
                    flow[:, 0, :, :, :] = D_f_xy[:, 0, :, :, :] * dd / 2
                    flow[:, 1, :, :, :] = D_f_xy[:, 1, :, :, :] * hh / 2
                    flow[:, 2, :, :, :] = D_f_xy[:, 2, :, :, :] * ww / 2
                    flow = flow.cuda()

                start_moment = time.process_time_ns()
                warped_mov = transform_vm(x, flow)
                end_moment = time.process_time_ns()
                sum_time += (end_moment - start_moment) / 1e+6

                total_time.append(sum_time)
                # print(total_time[-1])

                x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
                x_seg_oh = torch.squeeze(x_seg_oh, 1)
                x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

                x_segs = []
                for i in range(46):
                    def_seg = transform(x_seg_oh[:, i:i + 1, ...].float().to(device), D_f_xy.permute(0, 2, 3, 4, 1))
                    x_segs.append(def_seg)
                x_segs = torch.cat(x_segs, dim=1)
                def_out = torch.argmax(x_segs, dim=1, keepdim=True)

                def_out = def_out.detach().cpu().numpy().squeeze()
                y_seg = y_seg.detach().cpu().numpy().squeeze()

                D_f_xy = D_f_xy.detach().cpu().numpy().squeeze()
                dd, hh, ww = D_f_xy.shape[-3:]
                D_f_xy[0, :, :, :] = D_f_xy[0, :, :, :] * dd / 2
                D_f_xy[1, :, :, :] = D_f_xy[1, :, :, :] * hh / 2
                D_f_xy[2, :, :, :] = D_f_xy[2, :, :, :] * ww / 2

                single_dice = []
                single_hd = []
                single_hd_95 = []
                single_jac = utils.jacobian_determinant_vxm(D_f_xy)
                for i in tar_labels:
                    if ((y_seg == i).sum() == 0) or ((def_out == i).sum() == 0):
                        print(i)
                        continue
                    else:
                        single_dice.append(dice_simple((def_out == i), (y_seg == i)))
                        single_hd.append(
                            compute_robust_hausdorff(compute_surface_distances((def_out == i), (y_seg == i), np.ones(3)),
                                                     100.))
                        single_hd_95.append(
                            compute_robust_hausdorff(compute_surface_distances((def_out == i), (y_seg == i), np.ones(3)),
                                                     95.))

                # 保存形变结果
                warped_mov = sitk.GetImageFromArray(warped_mov.detach().cpu().numpy().squeeze())
                warped_seg = sitk.GetImageFromArray(def_out.astype(int))
                D_f_xy = sitk.GetImageFromArray(D_f_xy.transpose((1, 2, 3, 0)))

                sitk.WriteImage(warped_mov, os.path.join(save_path, str(stdy_idx) + '_warped_img.nii.gz'))
                sitk.WriteImage(warped_seg, os.path.join(save_path, str(stdy_idx) + '_warped_mask.nii.gz'))
                sitk.WriteImage(D_f_xy, os.path.join(save_path, str(stdy_idx) + '_dvf.nii.gz'))

                total_dice.append(single_dice)
                total_hd.append(single_hd)
                total_hd_95.append(single_hd_95)
                total_jac.append(np.sum(single_jac <= 0)/(160*192*224))
                stdy_idx += 1
                print(str(stdy_idx) + '/115\t')

        total_dice = np.array(total_dice).transpose((1, 0))
        total_hd = np.array(total_hd).transpose((1, 0))
        total_hd_95 = np.array(total_hd_95).transpose((1, 0))
        total_time = np.array(total_time)
        total_jac = np.array(total_jac)
        np.save(os.path.join(save_path, model_name + '_time.npy'), total_time)
        print(np.mean(total_time))
        np.save(os.path.join(save_path, model_name + '_jac.npy'), total_jac)
        print(np.mean(total_jac))

        total_dice_new = np.zeros((len(tar_structs), total_dice.shape[1]), dtype=float)
        for i, struct in enumerate(tar_structs):
            indexs = get_same_element_index(coor_name, struct)
            line = np.zeros((1, total_dice.shape[1]), dtype=float)
            for j in indexs:
                line += total_dice[j, :]
            total_dice_new[i, :] = line.squeeze() / len(indexs)
        np.save(os.path.join(save_path, model_name +'_dice.npy'), total_dice_new)
        print(np.mean(total_dice_new))
        print(np.std(total_dice_new))

        total_hd_new = np.zeros((len(tar_structs), total_hd.shape[1]), dtype=float)
        for i, struct in enumerate(tar_structs):
            indexs = get_same_element_index(coor_name, struct)
            line = np.zeros((1, total_hd.shape[1]), dtype=float)
            for j in indexs:
                line += total_hd[j, :]
            total_hd_new[i, :] = line.squeeze() / len(indexs)
        np.save(os.path.join(save_path, model_name + '_hd.npy'), total_hd_new)
        print(np.mean(total_hd_new))
        print(np.std(total_hd_new))

        total_hd_95_new = np.zeros((len(tar_structs), total_hd_95.shape[1]), dtype=float)
        for i, struct in enumerate(tar_structs):
            indexs = get_same_element_index(coor_name, struct)
            line = np.zeros((1, total_hd_95.shape[1]), dtype=float)
            for j in indexs:
                line += total_hd_95[j, :]
            total_hd_95_new[i, :] = line.squeeze() / len(indexs)
        np.save(os.path.join(save_path, model_name + '_hd_95.npy'), total_hd_95_new)
        print(np.mean(total_hd_95_new))
        print(np.std(total_hd_95_new))


if __name__ == '__main__':
    from LL_Net import Net

    model_path = "Models/LL_Net_L7_S3_C16.pkl"
    model_name = "LLNet_L7_S3_C16"

    with torch.no_grad():
        model = Net(start_channels=16, large_kernel=7, small_kernel=3, in_channels=2, out_channels=3)
        main(model, model_path, model_name)
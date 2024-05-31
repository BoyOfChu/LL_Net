import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import csv, os, json, math, random, sys
from scipy.ndimage import zoom
isDebug = True if sys.gettrace() else False

'''将当前目录加入搜索路径'''
current_work_dir = os.path.dirname(__file__)
sys.path.append(current_work_dir)

def ImageNormliazation(image):
    image = (image-image.min())/(image.max()-image.min())
    return image

class Train_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.paths = data_path
        self.transforms = transform
        '''读取文件列表'''
        with open(os.path.join(data_path, 'File_List.json'), 'r') as f:
            file_list = json.load(f)
        self.file_list = file_list['Train']

    def __getitem__(self, index):
        file_name = self.file_list[index]
        '''读取图像和掩膜'''
        fixed_image = sitk.ReadImage(os.path.join(self.paths, 'Image', file_name))
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_image = sitk.ReadImage(os.path.join(self.paths, 'Image', 'atlas.nii.gz'))
        moving_image = sitk.GetArrayFromImage(moving_image)
        '''图像变换'''
        if self.transforms is not None:
            aug_data = self.transforms({
                'image': [moving_image, fixed_image],
            })
            moving_image = aug_data['image'][0]
            fixed_image = aug_data['image'][1]
            moving_image = ImageNormliazation(moving_image)
            fixed_image = ImageNormliazation(fixed_image)
        moving_image = torch.from_numpy(((np.ascontiguousarray(moving_image[None, ...])).astype(np.float32)))
        fixed_image = torch.from_numpy(((np.ascontiguousarray(fixed_image[None, ...])).astype(np.float32)))    
        return moving_image, fixed_image

    def __len__(self):
        return len(self.file_list)
        # return 30

class Val_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.paths = data_path
        self.transforms = transform
        '''读取文件列表'''
        with open(os.path.join(data_path, 'File_List.json'), 'r') as f:
            file_list = json.load(f)
        self.file_list = file_list['Val']

    def __getitem__(self, index):
        file_name = self.file_list[index]
        '''读取图像和掩膜'''
        fixed_image = sitk.ReadImage(os.path.join(self.paths, 'Image', file_name))
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_image = sitk.ReadImage(os.path.join(self.paths, 'Image', 'atlas.nii.gz'))
        moving_image = sitk.GetArrayFromImage(moving_image)

        fixed_lab = sitk.ReadImage(os.path.join(self.paths, 'Label', file_name))
        fixed_lab = sitk.GetArrayFromImage(fixed_lab)
        fixed_lab = Seg_norm(fixed_lab)
        moving_lab = sitk.ReadImage(os.path.join(self.paths, 'Label', 'atlas.nii.gz'))
        moving_lab = sitk.GetArrayFromImage(moving_lab)
        moving_lab = Seg_norm(moving_lab)
            
        moving_image = torch.from_numpy(((np.ascontiguousarray(moving_image[None, ...])).astype(np.float32)))
        fixed_image = torch.from_numpy(((np.ascontiguousarray(fixed_image[None, ...])).astype(np.float32)))  
        moving_lab = torch.from_numpy(((np.ascontiguousarray(moving_lab[None, ...])).astype(np.float32)))
        fixed_lab = torch.from_numpy(((np.ascontiguousarray(fixed_lab[None, ...])).astype(np.float32)))  
        return moving_image, fixed_image, moving_lab, fixed_lab
    
    def __len__(self):
        return len(self.file_list)
        # return 5

class Test_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.paths = data_path
        self.transforms = transform
        '''读取文件列表'''
        with open(os.path.join(data_path, 'File_List.json'), 'r') as f:
            file_list = json.load(f)
        self.file_list = file_list['Test']

    def __getitem__(self, index):
        file_name = self.file_list[index]
        '''读取图像和掩膜'''
        fixed_image = sitk.ReadImage(os.path.join(self.paths, 'Image', file_name))
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_image = sitk.ReadImage(os.path.join(self.paths, 'Image', 'atlas.nii.gz'))
        moving_image = sitk.GetArrayFromImage(moving_image)

        fixed_lab = sitk.ReadImage(os.path.join(self.paths, 'Label', file_name))
        fixed_lab = sitk.GetArrayFromImage(fixed_lab)
        fixed_lab = Seg_norm(fixed_lab)
        moving_lab = sitk.ReadImage(os.path.join(self.paths, 'Label', 'atlas.nii.gz'))
        moving_lab = sitk.GetArrayFromImage(moving_lab)
        moving_lab = Seg_norm(moving_lab)
        
        moving_image = torch.from_numpy(((np.ascontiguousarray(moving_image[None, ...])).astype(np.float32)))
        fixed_image = torch.from_numpy(((np.ascontiguousarray(fixed_image[None, ...])).astype(np.float32)))    
        moving_lab = torch.from_numpy(((np.ascontiguousarray(moving_lab[None, ...])).astype(np.float32)))
        fixed_lab = torch.from_numpy(((np.ascontiguousarray(fixed_lab[None, ...])).astype(np.float32)))  
        return moving_image, fixed_image, moving_lab, fixed_lab
    
    def __len__(self):
        return len(self.file_list)

def Seg_norm(img):
    seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
    img_out = np.zeros_like(img)
    for i in range(len(seg_table)):
        img_out[img == seg_table[i]] = i
    return img_out

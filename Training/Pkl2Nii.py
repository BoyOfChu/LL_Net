import pickle
import numpy as np
import os, glob
import SimpleITK as sitk

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

src_path = 'pkl文件路径'
tar_path = 'nii保存路径'

file_list = glob.glob(src_path + '\*.pkl')
for file in file_list:
    data = pkload(file)
    image = sitk.GetImageFromArray(data[0])
    label = sitk.GetImageFromArray(data[1])
    file_name = file.split('\\')[-1].split('.')[0] + '.nii.gz'
    sitk.WriteImage(image, os.path.join(tar_path, 'Image', file_name))

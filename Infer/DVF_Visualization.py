import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

dvf = sitk.ReadImage(r'I:\Temp\Results\TransMorph\1_dvf.nii.gz')
dvf = sitk.GetArrayFromImage(dvf)
shape = dvf.shape
for i in range(3):
    dvf[:,:,:,i] = 255*((dvf[:,:,:,i] -np.min(dvf[:,:,:,i] ))/(np.max(dvf[:,:,:,i])-np.min(dvf[:,:,:,i] )))
dvf = dvf.astype(int)

plt.imshow(dvf[80,...])
plt.show()

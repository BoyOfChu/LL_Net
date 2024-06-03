This is a PyTorch implementation of my paper: 
[A light-weight rectangular decomposition large kernel convolution network for deformable medical image registration](https://www.sciencedirect.com/science/article/pii/S1746809424005342)

# Training
使用以下命令进行训练，数据集为TransMorph提供的IXI，训练时把pkl格式改为了nii格式
```text
cd Training/Train
python Train.py --config_path LL_Net.yaml
```
关于训练超参的设置请直接修改yaml文件，如有问题欢迎讨论

# Infer
测试的时候，还是使用的pkl格式，直接运行infer_bilinear_LLNet.py即可

# Reference:
[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)

[LKU-Net](https://github.com/xi-jia/LKU-Net)

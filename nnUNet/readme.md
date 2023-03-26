## Efficient Multi-Organ Segmentation from 3D Abdominal CT Images with Lightweight Network and Knowledge Distillation
This repository provides the code for "Efficient Multi-Organ Segmentation from 3D Abdominal CT Images with Lightweight Network and Knowledge Distillation". 

## Usages
1. To obtain the pre-trained 3D nnUNet and LCOVNet models, run:
```
nnUNet_train 3d_fullres nnUNetTrainerV2_UNet -t x -f x -p lcov
nnUNet_train 3d_fullres nnUNetTrainerV2_LCOVNet -t x -f x -p lcov
```

2. When using the knowledge distillation model, load the pre-trained model and run:
```
nnUNet_train 3d_fullres nnUNetTrainerV2_Our -t x -f x -p lcov
```

3. For the prediction stage, use the following command:
```
nnUNet_predict
```


## Acknowledgement
The code is revised from [nnUNet][nnunet].

[nnunet]:https://github.com/MIC-DKFZ/nnUNet
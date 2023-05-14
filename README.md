## Efficient Multi-Organ Segmentation from 3D Abdominal CT Images with Lightweight Network and Knowledge Distillation
[tmi_link]:https://ieeexplore.ieee.org/document/10083150
[isbi_link]:https://ieeexplore.ieee.org/abstract/document/9434023
This repository provides the code for "Efficient Multi-Organ Segmentation from 3D Abdominal CT Images with Lightweight Network and Knowledge Distillation"(accepted by [TMI][tmi_link]) and "https://ieeexplore.ieee.org/abstract/document/9434023"(published by [ISBI][isbi_link]).


## Usages
1. To obtain the pre-trained 3D nnUNet and LCOVNet models, run:
```
pymic_train LCOVNet/config/locvnet.cfg
pymic_train UNet/config/unet.cfg
```

2. When using the knowledge distillation model, load the pre-trained model and run:
```
pymic_train KD/config/unet.cfg
```

## Testing and evaluation
1. Run the following command to obtain segmentation results of testing images based on the best-performing checkpoint on the validation set. By default we use sliding window inference to get better results. You can also edit the `testing` section of `kd/config/kd.cfg` to use other inference strategies.

```bash
pymic_test kd/config/kd.cfg
```

2. Use the following command to obtain quantitative evaluation results in terms of Dice. 

```bash
pymic_eval_seg kd/config/evaluation.cfg
```

## Acknowledgement
[PyMIC]:https://github.com/HiLab-git/PyMIC
The code is revised from [PyMIC][nnunet].


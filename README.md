# MonoPCC

This is a PyTorch implementation for **MonoPCC: Photometric-invariant Cycle Constraint for Monocular Depth Estimation of Endoscopic Images**.

## ⚙️ Setup

Our experiments are conducted in a [conda](https://www.anaconda.com/download) environment (python 3.7 is recommended) and you can use the below commands to install necessary dependencies:
```shell
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
pip3 install mmcv-full==1.3.0 mmsegmentation==0.11.0  
pip install timm einops IPython
```


## 💾 Data Preparation

The datasets in our experimental results are [SCARED](https://endovissub2019-scared.grand-challenge.org)(additional application to max.allan@intusurg.com is necessary), [SimCol3D](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-3d-reconstruction-during-colonoscopy-challenge) and [SERV-CT](https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct).

**SCARED split**

The train/test split for SCARED in our works is defined in the `splits/endovis` and further preprocessing is available in [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner).


## 📊 Evaluation on SCARED

To prepare the ground truth depth maps, please follow the [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner/blob/main/export_gt_depth.py). Moreover, here we provide the [model files](https://drive.google.com/drive/folders/13A9TZDETPgEm3D-c37YsGHn8OZcd-VMh?usp=sharing) to reproduce the reported results.

To evaluate model performance on SCARED, you need to run the following command: 
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --load_weights_folder <your_weight_path> \
    --eval_split endovis --dataset endovis  --max_depth 150 --png --eval_mono
```


## 🎞️ Demo of 3D reconstruction

Here we use MonoPCC to estimate depth maps for a video sequence in SCARED, and then perform 3D reconstruction ([ElasticFusion](https://github.com/mp3guy/ElasticFusion)) with the RGB and pseudo depth data:
![image](assets/fusion.gif)


## 🚗 Results of KITTI

To further demonstrate the effectiveness of MonoPCC in more general scenarios, we conduct comparison experiment on [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset. Here we exhibit SOTAs mentioned in our paper, and provide the [depth models]() to validate the below experimental results. 

| `Methods`          | Abs Rel| Sq Rel| RMSE| RMSE log|  $\delta$ < 1.25  |
|-----------------------|----|----|----|------|--------|
| [`Monodepth2`](https://github.com/nianticlabs/monodepth2)          | 0.115 |0.903 |4.863| 0.193 |0.877 |
| [`FeatDepth`](https://github.com/sconlyshootery/FeatDepth)   | 0.104| 0.729| 4.481| 0.179| 0.893|
| [`HR-Depth`](https://github.com/shawLyu/HR-Depth)         | 0.109|  0.792|  4.632|  0.185|  0.884|
| [`DIFFNet`](https://github.com/brandleyzhou/DIFFNet)  | 0.102 |0.749 |4.445 |0.179 |0.897 |
| [`MonoViT`](https://github.com/zxcqlf/monovit)         | 0.099 |0.708 |4.372| 0.175 |**0.900** |
| [`Lite-Mono`](https://github.com/noahzn/Lite-Mono)         | 0.101 |0.729| 4.454| 0.178| 0.897|
| `MonoPCC(Ours)`         | **0.098** |**0.677**| **4.318**| **0.173**| **0.900**|


## ⏳ To do

Currently, we have released the evaluation code and model weight files of MonoPCC, which can reproduce the result in our work. In the near future, we will continue to update the complete training code and model zoo. Also, we plan to evaluate our model on more challenging datasets, e.g., [nuScenes-Night](https://github.com/nutonomy/nuscenes-devkit).

## Acknowledgement
Thanks the authors for their works:

[MonoViT](https://github.com/zxcqlf/monovit)

[AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner)
# Directory 
## Root
The `${ROOT}` is described as below.  
```  
${ROOT} 
|-- assset 
|-- data  
|-- experiment
|-- lib
|-- main  
```  
* `data` contains required data and soft links to images and annotations directories.  
* `experiment` contains log, trained models, visualized outputs.  
* `lib` contains kernel codes for CycleAdapt.  
* `main` contains high-level codes for training or testing the network.  


## Required data
You need to follow directory structure of the `data` as below. 
```
${ROOT} 
|-- data  
|   |-- base_data
|   |   |-- human_models
|   |   |   |-- smpl
|   |   |   |   |-- SMPL_FEMALE.pkl
|   |   |   |   |-- SMPL_MALE.pkl
|   |   |   |   |-- SMPL_NEUTRAL.pkl
|   |   |   |-- J_regressor_h36m_smpl.npy
|   |   |   |-- smpl_mean_params.npz
|   |   |-- pose_prior
|   |   |   |-- gmm_08.pkl
|   |   |-- pretrained_models
|   |   |   |-- hmr_basemodel.pt
|   |   |   |-- md_basemodel.pt
|   |-- ...
```
* `base_data/human_models` contains `smpl` 3D model files. Download the SMPL model files from [[smpl](https://smpl.is.tue.mpg.de/)].
* `base_data/pretrained_models` contains HMRNet&MDNet checkpoints to be adapted.
* All files except `smpl` folder can be downloaded from [[base_data](https://drive.google.com/file/d/1ZxaM8VCB-3J5wbjUTPs1D77zPDcDnNYW/view?usp=drive_link)].


## Demo
To run CycleAdapt on a custom video, prepare parsed images and the annotation file as below. 
```  
${ROOT} 
|-- data  
|   |-- ...
|   |-- Demo
|   |   |-- images
|   |   |-- annotation.json
|   |-- ...
```  
* To obtain parsed images from video, you can utilize `python lib/utils/video2image.py --video {video_path}`.
* To construct annotation file, we recommend running AlphaPose [[codes]](https://github.com/MVIG-SJTU/AlphaPose.git) from the parsed images.
* We provide simple example, which can be downloaded from [[here]](https://drive.google.com/file/d/1hPBujCx2F4xC5_8iiorkGePo1r7ac425/view?usp=sharing)


## Dataset
```  
${ROOT} 
|-- data  
|   |-- ...
|   |-- PW3D
|   |   |-- imageFiles
|   |   |-- 3DPW_test.json
|   |-- ...
```  
* Download 3DPW parsed data [[data](https://virtualhumans.mpi-inf.mpg.de/3DPW/)] [[annot]](https://drive.google.com/drive/folders/1w3Y1lS1F9p-ybWbXW_wy9_nYOvqKq1R5?usp=sharing)


### Experiment
```  
${ROOT}  
|-- experiment
|-- |-- {exp_name}  
|   |   |-- checkpoints  
|   |   |-- log  
|   |   |-- vis  
```  
* Creating `experiment` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `checkpoints` folder contains saved checkpoints after adaptation.  
* `log` folder contains training log file.
* `vis` folder contains visualized results.  
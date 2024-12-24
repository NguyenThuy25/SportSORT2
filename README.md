# SportSORT

This is the official code for paper "SportsSORT: Overcoming Challenges of Multi-Object Tracking in Sports through Domain-Specific Features and Out of view Re-Association"

## Setup Instructions
- Clone this repo
- Install dependencies.
```
conda create -n sport_sort python=3.7
conda activate sport_sort
cd SportSORT/reid
pip install -r requirements.txt
pip install cython_bbox
python setup.py develop
cd ..
```


## Inference on SportsMOT dataset
### 1. Model preparation
Download the [Pose Detector]([url](https://drive.google.com/drive/folders/1W4SbuDpotv8r-ZMwIyaaF4scqwXCTedV?usp=sharing)) model and put it in the SportSORT/jersey_team/rtmpose/ckpt folder.

Download the weight of [PARSeq]([url](https://drive.google.com/drive/folders/1L5dYSFj_ARXHK0rx_qnDJmSs_RLjVzMG?usp=sharing)) model and put it in the SportSORT/jersey_team/models folder.

Download the [Detector and ReID](https://drive.google.com/drive/folders/19Ikrz0yu3CUeO4soyeRfKPmk27oo6252?usp=sharing) model and put them in the SportSORT/checkpoint folder.


### 2. Running Inference
- Online
```
cd SportSORT/SportSORT
python tools/infer.py --split train --online
```
- Offline (Using cache):
  Download the cache folder (include [detection and embedding]([url](https://drive.google.com/drive/folders/14gh9e5nQhqHsw77EfxZaUyn9NgPP0-Tq?usp=drive_link)), [team and jersey](https://drive.google.com/drive/folders/1guJ5jBCFYsZyM5CJELX7cUOA80CMwMpj?usp=sharing)) and put in the SportSORT/cache folder and run the bellow script to run SportSORT using cache data.
```
cd SportSORT/SportSORT
python tools/infer.py --split train
```

## Implementation Detail
Our main method is implemented in file "/SportSORT/tracker/SportSORT.py"

## Acknowledgement
The codebase is built highly upon [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) and [jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline), thanks for their wonderful work!

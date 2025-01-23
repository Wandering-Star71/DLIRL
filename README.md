# DLIRL: Data-light Inverse Reinforcement Learning
## setup
This repo is designed under python 3.9.x. Can work using cpu.
## Running
1. Downloading three datasets from following links:
    - [HighD](https://levelxdata.com/highd-dataset/)
    - [DJI](https://dronenr.com/2024/07/05/zyts-ad4che-dataset-powered-dji-drones/)
    - [NGSIM](https://datahub.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/)
2. Replace file path to your own path and run `collect_xxx.py`. This pre-generates transitions used for training.
3. Run `train.py`
4. Find trained model in output path. Use `load.py` to get successor_features. Specify successor features and another culture you want to deploy cross-culture in `train.py` and run it again.
5. Now you have 2 models: origin one and cross-cultural one.
## Overview
├ ── `collect_dji.py`  used to pre-generate DJI data   
├ ── `collect_highd.py` used to pre-generate HighD data  
├ ── `collect_ngsim.py` used to pre-generate NGSIM data  
├ ── `dlirl`  
│   ├ ── `agent.py` DLIRL agent  
│   ├ ── `buffer.py` training loop, load method  
│   ├ ── `loss.py` calculate loss  
│   ├ ── `networks.py` network structure  
│   ├ ── `part.py` some class definition  
│   ├ ── `util.py` some utils method and log   
│   ├ ── `data_management` methods used for pre-generat data  
├ ── `load.py `  
└ ── `train.py`  

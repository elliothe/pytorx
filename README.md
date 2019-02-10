  
  
<p align="center">
  <img src="./imgs/pytorx_logo3.jpeg" alt="Slate: API Documentation Generator" width="700">
  <br>
  <a href="https://travis-ci.org/lord/slate"><img src="https://travis-ci.org/lord/slate.svg?branch=master" alt="Build Status"></a>
</p>
  
<p align="center">PytorX helps you evaluate Neural Network performance on Crossbar Accelerator.</p>
  
  
  
  
  
Getting Started with PytorX
------------------------------
  
This project aims at building an easy-to-use framework for neural network mapping on crossbar-based accelerator with resistive memory (e.g., ReRAM, MRAM, etc.).
  
  
- [Dependencies](#Dependencies )
- [Usage](#Usage )
- [Results](#Results )
- [Methods](#Methods )
  
  
If you find this project useful to you, please cite [our work](https://arxiv.org/abs/1807.07948 ):
```
@inproceedings{He2019NIA,
  title={Noise Injection Adaption: End-to-End ReRAM Crossbar Non-ideal Effect Adaption for Neural Network Mapping},
  author={He, Zhezhi and Lin, Jie and Ewetz, Rickard and Yuan, Jiann-Shiun and Fan, Deliang},
  booktitle={Proceedings of the 56th Annual Design Automation Conference},
  pages={105},
  year={2019},
  organization={ACM}
}
```
## Dependencies:
  
  
* Python 3.6 (Anaconda)
* Pytorch 
* cuDNN 
  
The installation of pytorch environment could follow the steps in .... :+1:
  
## Usage
  
  
```bash
python main.py --
```
  
## Results
  
  
　Some experimental results are shown here
  
## Methods
  
  
## Task list
  
- [x] @mentions, #refs, [links]( ), **formatting**, and <del>tags</del> supported
- [x] list syntax required (any unordered or ordered list supported)
- [x] this is a complete item
- [ ] Evaluation of accelerator characteristics in terms of latency, power, etc.
  
  
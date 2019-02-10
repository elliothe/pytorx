---
markdown:
  image_dir: /assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false #是否使用绝对（相对于项目文件夹）图片路径
---

![PyTorch Logo](./imgs/pytorx_logo3.jpeg)<!-- .element height="50%" width="50%" -->

--------------------------------------------------------------------------------

# PytorX

This project aims at building an easy-to-use framework for neural network mapping on crossbar-based accelerator with resistive memory.

- [Dependencies](#Dependencies)
- [Usage](#Usage)
- [Results](#Results)
- [Methods](#Methods)


If you find this project useful to you, please cite [our work](https://arxiv.org/abs/1807.07948):
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

```bash {.line-numbers}
python main.py --
```

## Results

　Some experimental results are shown here

## Methods

## Task list
- [x] @mentions, #refs, [links](), **formatting**, and <del>tags</del> supported
- [x] list syntax required (any unordered or ordered list supported)
- [x] this is a complete item
- [ ] Evaluation of accelerator characteristics in terms of latency, power, etc.


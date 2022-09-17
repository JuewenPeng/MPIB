# MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects (ECCV 2022)

[Juewen Peng](https://scholar.google.com/citations?hl=en&user=fYC6lCUAAAAJ)<sup>1</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>
[Xianrui Luo](https://scholar.google.com/citations?hl=en&user=tUeWQ5AAAAAJ)<sup>1</sup>,
[Hao Lu](http://faculty.hust.edu.cn/LUHAO/en/index.htm)<sup>1</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>1*</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research

<p align="center">
<img src=https://user-images.githubusercontent.com/38718148/178954886-9c8bc5bf-fdb2-4bba-bbcc-2b77831884e8.jpg width="90%">
</p>

### [Project](https://juewenpeng.github.io/MPIB/) | [Paper](https://github.com/JuewenPeng/MPIB/blob/main/pdf/MPIB.pdf) | [Supp](https://github.com/JuewenPeng/MPIB/blob/main/pdf/MPIB-supp.pdf) | [Poster](https://github.com/JuewenPeng/MPIB/blob/main/pdf/MPIB-poster.pdf) | [Video](https://www.youtube.com/watch?v=xeSXrmVHBsk)

This repository is the official PyTorch implementation of the ECCV 2022 paper "MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects".


## Installation
```
git clone https://github.com/JuewenPeng/MPIB.git
cd MPIB
pip install -r requirements.txt
```


## Usage
Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/13t9M4s9CnAYrjArhrLfJtQAH80Cnn2kF?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1-VWY-3GWrii7Ip7rcK9v4Q?pwd=99x5), and put them in the directory `checkpoints`. Note that we use LaMa as our inpainting model, and `inpnet.pth` orignates from [https://github.com/saic-mdal/lama](https://github.com/saic-mdal/lama).

Run the following inference code.
```
python demo.py --image_path 'inputs/new_11_all-in-focus.jpg' --disp_path 'inputs/new_11_disparity.jpg' --save_dir 'outputs' --K 60 --disp_focus 207/255 --gamma 2.2
```
- `image_path`: path of the input all-in-focus image
- `disp_path`: path of the input disparity map (predicted by [DPT](https://github.com/isl-org/DPT) in this example)
- `save_dir`: directory to save the results
- `K`: blur parameter
- `disp_focus`: refocused disparity (range from 0 to 1)
- `gamma`: gamma value

See `demo.py` for more details.


## Citation
If you find our work useful in your research, please cite our paper.

```
@inproceedings{Peng2022MPIB,
  title = {MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects},
  author = {Peng, Juewen and Zhang, Jianming and Luo, Xianrui and Lu, Hao and Xian, Ke and Cao, Zhiguo},
  booktitle = {The Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```

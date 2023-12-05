## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, RTX 2080 GPUs

- Clone the repository 
```
git clone https://github.com/baranidharan-g23/mini_project && cd SiamMask
export SiamMask=$PWD
```
- Setup python environment
```
conda create -n siammask python=3.6
source activate siammask
pip install -r requirements.txt
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Run `final_tracking.py`

```shell
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/final_tracking.py --resume SiamMask_DAVIS.pth --config config_davis.json
```

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>
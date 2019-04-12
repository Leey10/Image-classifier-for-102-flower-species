# Image-classifier-for-102-flower-species
This is the final project for Facebook Pytorch Scholarship Challenge Phase 1. Data set containing images for 102 flower species is given with testing set and validation set divided. Task is to select appropriate CNN and try to train a model with classification accuracy as high as possible.
## Getting Started
This project is running on Google Colab, so there are some configurations needed before staring.
### Installation
First, Pytorch need to be installed on Colab, using the following commands. Pytorch version used here is 0.4.1, other versions can also be installed.
```
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
```

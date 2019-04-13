# Image-classifier-for-102-flower-species
This is the final project for Facebook Pytorch Scholarship Challenge Phase 1. Data set containing images for 102 flower species is given with testing set and validation set divided. Task is to select appropriate CNN and try to train a model with classification accuracy as high as possible.
## Getting Started
This project is running on Google Colab, so there are some configurations needed before staring.
### Installation
First, Pytorch needs to be installed on Colab, using the following commands. Pytorch version used here is 0.4.1, other versions can also be installed.
```
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
```
Install torchvision and Pillow :
```
!pip install torchvision
!pip install Pillow==5.3.0
```
In notebook settings, selec 'Python3' for 'Runtime type' and 'GPU' for 'Hardware accelerator'. Before starting the trainning, it's useful to double check the environment:
```
import PIL
import torch

print(PIL.PILLOW_VERSION)
print("PyTorch version: ")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is: ")
print(torch.backends.cudnn.version())

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
   print('CUDA is not available.  Training on CPU ...')
else:
   print('CUDA is available!  Training on GPU ...')
```
and the following results should show up :
```
5.3.0
PyTorch version: 
0.4.1
CUDA Version: 
9.2.148
cuDNN version is: 
7104

CUDA is available!  Training on GPU ...
```
### Mounting Google Drive
It's possible to download the trainning data every time, but a more practical way is to have data set downloaded and uploaded to Google drive for easy access. To do this, Colab has to be authorized for accessing google drive, use the following commands:
```
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!wget https://launchpad.net/~alessandro-strada/+archive/ubuntu/google-drive-ocamlfuse-beta/+build/15740102/+files/google-drive-ocamlfuse_0.7.1-0ubuntu3_amd64.deb
!dpkg -i google-drive-ocamlfuse_0.7.1-0ubuntu3_amd64.deb
!apt-get install -f
!apt-get -y install -qq fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```
There will be an URL generated, where an access code will be provided for the authorization. The process needs to be conducted twice for successfully authorization. The second step is to mount google drive:
```
from google.colab import drive
drive.mount("/content/drive")
```
and as a double check, list the files in the working directory:
```
!ls "/content/drive/My Drive/Colab Notebooks"
```
The trainning data folder 'flower_data' and the working notebook 'Pytorch_Challenge_Prj' can be seen:
```
cat_to_name.json	      GPU_StyleTransfer.ipynb
 flower_data		      images
 GPU_CIFAR10_densenet.ipynb  'Pytorch_Challenge_Prj(2).ipynb'
 GPU_CIFAR10_smallCNN.ipynb   Pytorch_Challenge_Prj.ipynb
 GPU_dog_cat.ipynb
 ```
 After these configuration steps, it's ready for the trainning with Pytorch!
## Data Set
The flower image folder can be downloaded from the URL:
```
https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
```
## Result
The data folder has train and valid sets, the final model achives 95% trainning accuracy and 96% valid accuracy.
## Project State
On going, trying different methods to increase valid accuracy >98%.





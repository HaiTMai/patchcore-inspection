import os
from enum import Enum

import PIL
import torch
import torchaudio
from torchvision import transforms
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from patchcore.utils import SaveImage

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [ 0.406]
IMAGENET_STD = [0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def make_features(waveform, sr=16000, mel_bins=128, target_length=1024):

    waveform = waveform - waveform.mean()
    waveform = waveform.unsqueeze(dim=0)    

    assert sr == 16000, 'input audio sampling rate must be 16kHz'
    
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

def audiodir(machine,id,Data = 'normal', base_dir = '/content/data/MIMII/'):#/content/drive/MyDrive/SADCL/Dataset/'
  '''
  Find the audio directory
  Inputs:
  machine: Name of the machine (valve/slider/fan/pump)
  id: ID of the machine (0,2,4,6)
  base_dir = Base directory of the dataset

  Outputs:
  dir = List of data adresses
  label = List of labels (0 -> normal, 1 -> abnormal)
  '''
  normaldir = base_dir + machine + '/id_' + str(format(id,'02d')) + '/normal'
  abnormaldir = base_dir + machine + '/id_' + str(format(id,'02d')) + '/abnormal'
  dir = []
  label = []
  if Data == 'normal':
    list = os.listdir(normaldir)
    for i in list:
      dir_address = normaldir + '/' + i
      dir.append(dir_address)
      label.append(0)

  else:
    list = os.listdir(abnormaldir)
    for i in list:
      dir_address = abnormaldir + '/' + i
      dir.append(dir_address)
      label.append(1)

  return dir,label
def train_test(subdataset):
  ds = subdataset.split('_')
  
  train_size = 0.75
  device = 'valve'
  id =0

  if len(ds) >0:
    device = ds[0]
    id= int(ds[1]) 

  dir,label = audiodir(device,id)  
  dir_abnormal,label_abnormal = audiodir(device,id,Data='abnormal')

  dataset_normal = MVTecDataset(classname='{}_{}'.format(device,id),data_dir=dir,labels=label)
  dataset_abnormal = MVTecDataset(classname='{}_{}'.format(device,id),data_dir=dir_abnormal,labels=label_abnormal)
  imagesize = dataset_normal.imagesize

  train_dataset, test_normal_dataset = torch.utils.data.random_split(dataset_normal, [int(len(dataset_normal)*train_size), len(dataset_normal)- int(len(dataset_normal)*train_size)])

  
  test_dataset = ConcatDataset([test_normal_dataset, dataset_abnormal])
  


  #Set again fro imagesize due to after split it lost the properties
  train_dataset.imagesize=imagesize
  test_dataset.imagesize=imagesize
  test_dataset.labels = [0]*len(test_normal_dataset)+[1]*len(dataset_abnormal)

  # Train = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=4,drop_last=True)
  # Test = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=True,num_workers=4)

#   Train = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=4,drop_last=True)
#   Test = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=True,num_workers=4)
  return train_dataset, test_dataset
   
class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """    
    def __init__(
        self,
        data_dir,
        labels,
        classname,
        resize=256,
        imagesize=224,      
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.data_dir = data_dir
        self.labels = labels
        self.classname = classname
        self.train_val_split = train_val_split

        self.resize = resize
        
        self.imagesize = (3, self.resize, 128)

    def __getitem__(self, idx):
        anomaly = self.labels[idx]
        path = self.data_dir[idx]
        x,sr = torchaudio.load(path)
        
        x = x.mean(axis=0)
        image = make_features(x)
        image=torch.transpose(image, 0,1)
    
        # To make it fit for all backbones that have RGB inputs
        image_3 = torch.zeros(3, image.shape[0], image.shape[1])
        image_3[0,:,:]= image
        image_3[1,:,:]= image
        image_3[2,:,:]= image

        image = image_3
        image = F.interpolate(image,size=(self.resize))
        image = torch.transpose(image, 1,2)

        if idx==0:
          print_image = image.cpu().detach().numpy()[0,:,:]
          print_image *= (255.0/print_image.max())
          SaveImage(print_image)
          print('===============>Save Image at idx =0')

        mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": self.classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            # "image_name": "/".join(image_path.split("/")[-4:]),
            # "image_path": image_path,
        }

    def __len__(self):
        return len(self.labels)



import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import cv2


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None, disable_normalization=False, use_uint16_conversion=False):
        self.frame = pd.read_csv(csv_file, header=None, sep=';')
        self.transform = transform
        self.disable_normalization = disable_normalization
        self.use_uint16_conversion = use_uint16_conversion


    def __getitem__(self, idx):
        image_name = self.frame.loc[idx, 0]
        depth_name = self.frame.loc[idx, 1]
        


        depth = cv2.imread(depth_name,-1)
        # Handle different depth image formats with optional normalization
        if not self.disable_normalization:
            # Apply original normalization (×1000)
            if depth.dtype == np.uint8:
                # For uint8 datasets convert to float first to avoid overflow
                depth = depth.astype(np.float32) * 1000
            elif depth.dtype == np.float32:
                # For float32 datasets multiply directly
                depth = depth * 1000
            else:
                # For other formats, convert to float32 and multiply
                depth = depth.astype(np.float32) * 1000
            
            # After normalization, choose final data type
            if self.use_uint16_conversion:
                # Original IM2ELEVATION approach: depth = (depth*1000).astype(np.uint16)
                depth = depth.astype(np.uint16)
            else:
                # Keep as float32 (upgraded approach)
                depth = depth.astype(np.float32)
        else:
            # Skip normalization - convert to float32 but keep original values
            depth = depth.astype(np.float32)
        #depth  = cv2.cvtColor(depth  , cv2.COLOR_BGR2GRAY)
        depth = Image.fromarray(depth)
        image = Image.open(image_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.frame)

def getTrainingData(batch_size=64, csv_data='', dataset_name=None, disable_normalization=False, use_uint16_conversion=False):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}




    csv = csv_data
    transformed_training_trans =  depthDataset(csv_file=csv,
                                        disable_normalization=disable_normalization,
                                        use_uint16_conversion=use_uint16_conversion,
                                        transform=transforms.Compose([
                                            PreprocessInput(max_size=440, dataset_name=dataset_name),  # New preprocessing step with dataset name
                                            #RandomHorizontalFlip(),
                                            CenterCrop([440, 440], [220, 220]),
                                            ToTensor(is_train=True, disable_normalization=disable_normalization),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))




   


    #x = ConcatDataset([transformed_training1,transformed_training2,transformed_training3,transformed_training4,transformed_training5,transformed_training_no_trans])
    #dataloader_training = DataLoader(x,batch_size,
                                      #shuffle=True, num_workers=4, pin_memory=False)
    dataloader_training = DataLoader(transformed_training_trans, batch_size, num_workers=4, pin_memory=False)

   
   

    return dataloader_training


def getTestingData(batch_size=3, csv='', dataset_name=None, disable_normalization=False, use_uint16_conversion=False):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
   

    # transformed_testing = depthDataset(csv_file='./data/building_test_meter_0.csv',
    #                                    transform=transforms.Compose([
    #                                        Scale(500),
    #                                        CenterCrop([400, 400],[400,400]),
    #                                        ToTensor(),
    #                                        Normalize(__imagenet_stats['mean'],
    #                                                  __imagenet_stats['std'])
    #                                    ]))

    csvfile = csv

    transformed_testing = depthDataset(csv_file=csvfile,
                                       disable_normalization=disable_normalization,
                                       use_uint16_conversion=use_uint16_conversion,
                                       transform=transforms.Compose([
                                           PreprocessInput(max_size=440, dataset_name=dataset_name),  # New preprocessing step with dataset name
                                           CenterCrop([440, 440],[440,440]),
                                           ToTensor(is_train=False, disable_normalization=disable_normalization),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=12, pin_memory=False)

    return dataloader_testing

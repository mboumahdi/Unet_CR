from torch.utils.data import Dataset
import numpy as np
import rasterio
from rasterio.enums import Resampling
import time
import torch
import torch.nn as nn
import timm
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm
import time
import os
import math
import random
from skimage.transform import resize
from scipy.ndimage import rotate

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

from sentinel import SentinelDataset
from unet import Sentinel12UNet
from unet import UNet
#from model.networks.deeplabv3plus import DeepLabV3Plus
#from model.losses.loss import FocalTverskyLoss
from datetime import datetime
#from data.asbestos import AsbestosDataset

from sklearn.model_selection import train_test_split
from tools.utils import save_checkpoint, load_checkpoint
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()


# set flags / seeds
SEED = 5
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

## function
def to_tensor(x, **kwargs):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))



class RandomCrop(object):
    def __init__(self, crop_size, p=0.5):
        self.crop_size = crop_size
        self.p = p

    def __call__(self, **data):
        image = data['image']
        mask = data['mask']

        if np.random.rand() <= self.p:
            h, w = image.shape[:2]
            new_h, new_w = self.crop_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h, left: left + new_w]
            mask = mask[top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}
    

class MyStructureMain:
    def __init__(self,lr, resume, data_dir, csv_file, path_to_checkpoint, path_to_save_models,epochs, batch_size, num_workers):
        
        self.lr = lr
        self.resume = resume
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.path_to_checkpoint = path_to_checkpoint
        self.path_to_save_models = path_to_save_models
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers



# Create an instance of the "structure"
opt = MyStructureMain(0.0002, False, '/mnt/nvme1/dataset_S2A', '/data/mboumahdi/codes/Tiles_T18_dataset_1.0_256_256.csv', 
                      '/mnt/nvme1/models_decloud/','/mnt/nvme1/models_decloud/', 50, 8,2)


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    df = pd.read_csv(opt.csv_file)
    df_train = df[df['set'] == 'train']
    df_valid = df[df['set'] == 'validation']
    df_test = df[df['set'] == 'test']
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    
    dataset_train = SentinelDataset(opt.data_dir, df_train['filename'], subset='train')  # You can adjust subset
    dataset_test  = SentinelDataset(opt.data_dir, df_test['filename'], subset = 'test')  # You can adjust subset
    dataset_valid = SentinelDataset(opt.data_dir, df_valid['filename'], subset='valid')  # You can adjust subset

    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                    drop_last=True, num_workers=opt.num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset_valid, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Sentinel12UNet().to(device)
    date = datetime.now().strftime('%Y-%m-%d')
    project_name = f'Sentinel12UNet_Tiles_T18_EarlyStop_Batch_8_{date}'
    
    log_dir = os.path.join(opt.path_to_save_models, project_name)

# Create the SummaryWriter with the specified log directory and filename_suffix
    writer = SummaryWriter(log_dir, filename_suffix=project_name)
    
    
    #writer = SummaryWriter(f'./runs/{project_name}', filename_suffix=project_name)
    s2_10 = torch.randn([1, 4, 64, 64]).to(device)
    s2_20 = torch.randn([1, 6, 64, 64]).to(device)
    s1_10 = torch.randn([1, 2, 64, 64]).to(device)
    dem_20 = torch.randn([1, 1, 32, 32]).to(device)
    writer.add_graph(model, input_to_model=(s2_10, s2_20, s1_10, dem_20))
    
    metric = MeanSquaredError().to(device)
    criterion = nn.L1Loss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
# load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 1
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)  # custom method for loading last checkpoint
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")

    # now we start the main loop
    n_iter = start_n_iter
    best_val_loss = np.inf
    patience = 5
    trigger_times = 0
    
    epochs = opt.epochs
    
    
    for epoch in range(start_epoch, opt.epochs + 1):
        n_iter = n_iter + 1
        # set models to train mode
        model.train()

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(dataloader_train),
                    total=len(dataloader_train))

        start_time = time.time()

        # for loop going through dataset
        running_train_loss = 0

        for i, data in pbar:
            s2_10_input = data['s2_10m_input'].to(device)
            s2_20_input = data['s2_20m_input'].to(device)
            s1_10 = data['s1_10m'].to(device)
            dem_20 = data['dem'].to(device)

            s2_10_output = data['s2_10m_output'].to(device)
            s2_20_output = data['s2_20m_output'].to(device)

            y = torch.cat([s2_10_output, s2_20_output], 1)

            # It's very good practice to keep track of preparation time and
            # computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()

            # forward and backward pass
            y_hat = model(s2_10_input, s2_20_input, s1_10, dem_20)

            loss = criterion(y_hat, y)  # TODO: check order parameters

            running_train_loss += loss.item()

            # backward
            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f'Train. Compute efficiency: {compute_efficiency:.2f}, '
                f'loss: {loss.item():.6f},  epoch: {epoch}/{epochs}')
            start_time = time.time()

        # update tensorboard
        writer.add_scalar("Loss/train", running_train_loss / len(dataloader_train), n_iter)

        model.eval()  # evaluation mode

        ###### Validation
        running_val_loss = 0.0
        running_val_acc = 0.0
        pbar = tqdm(enumerate(dataloader_val), total=len(dataloader_val))
        with torch.no_grad():
            for i, data in pbar:
                # data preparation
                s2_10_input = data['s2_10m_input'].to(device)
                s2_20_input = data['s2_20m_input'].to(device)
                s1_10 = data['s1_10m'].to(device)
                dem_20 = data['dem'].to(device)

                s2_10_output = data['s2_10m_output'].to(device)
                s2_20_output = data['s2_20m_output'].to(device)

                y = torch.cat([s2_10_output, s2_20_output], 1)

                y_hat = model(s2_10_input, s2_20_input, s1_10, dem_20)
                loss = criterion(y_hat, y)  # TODO: check order parameters

                acc_value = metric(y_hat, y)

                running_val_loss += loss.item()
                running_val_acc += acc_value

                pbar.set_description(
                    f'Validation. Loss: {loss.item():.6f}, accuracy: {acc_value:.6f}, epoch: {epoch}/{epochs}')

            writer.add_scalar("Loss/validation", running_val_loss / len(dataloader_val), n_iter)
            writer.add_scalar("Accuracy/validation", running_val_acc / len(dataloader_val), n_iter)
            '''
            if running_val_loss >= best_val_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    break
            elif running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                trigger_times = 0
                cpkt = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'n_iter': n_iter,
                    'optim': optimizer.state_dict(),
                }
            
                '''
            best_val_loss = running_val_loss
            trigger_times = 0
            cpkt = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'n_iter': n_iter,
                    'optim': optimizer.state_dict(),
                }
            if 1 == 1 :
             
                save_checkpoint(cpkt, os.path.join(opt.path_to_save_models, f'{project_name}.ckpt'))

                if epoch % 1 == 0:
                    pbar = tqdm(enumerate(dataloader_test), total=len(dataloader_test))

                    running_test_acc = 0.0
                    for i, data in pbar:
                        # data preparation
                        s2_10_input = data['s2_10m_input'].to(device)
                        s2_20_input = data['s2_20m_input'].to(device)
                        s1_10 = data['s1_10m'].to(device)
                        dem_20 = data['dem'].to(device)

                        s2_10_output = data['s2_10m_output'].to(device)
                        s2_20_output = data['s2_20m_output'].to(device)

                        y = torch.cat([s2_10_output, s2_20_output], 1)

                        y_hat = model(s2_10_input, s2_20_input, s1_10, dem_20)
                        loss = criterion(y_hat, y)  # TODO: check order parameters

                        acc_value = metric(y_hat, y)

                        running_test_acc += acc_value

                        pbar.set_description(
                            f'Test Accuracy: {acc_value:.6f},  epoch: {epoch}/{epochs}')

                    writer.add_scalar("Accuracy/test", running_test_acc / len(dataloader_test), n_iter)

                    num_figures = np.min((5, y_hat.shape[0]))
                    fig, axes = plt.subplots(nrows=num_figures, ncols=3, figsize=(12, int(3 * num_figures)), squeeze=False)
                    for idx in range(num_figures):
                        img_1 = np.rollaxis(np.squeeze(s2_10_input[idx][[2, 1, 0], :, :].cpu().numpy()), 0, 3)
                        axes[idx][0].imshow(np.clip(img_1, 0.0, 1.0))
                        axes[idx][0].set_axis_off()

                        img_2 = np.rollaxis(np.squeeze(10*s2_10_output[idx][[2, 1, 0], :, :].cpu().numpy()), 0, 3)
                        axes[idx][1].imshow(np.clip(img_2, 0.0, 1.0))
                        axes[idx][1].set_axis_off()

                        img_pred = np.rollaxis(np.squeeze(10*y_hat[idx][[2, 1, 0], :, :].cpu().numpy()), 0, 3)
                        axes[idx][2].imshow(np.clip(img_pred, 0.0, 1.0))
                        axes[idx][2].set_axis_off()
                    plt.tight_layout()

                    fig.canvas.draw()
                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    img = np.swapaxes(img, 0, 2)

                    writer.add_image(f'test_{n_iter:0>4}', img, n_iter)
                    plt.close(fig)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processing time: {elapsed_time:.4f} seconds")
    print(project_name)
    
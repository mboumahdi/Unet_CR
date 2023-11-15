from torch.utils.data import Dataset
import numpy as np
import rasterio
from rasterio.enums import Resampling
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

S2_UNSCALE_COEF = 10000.0
S2_SCALE_COEF = 1.0 / S2_UNSCALE_COEF
DEM_SCALE_COEF = 0.0001
S1_UNSCALE_COEF = 65535.0
S1_SCALE_COEF = 1 / S1_UNSCALE_COEF
CLOUD_SCALE_COEF =  0.001
ASPECT_SCALE_COEF = 1 / 360
SLOPE_SCALE_COEF = 1 / 90
HILLSHADE_SCALE_COEF = 1 / 255

def get_transforms(subset):
    # image: s2_10m_input, image0: s2_10m_output, image1: s2_20m_input, image2: s2_20m_output,
    # image3: s1_10m, image4: dem, image5:aspect, image6:slope, image7:hillshade
    if subset == 'train':
        transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ], additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image',
                               'image3': 'image', 'image4': 'image',
                               'image5': 'image', 'image6': 'image', 'image7': 'image', 'mask': 'image'}, is_check_shapes=False)
    else:
        transform = A.Compose([
            ToTensorV2()
        ], additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image',
                               'image3': 'image', 'image4': 'image',
                               'image5': 'image', 'image6': 'image', 'image7': 'image', 'mask': 'image'}, is_check_shapes=False)
    return transform


class SentinelDataset(Dataset):
    def __init__(self, filepath, files: np.array, subset='train') -> None:
        self.filepath = filepath
        self.samples = files
        self.transform = get_transforms(subset)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        #print(sample)
        # TODO: Check all
        s2_10m_cloudy = rasterio.open(f'{self.filepath}/{sample}_input_s2.tif').read()
        s2_10m_cloudy = s2_10m_cloudy.astype(np.float32) * S2_SCALE_COEF
        s2_10m_cloudy = np.rollaxis(s2_10m_cloudy, 0, 3)

        #print('s2_10m_cloudy', s2_10m_cloudy.shape)

        s2_10m_no_cloudy = rasterio.open(f'{self.filepath}/{sample}_target_s2.tif').read()
        s2_10m_no_cloudy = s2_10m_no_cloudy.astype(np.float32) * S2_SCALE_COEF
        s2_10m_no_cloudy = np.rollaxis(s2_10m_no_cloudy, 0, 3)

        #print('s2_10m_no_cloudy', s2_10m_no_cloudy.shape)

        s1_10m = rasterio.open(f'{self.filepath}/{sample}_input_s1.tif').read()
        s1_10m = s1_10m.astype(np.float32) * S1_SCALE_COEF
        s1_10m = np.rollaxis(s1_10m, 0, 3)

        #print('s1_10m', s1_10m.shape)

        s2_20m_cloudy_src = rasterio.open(f'{self.filepath}/{sample}_input_s2_20m.tif')
        s2_20m_cloudy = s2_20m_cloudy_src.read(
            out_shape=(
                s2_20m_cloudy_src.count,
                int(s2_20m_cloudy_src.height * 2),
                int(s2_20m_cloudy_src.width * 2)
            ),
            resampling=Resampling.nearest
        )
        s2_20m_cloudy = s2_20m_cloudy.astype(np.float32) * S2_SCALE_COEF
        s2_20m_cloudy = np.rollaxis(s2_20m_cloudy, 0, 3)

        #print('s2_20m_cloudy', s2_20m_cloudy.shape)

        s2_20m_no_cloudy_src = rasterio.open(f'{self.filepath}/{sample}_target_s2_20m.tif')
        s2_20m_no_cloudy = s2_20m_no_cloudy_src.read(
            out_shape=(
                s2_20m_no_cloudy_src.count,
                int(s2_20m_no_cloudy_src.height * 2),
                int(s2_20m_no_cloudy_src.width * 2)
            ),
            resampling=Resampling.nearest
        )
        s2_20m_no_cloudy = s2_20m_no_cloudy.astype(np.float32) * S2_SCALE_COEF
        s2_20m_no_cloudy = np.rollaxis(s2_20m_no_cloudy, 0, 3)

        #print('s2_20m_no_cloudy', s2_20m_no_cloudy.shape)
        '''
        dem_src = rasterio.open(f'{self.filepath}/{sample}_input_dem_20m.tif')
        dem = dem_src.read(
            out_shape=(
                dem_src.count,
                int(dem_src.height * 2),
                int(dem_src.width * 2)
            ),
            resampling=Resampling.nearest
        )
        
        dem = dem.astype(np.float32) * DEM_SCALE_COEF
        dem = np.rollaxis(dem, 0, 3)
        '''
        dem_src = rasterio.open(f'{self.filepath}/{sample}_input_dem_20m.tif')
        dem = dem_src.read()
        dem = dem.astype(np.float32) * DEM_SCALE_COEF
        dem = np.rollaxis(dem, 0, 3)
        
        
        aspect_src = rasterio.open(f'{self.filepath}/{sample}_input_aspect_20m.tif')
        aspect = aspect_src.read(
            out_shape=(
                aspect_src.count,
                int(aspect_src.height * 2),
                int(aspect_src.width * 2)
            ),
            resampling=Resampling.nearest
        )
        aspect = aspect.astype(np.float32) * ASPECT_SCALE_COEF
        aspect = np.rollaxis(aspect, 0, 3)
        #print('aspect', aspect.shape)
        
        slope_src = rasterio.open(f'{self.filepath}/{sample}_input_slope_20m.tif')
        slope = slope_src.read(
            out_shape=(
                slope_src.count,
                int(slope_src.height * 2),
                int(slope_src.width * 2)
            ),
            resampling=Resampling.nearest
        )
        slope = slope.astype(np.float32) * SLOPE_SCALE_COEF
        slope = np.rollaxis(slope, 0, 3)
        #print('slope', slope.shape)
        
        
        hillshade_src = rasterio.open(f'{self.filepath}/{sample}_input_hillshade_20m.tif')
        hillshade = hillshade_src.read(
            out_shape=(
                hillshade_src.count,
                int(hillshade_src.height * 2),
                int(hillshade_src.width * 2)
            ),
            resampling=Resampling.nearest
        )
        hillshade = hillshade.astype(np.float32) * ASPECT_SCALE_COEF
        hillshade = np.rollaxis(hillshade, 0, 3)
        #print('hillshade', hillshade.shape)

        cloud_mask = rasterio.open(f'{self.filepath}/{sample}_cloud_s2.tif').read()
        cloud_mask = cloud_mask.astype(np.float32) * CLOUD_SCALE_COEF
        cloud_mask = np.rollaxis(cloud_mask, 0, 3)

        #print('cloud_mask', cloud_mask.shape)

        if self.transform:
            data = self.transform(image=s2_10m_cloudy, image0=s2_10m_no_cloudy, image1=s2_20m_cloudy,
                                  image2=s2_20m_no_cloudy, image3=s1_10m, image4=dem,
                                  image5 = aspect, image6 = slope, image7 = hillshade, mask=cloud_mask)
        else:
            data = {'image': s2_10m_cloudy, 'image0': s2_10m_no_cloudy, 'image1': s2_20m_cloudy,
                    'image2': s2_20m_no_cloudy, 'image3': s1_10m, 'image4': dem, 
                    'image5': aspect, 'image6': slope, 'image7': 'hillshade', 'mask': cloud_mask}

        return {'s2_10m_input': data['image'], 's2_10m_output': data['image0'], 's2_20m_input': data['image1'],
                's2_20m_output': data['image2'], 's1_10m': data['image3'], 'dem': data['image4'],
                'aspect':data['image5'], 'slope':data['image6'], 'hillshade':data['image7'],
                'cloud_mask': data['mask']}

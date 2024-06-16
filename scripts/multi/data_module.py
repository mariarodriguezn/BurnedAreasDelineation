import os
import pandas as pd
import numpy as np
import torch
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg

class BurnedAreasDataset(Dataset):
    def __init__(self, dataset_metadata, data_dir, stats, additional_features=None, transform=None):
        self.dataset_metadata = dataset_metadata
        self.data_dir = data_dir
        self.stats = stats
        self.additional_features = additional_features
        self.transform = transform
        self.lc_label_mapping = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 95: 10, 100: 11}


    def __len__(self):
        return len(self.dataset_metadata)

    def normalize_data(self, data, feature_name):
        if feature_name in self.stats:
            feature_stats = self.stats[feature_name]
            min_val = np.array(feature_stats['min'])
            max_val = np.array(feature_stats['max'])
            if data.ndim == 3: 
                min_val = min_val.reshape((-1, 1, 1))
                max_val = max_val.reshape((-1, 1, 1))
            normalized_data = (data - min_val) / (max_val - min_val + 1e-10)
            return normalized_data
        else:
            raise ValueError(f"{feature_name} not supported")
    
    # Distance Function taken from https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py#L260    
    def label2dm(self, label_tensor: torch.Tensor) -> torch.Tensor:
        label_np = label_tensor.cpu().numpy().astype(bool)[0]
        posmask = label_np
        negmask = ~posmask
        if posmask.any() and negmask.any():
            posdis = distance_transform_edt(posmask)
            negdis = distance_transform_edt(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            dm = negdis - posdis
            dm[boundary == 1] = 0
        else:
            dm = np.zeros_like(label_np, dtype=np.float32)
        dm_tensor = torch.from_numpy(dm).to(label_tensor.device).type(torch.float32).unsqueeze(0)
        return dm_tensor

    def __getitem__(self, idx):
        row = self.dataset_metadata.iloc[idx]
        activation_id = row['activation_id']
        patch_id = row['patch_id']

        # Load and normalize spectral bands
        data_path = os.path.join(self.data_dir, activation_id, 'data', f'{patch_id}.TIF')
        with rasterio.open(data_path) as data_file:
            spectral_bands = data_file.read([1, 2, 3, 4]).astype(np.float32)
            norm_spectral_bands = self.normalize_data(spectral_bands, 'spectral_bands')
        
        data_arrays = [norm_spectral_bands]
        
        # Load and normalize additional features
        if self.additional_features:
            red, green, nir = spectral_bands[0], spectral_bands[1], spectral_bands[3]

            if 'NDVI' in self.additional_features:
                ndvi = (nir - red) / (nir + red + 1e-10)  # Avoid division by zero
                norm_ndvi = self.normalize_data(np.expand_dims(ndvi, axis=0), 'NDVI')
                data_arrays.append(norm_ndvi)
                
            if 'BAI' in self.additional_features:
                bai = 1 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2 + 1e-10)
                norm_bai = self.normalize_data(np.expand_dims(bai, axis=0), 'BAI')
                data_arrays.append(norm_bai)
                
            if 'NDWI' in self.additional_features:
                ndwi = (green - nir) / (green + nir + 1e-10)
                norm_ndwi = self.normalize_data(np.expand_dims(ndwi, axis=0), 'NDWI')
                data_arrays.append(norm_ndwi)
        
        # Combine all arrays into a single normalized array
        data_array = np.concatenate(data_arrays, axis=0).transpose(1, 2, 0)

        # Load labels Main - Burned Areas
        label_path = os.path.join(self.data_dir, activation_id, 'labels', f'{patch_id}.TIF')
        with rasterio.open(label_path) as label_file:
            label_array = label_file.read(1).astype(np.uint8)
            
        # Load Land Cover
        lc_path = os.path.join(self.data_dir, activation_id, 'lc', f'{patch_id}.TIF')
        with rasterio.open(lc_path) as lc_file:
            lc_array = lc_file.read(1).astype(np.uint8)
            lc_array = np.vectorize(self.lc_label_mapping.get)(lc_array)        
            
        # Load cloud mask 1: no clouds 0: clouds
        cm_path = os.path.join(self.data_dir, activation_id, 'cm', f'{patch_id}.TIF')        
        if os.path.exists(cm_path):
            with rasterio.open(cm_path) as cm_file:
                cm_array = cm_file.read(1).astype(np.uint8)
        else:
            cm_array = np.ones_like(label_array, dtype=np.uint8)
                
        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=data_array, mask=label_array, cm=cm_array, lc=lc_array)
            data_tensor = transformed["image"]
            label_tensor = transformed["mask"].unsqueeze(0)
            cm_tensor = transformed["cm"].unsqueeze(0)
            lc_tensor = transformed["lc"].unsqueeze(0)
        
        # Load Cloud Mask as a channel if required         
        if 'CM' in self.additional_features:
            data_tensor = torch.cat((data_tensor, cm_tensor), dim=0)
            
        # Calculate distance maps for the labels
        dm_tensor = self.label2dm(label_tensor)
                
        return data_tensor, label_tensor, patch_id, cm_tensor, dm_tensor, lc_tensor

class BurnedAreasDataModule(pl.LightningDataModule):
    def __init__(self, data_root_dir, batch_size=8, additional_features=None, stats=None, random_state=None):
        super().__init__()

        self.data_root_dir = data_root_dir
        self.batch_size = batch_size
        self.additional_features = additional_features
        self.stats = stats
        self.random_state = random_state
        self.train_metadata = self.val_metadata = self.test_metadata = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def prepare_data(self):
        # Load patches metadata
        patches_metadata_csv = os.path.join(self.data_root_dir, 'metadata', 'patches_metadata.csv')
        df_p = pd.read_csv(patches_metadata_csv)  
        # Filter metadata   
        filtered_df_p = df_p[(df_p['height_p'] == 512) & (df_p['width_p'] == 512) & (df_p['perc_nodata_p'] == 0)]
        # Load the fishnet metadata
        fishnet_metadata_csv = os.path.join(self.data_root_dir, 'metadata', 'fishnet_metadata.csv')
        df_f = pd.read_csv(fishnet_metadata_csv)
        filtered_df_f = df_f[df_f['no_patches'] > 1]
        
        # Split train, validation and test metadata
        bin_count = 20
        bin_numbers = pd.qcut(x=filtered_df_f['perc_burned_f'], q=bin_count, labels=False, duplicates='drop')
        train_cells, temp_cells = train_test_split(filtered_df_f, test_size=0.2, random_state=self.random_state, stratify= bin_numbers)
        bin_numbers_temp = pd.qcut(x=temp_cells['perc_burned_f'], q=bin_count, labels=False, duplicates='drop')  
        val_cells, test_cells = train_test_split(temp_cells, test_size=0.5, random_state=self.random_state, stratify= bin_numbers_temp)
        
        filtered_df_f['split'] = 'train'
        filtered_df_f.loc[val_cells.index, 'split'] = 'val'
        filtered_df_f.loc[test_cells.index, 'split'] = 'test'

        merged_df = pd.merge(filtered_df_p, filtered_df_f[['fishnet_id', 'split']], on='fishnet_id', how='left')
        
        self.train_metadata = merged_df[merged_df['split'] == 'train']
        self.val_metadata = merged_df[merged_df['split'] == 'val']
        self.test_metadata = merged_df[merged_df['split'] == 'test']

        # Print metadata length and shape
        print("Length Train Metadata", len(self.train_metadata))
        print("Length Val Metadata", len(self.val_metadata))
        print("Length Test Metadata", len(self.test_metadata))

    def setup(self, stage=None):
        # Load stats
        filtered_means, filtered_stds = self._get_stats()   
        if stage in (None, 'fit', 'validate'):
            self.train_dataset = BurnedAreasDataset(self.train_metadata, 
                                                    self.data_root_dir, 
                                                    stats=self.stats,
                                                    additional_features=self.additional_features, 
                                                    transform=self._get_transform('train', filtered_means, filtered_stds))
            self.val_dataset = BurnedAreasDataset(self.val_metadata, 
                                                  self.data_root_dir, 
                                                  stats=self.stats,
                                                  additional_features=self.additional_features,
                                                  transform=self._get_transform('val', filtered_means, filtered_stds))
        if stage in (None, 'test'):
            self.test_dataset = BurnedAreasDataset(self.test_metadata, 
                                                   self.data_root_dir,
                                                   stats=self.stats,
                                                   additional_features=self.additional_features, 
                                                   transform=self._get_transform('test', filtered_means, filtered_stds))
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def _get_stats(self):
        filtered_means = self.stats['spectral_bands']['mean'][:]
        filtered_stds = self.stats['spectral_bands']['std'][:]  
        if self.additional_features:
            for feature in self.additional_features:
                if feature in self.stats:
                    filtered_means.append(self.stats[feature]['mean'])
                    filtered_stds.append(self.stats[feature]['std'])  
                elif feature == 'CM':
                    continue
                else:
                    raise ValueError(f"{feature} not supported")      
        return filtered_means, filtered_stds
    
    def _get_transform(self, phase, filtered_means, filtered_stds):
        if phase == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=filtered_means, std=filtered_stds, max_pixel_value=1.0),
                ToTensorV2()
            ], additional_targets={'cm': 'mask', 'lc':'mask'}) 
        else:
            return A.Compose([
                A.Normalize(mean=filtered_means, std=filtered_stds, max_pixel_value=1.0),
                ToTensorV2()
            ], additional_targets={'cm': 'mask', 'lc':'mask'})
import os
import time
from datetime import datetime
import shutil
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as MplNormalize
import rasterio
from rasterio.transform import from_origin
import cv2
import segmentation_models_pytorch as smp
import ttach as tta
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from custom_losses import DL_BL

class LightningSegmentation(pl.LightningModule):
    def __init__(self, 
                 batch_size, 
                 num_epochs,
                 model_name, 
                 optimizer_name, 
                 scheduler_name, 
                 loss_func_name,
                 data_root_dir, 
                 additional_features, 
                 all_stats,
                 use_tta,
                 outputs_dir,
                 export_preds,
                 model_kwargs={}, 
                 optimizer_kwargs={},
                 scheduler_kwargs={}, 
                 **loss_function_kwargs):
        
        super().__init__()
        self.additional_features = additional_features
        
        self.input_channels = 4 + len(self.additional_features) if self.additional_features else 4
        self.num_classes = 1   
        
        self.batch_size = batch_size
        self.model = self._get_model(model_name, **model_kwargs)
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs        
        self.loss_func = self._get_loss_function(loss_func_name, **loss_function_kwargs)
        self.use_tta = use_tta
        self.data_root_dir = data_root_dir
        self.outputs_dir=outputs_dir
        self.export_preds=export_preds
        self.example_input_array = torch.Tensor(batch_size,  self.input_channels, 512, 512)
        self.alpha = 0.01
        
        self.save_hyperparameters()
    
    # Forward Pass
    def forward(self, x):
        outputs = self.model(x)
        if isinstance(self.model, SegformerForSemanticSegmentation):
            outputs = F.interpolate(outputs.logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return outputs
    
    # Set Up Optmizer
    def configure_optimizers(self):
        optimizer = self._get_optimizer(self.optimizer_name, self.parameters(), **self.optimizer_kwargs)
        
        if self.scheduler_name:
            scheduler = self._get_scheduler(self.scheduler_name, optimizer, **self.scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        return optimizer
    
    # Training
    def training_step(self, batch, batch_idx):
        data, labels, patch_ids, cms, dms = batch        
        logits = self(data)
        loss = self.loss_func(logits, labels.float(), cms.float(), dms)
        tp, fp, fn, tn = smp.metrics.get_stats(logits, labels, mode='binary', threshold=0.5)
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_precision', precision, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_recall', recall, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)       
        if batch_idx == 0 and self.current_epoch == 0:
            self._log_first_batch_data(data, labels, patch_ids, cms, dms)
        return loss
    
    # Validation
    def validation_step(self, batch, batch_idx):
        data, labels, patch_ids, cms, dms = batch
        logits = self(data)
        loss = self.loss_func(logits, labels.float(), cms.float(), dms)
        tp, fp, fn, tn = smp.metrics.get_stats(logits, labels, mode='binary', threshold=0.5)
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)       
        
    # Testing
    def test_step(self, batch, batch_idx):
        data, labels, patch_ids, cms, dms = batch
        if self.use_tta:
            tta_transforms = self._get_tta_transformations()
            tta_logits = []

            for transform in tta_transforms:
                augmented_data = transform.augment_image(data)
                output = self(augmented_data)
                output = transform.deaugment_mask(output)
                tta_logits.append(output)
            
            logits = torch.mean(torch.stack(tta_logits), dim=0)
        else:
            logits = self(data)

        loss = self.loss_func(logits, labels.float(), cms.float(), dms)
        tp, fp, fn, tn = smp.metrics.get_stats(logits, labels, mode='binary', threshold=0.5)
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")       
        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_dice', dice, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_iou', iou, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_precision', precision, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_recall', recall, on_epoch=True, logger=True, batch_size=self.batch_size)      
        return {"test_loss": loss, "data": data, "labels": labels, "patch_ids": patch_ids, "logits": logits}

    def on_train_epoch_start(self):
        self.train_start_time = time.time()
        
    def on_train_epoch_end(self): 
        self.log('alpha', self.alpha)
        self.alpha += 0.01
        train_duration = time.time() - self.train_start_time
        self.log("train_time", train_duration, on_epoch=True, logger=True)
        
    def on_validation_epoch_start(self):
        self.val_start_time = time.time()
                
    def on_validation_epoch_end(self):
        val_duration = time.time() - self.val_start_time
        self.log("val_time", val_duration, on_epoch=True, logger=True)
        
    def on_test_epoch_start(self):
        self.test_start_time = time.time()
        self.agg_test_data = []
        self.agg_test_labels = []
        self.agg_test_patch_ids = []
        self.agg_test_logits = []
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.agg_test_data.append(outputs['data'])
        self.agg_test_labels.append(outputs['labels'])
        self.agg_test_patch_ids.extend(outputs['patch_ids'])
        self.agg_test_logits.append(outputs['logits'])
    
    def on_test_epoch_end(self):
        test_duration = time.time() - self.test_start_time
        self.log("test_time", test_duration, on_epoch=True, logger=True)
        
        agg_test_data = torch.cat(self.agg_test_data, dim=0)
        agg_test_labels = torch.cat(self.agg_test_labels, dim=0)
        agg_test_logits = torch.cat(self.agg_test_logits, dim=0)
        
        self._log_predictions_results_table(agg_test_data, agg_test_labels, self.agg_test_patch_ids, agg_test_logits)  
        
        # Check if prediction export is enabled
        if self.export_preds is True:
            self._export_predictions(agg_test_logits, self.agg_test_patch_ids)
    
    # Get Model Selection
    def _get_model(self, model_name, **kwargs):
        backbone = kwargs.get('backbone', 'resnet34')
        use_pretrained = kwargs.get('use_pretrained', True)
        if model_name == 'UNet':
            model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if use_pretrained else None,
                in_channels=self.input_channels,
                classes=self.num_classes
            )
        elif model_name == 'SegFormer':
            if use_pretrained:
                model = SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=self.num_classes , num_channels=self.input_channels, ignore_mismatched_sizes=True)
            else:
                config = SegformerConfig.from_pretrained(backbone, num_labels=self.num_classes , num_channels=self.input_channels)
                model = SegformerForSemanticSegmentation(config)
        else:
            raise ValueError(f"{model_name} not supported")
        return model

    # Get Optimizer Selection
    def _get_optimizer(self, name, params, **kwargs):
        if name == 'Adam':
            return torch.optim.Adam(params, **kwargs)
        elif name == 'AdamW':
            return torch.optim.AdamW(params, **kwargs)
        elif name == 'SGD':
            return torch.optim.SGD(params, **kwargs)
        else:
            raise ValueError(f"{name} not supported")

    # Get Scheduler Selection
    def _get_scheduler(self, name, optimizer, **kwargs):
        if name == 'PolynomialLR':
            return torch.optim.lr_scheduler.PolynomialLR(optimizer, **kwargs)
        elif name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        else:
            raise ValueError(f"{name} not supported")

    # Get Loss Function Selection
    def _get_loss_function(self, name, **kwargs):
        if name == 'DL':
            loss_func = smp.losses.DiceLoss(mode='binary', **kwargs)
            return lambda logits, labels, cm, dm: loss_func(logits, labels)
        elif name == 'DL_BL':
            loss_func =  DL_BL()
            return lambda logits, labels, cm, dm: loss_func(logits, labels, dm, self.alpha)
        else:
            raise ValueError(f"{name} not supported")
     
     # Get TTA Geometric Transformations
    def _get_tta_transformations(self):
        return tta.Compose([
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270])
        ])
           
    # Log - W&B some input samples to verify data is being loaded correctly
    def _log_first_batch_data(self, data, labels, patch_ids, cms, dms):
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            for i, patch_id in enumerate(patch_ids[:8]):
                true_color_img = self._generate_true_color_composite(data[i])
                label_img = labels[i].squeeze().cpu().numpy()
                cm_img = cms[i].squeeze().cpu().numpy()
                dm_img = dms[i].squeeze().cpu().numpy()

                images_to_plot = [true_color_img, label_img, cm_img, dm_img]
                titles = ['True Color', 'Label', 'Cloud Mask', 'Distance Map']

                if self.additional_features:
                    additional_bands = data[i][4:, :, :].cpu().numpy()
                    additional_bands_imgs = []
                    additional_titles = []    
                    
                    for band, feature in zip(additional_bands, self.additional_features):
                        if feature != 'CM':
                            additional_bands_imgs.append(self._generate_additional_bands(band))
                            additional_titles.append(feature)
                    
                    images_to_plot.extend(additional_bands_imgs)
                    titles.extend(additional_titles)

                num_images = len(images_to_plot)
                fig, axs = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

                for j, (img, title) in enumerate(zip(images_to_plot, titles)):
                    cmap = 'gray' if title in ['Label', 'Cloud Mask'] else ('RdBu' if title == 'Distance Map' else None)
                    norm = MplNormalize(vmin=0, vmax=1) if title in ['Label', 'Cloud Mask'] else None
                    axs[j].imshow(img, cmap=cmap, norm=norm)
                    axs[j].set_title(title)
                    axs[j].axis('off')

                # Save the figure temporarily
                image_path = os.path.join(temp_dir, f"{patch_id}.png")
                fig.savefig(image_path)
                plt.close(fig)

                # Log image with W&B
                wandb.log({f"Training_Batch_0_PatchID_{patch_id}": [wandb.Image(image_path)]})

        finally:
            # Remove the temporary directory and its contents
            shutil.rmtree(temp_dir)

    # Log - W&B results table comparing some predicted samples with ground truth
    def _log_predictions_results_table(self, data, labels, patch_ids, logits):
        class_labels = {0: "Non-burned", 1: "Burned"}
        columns = ["ID", "True Color Image with Masks", "Dice Score", "IoU Score", "Boundary Complexity"]
        table = wandb.Table(columns=columns)

        evaluations = []

        for i in range(len(patch_ids)):
            # Compute dice and iou scores per patch
            logits_i = logits[i].unsqueeze(0)
            labels_i = labels[i].unsqueeze(0)
            tp, fp, fn, tn = smp.metrics.get_stats(logits_i, labels_i, mode='binary', threshold=0.5)
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise").item()
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise").item()

            # Edge detection on the ground truth to estimate boundary complexity
            edge_gt = cv2.Canny((labels_i.squeeze().cpu().numpy() * 255).astype(np.uint8), 100, 200)
            boundary_complexity = np.sum(edge_gt) / (labels_i.size(-1) * labels_i.size(-2))  # Normalize by patch size

            evaluations.append((patch_ids[i], data[i], labels[i], logits[i], dice, iou, boundary_complexity))

        # Sort by Dice score for selecting the highest and lowest scores
        evaluations.sort(key=lambda x: x[4], reverse=True)
        high_low_dice_samples = evaluations[:10] + evaluations[-10:]

        # Sort by boundary complexity for selecting the most complex cases
        evaluations.sort(key=lambda x: x[6], reverse=True)
        complex_boundary_samples = evaluations[:40] 

        # Combine selected samples ensuring unique entries in case of overlap
        selected_samples = {sample[0]: sample for sample in high_low_dice_samples + complex_boundary_samples}.values()

        for patch_id, data, label, logit, dice, iou, boundary_complexity in selected_samples:
            true_color_img = self._generate_true_color_composite(data)
            mask_img = wandb.Image(
                true_color_img,
                masks={
                    "predictions": {
                        "mask_data": (torch.sigmoid(logit) > 0.5).float().squeeze().cpu().numpy(),
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": label.squeeze().cpu().numpy(),
                        "class_labels": class_labels
                    }
                }
            )
            table.add_data(patch_id, mask_img, dice, iou, boundary_complexity)

        wandb.log({f"Testing_Results_Prediction_Table": table})
    
    # Generate Normalized True Color Composite  (RGB)  
    def _generate_true_color_composite(self, data_tensor):
        rgb = data_tensor[[0, 1, 2], :, :].cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = rgb.transpose((1, 2, 0))  
        return rgb
    
    # Normalize any additional band
    def _generate_additional_bands(self, band):
        band = (band - band.min()) / (band.max() - band.min() + 1e-10)
        return band
    
    # Export georeferenced Predictions
    def _export_predictions(self, logits, patch_ids):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_output_dir = os.path.join(self.outputs_dir, current_datetime)
        os.makedirs(base_output_dir, exist_ok=True)

        # Load coordinate system information
        metadata_path = os.path.join(self.data_root_dir, 'metadata', 'activations_metadata.csv')
        metadata_df = pd.read_csv(metadata_path, encoding='Windows-1252')
        
        for i, patch_id in enumerate(patch_ids):
            activation_id, patch_number = patch_id.split('_')
            
            # Ensure subdirectory for activation_id exists
            output_dir = os.path.join(base_output_dir, activation_id)
            os.makedirs(output_dir, exist_ok=True)

            # Find the coordinate system
            cs_text = metadata_df.loc[metadata_df['activation_id'] == activation_id, 'cs_i'].values[0]
            
            # Find and copy the .tfw file
            tfw_path = os.path.join(self.data_root_dir, activation_id, 'labels', f"{patch_id}.tfw")
            with open(tfw_path, 'r') as tfw_file:
                geotransform = [float(line.strip()) for line in tfw_file.readlines()]

            # Prepare probabilities for exporting
            probs  = torch.sigmoid(logits[i]).detach().cpu().numpy()
            probs  = probs.squeeze()  # Assuming single-channel output

            # Prepare GeoTIFF path within the activation_id subdirectory
            output_path = os.path.join(output_dir, f"{patch_id}.tif")
            
            # Write to GeoTIFF
            with rasterio.open(
                output_path, 
                'w',
                driver='GTiff',
                height=probs.shape[0],
                width=probs.shape[1],
                count=1,
                dtype='float32',
                crs=cs_text,
                transform=from_origin(geotransform[4], geotransform[5], geotransform[0], -geotransform[3]),
                compress='lzw',
                nodata=3
            ) as dst:
                dst.write(probs, 1)
        print(f"Predictions were successfully exported to: {base_output_dir}")
        # Compress the output directory
        output_zip_path = os.path.join(self.outputs_dir, f"{current_datetime}.zip")
        shutil.make_archive(base_output_dir, 'zip', base_output_dir)
        print(f"Directory compressed into: {output_zip_path}")
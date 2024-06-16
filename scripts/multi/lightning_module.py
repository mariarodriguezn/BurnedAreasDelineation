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

# Define Multi Task Unet Architecture
class MultiTaskUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', use_pretrained=True, in_channels=6, num_ba_classes=1, num_lc_classes=12):
        super(MultiTaskUNet, self).__init__()
        # Base U-Net model
        self.base_model = smp.Unet(encoder_name=encoder_name,
                                   encoder_weights="imagenet" if use_pretrained else None,
                                   in_channels=in_channels,
                                   classes=num_ba_classes
                                  )
        
        # Adding a new segmentation head for land cover
        self.segmentation_head_lc = smp.base.SegmentationHead(
            in_channels=self.base_model.decoder.blocks[-1].conv2[0].out_channels,
            out_channels=num_lc_classes,
            kernel_size=3,
        )

    def forward(self, x):
        features = self.base_model.encoder(x)
        x = self.base_model.decoder(*features)

        # Output for burned area segmentation
        output_ba = self.base_model.segmentation_head(x)
        
        # Output for land cover segmentation
        output_lc = self.segmentation_head_lc(x)
        
        return output_ba, output_lc

# Define Multi Task Segformer Architecture
class MultiTaskSegformer(nn.Module):
    def __init__(self, encoder_name, use_pretrained=True, in_channels=6, num_ba_classes=1, num_lc_classes=12):
        super(MultiTaskSegformer, self).__init__()

        # Load the pre-trained Segformer or initialize from a config
        if use_pretrained:
            self.base_model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name,
                num_labels=num_ba_classes,
                num_channels=in_channels,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig.from_pretrained(encoder_name, num_labels=num_ba_classes, num_channels=in_channels)
            self.base_model = SegformerForSemanticSegmentation(config)

        # Remove the classifier layer from the decode head
        self.base_model.decode_head.classifier = nn.Identity()

        # Define new segmentation heads for different tasks
        self.output_ba = nn.Conv2d(
            in_channels=self.base_model.config.decoder_hidden_size,
            out_channels=num_ba_classes,
            kernel_size=1,
            stride=1
        )

        self.output_lc = nn.Conv2d(
            in_channels=self.base_model.config.decoder_hidden_size,
            out_channels=num_lc_classes,
            kernel_size=1,
            stride=1
        )

    # Forward Pass
    def forward(self, x):
        # Compute features
        features = self.base_model(x)

        # Get the logits from the Segformer model (decoder output)
        logits = features.logits

        # Generate outputs for each task using their specific heads
        output_ba = self.output_ba(logits)
        output_lc = self.output_lc(logits)
        
        # Interpolate to initial dimensions
        output_ba = F.interpolate(output_ba, size=x.shape[-2:], mode='bilinear', align_corners=False)
        output_lc= F.interpolate(output_lc, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return output_ba, output_lc

class LightningSegmentation(pl.LightningModule):
    def __init__(self, 
                 batch_size, 
                 num_epochs,
                 model_name, 
                 optimizer_name, 
                 scheduler_name, 
                 loss_func_name_ba, 
                 loss_func_name_lc,
                 loss_factor_lc,
                 data_root_dir,
                 additional_features, 
                 all_stats,
                 use_tta,
                 outputs_dir,
                 export_preds,
                 model_kwargs={}, 
                 optimizer_kwargs={},
                 scheduler_kwargs={}, 
                 loss_func_kwargs_ba={},
                 loss_func_kwargs_lc={}):
        
        super().__init__()
        self.additional_features = additional_features
        
        self.input_channels = 4 + len(self.additional_features) if self.additional_features else 4

        self.num_ba_classes = 1   
        self.num_lc_classes = 12  # Number of classes for the auxiliary head
        
        self.batch_size = batch_size
        self.model = self._get_model(model_name, **model_kwargs)
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs        
        self.loss_func_ba = self._get_loss_function(loss_func_name_ba, **loss_func_kwargs_ba)
        self.loss_func_lc = self._get_loss_function(loss_func_name_lc, **loss_func_kwargs_lc)
        self.loss_factor_lc =loss_factor_lc
        self.use_tta = use_tta
        self.data_root_dir = data_root_dir
        self.outputs_dir=outputs_dir
        self.export_preds=export_preds
        self.example_input_array = torch.Tensor(batch_size,  self.input_channels, 512, 512)
        
        # Save hyperparameters
        self.save_hyperparameters()
            
    def forward(self, x):
        output_ba, output_lc = self.model(x)
        return output_ba, output_lc
    
    # Set Up Optimizer
    def configure_optimizers(self):
        optimizer = self._get_optimizer(self.optimizer_name, self.parameters(), **self.optimizer_kwargs)
        
        if self.scheduler_name:
            scheduler = self._get_scheduler(self.scheduler_name, optimizer, **self.scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        return optimizer
    
    # Training
    def training_step(self, batch, batch_idx):
        data, labels, patch_ids, cms, dms, lcs = batch        
        logits_ba, logits_lc = self(data)        
        preds_lc = torch.argmax(logits_lc, dim=1).unsqueeze(1)
        
        loss_ba = self.loss_func_ba(logits_ba, labels.float())
        loss_lc = self.loss_func_lc(logits_lc, lcs.long())
        loss = loss_ba + loss_lc * self.loss_factor_lc
        
        # Metrics ba
        tp, fp, fn, tn = smp.metrics.get_stats(logits_ba, labels, mode='binary', threshold=0.5)
        dice_ba = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou_ba = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision_ba = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall_ba = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")    
         # Metrics lc
        tp, fp, fn, tn = smp.metrics.get_stats(preds_lc, lcs, mode='multiclass', num_classes=self.num_lc_classes)
        dice_lc = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou_lc = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision_lc = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall_lc = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")        
            
        self.log('train_loss_ba', loss_ba, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_loss_lc', loss_lc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        self.log('train_dice_ba', dice_ba, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_dice_lc', dice_lc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        self.log('train_iou_ba', iou_ba, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_iou_lc', iou_lc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        self.log('train_precision_ba', precision_ba, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_precision_lc', precision_lc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        self.log('train_recall_ba', recall_ba, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_recall_lc', recall_lc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)       
       
        if batch_idx == 0 and self.current_epoch == 0:
            self._log_first_batch_data(data, labels, patch_ids, cms, dms, lcs)
        return loss
    
    # Validation
    def validation_step(self, batch, batch_idx):
        data, labels, patch_ids, cms, dms, lcs = batch        
        logits_ba, logits_lc = self(data)
        preds_lc = torch.argmax(logits_lc, dim=1).unsqueeze(1)
        loss_ba = self.loss_func_ba(logits_ba, labels.float())
        loss_lc = self.loss_func_lc(logits_lc, lcs.long())
        loss = loss_ba + loss_lc * self.loss_factor_lc
        
        # Metrics ba
        tp, fp, fn, tn = smp.metrics.get_stats(logits_ba, labels, mode='binary', threshold=0.5)
        dice_ba = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou_ba = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision_ba = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall_ba = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")    
         # Metrics lc
        tp, fp, fn, tn = smp.metrics.get_stats(preds_lc, lcs, mode='multiclass', num_classes=self.num_lc_classes)
        dice_lc = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou_lc = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision_lc = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall_lc = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")   
               
        self.log('val_loss_ba', loss_ba, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_loss_lc', loss_lc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        self.log('val_dice_ba', dice_ba, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_dice_lc', dice_lc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        self.log('val_iou_ba', iou_ba, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_iou_lc', iou_lc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        self.log('val_precision_ba', precision_ba, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_precision_lc', precision_lc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        self.log('val_recall_ba', recall_ba, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)   
        self.log('val_recall_lc', recall_lc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)       
    
    # Testing    
    def test_step(self, batch, batch_idx):
        data, labels, patch_ids, cms, dms, lcs = batch        
        if self.use_tta:
            tta_transforms = self._get_tta_transformations()
            tta_logits_ba = []

            for transform in tta_transforms:
                augmented_data = transform.augment_image(data)
                output_ba, _ = self(augmented_data)
                output_ba = transform.deaugment_mask(output_ba)
                tta_logits_ba.append(output_ba)
            
            logits_ba = torch.mean(torch.stack(tta_logits_ba), dim=0)
        else:
            logits_ba, _ = self(data)

        loss_ba = self.loss_func_ba(logits_ba, labels.float())
        
        # Metric ba
        tp, fp, fn, tn = smp.metrics.get_stats(logits_ba, labels, mode='binary', threshold=0.5)
        dice_ba = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou_ba= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        precision_ba = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        recall_ba = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")       
        self.log('test_loss', loss_ba, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_dice', dice_ba, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_iou', iou_ba, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_precision', precision_ba, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_recall', recall_ba, on_epoch=True, logger=True, batch_size=self.batch_size)      
        return {"test_loss": loss_ba, "data": data, "labels": labels, "patch_ids": patch_ids, "logits": logits_ba}

    def on_train_epoch_start(self):
        self.train_start_time = time.time()
        
    def on_train_epoch_end(self): 
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
    
    def _get_model(self, model_name, **kwargs):
        backbone = kwargs.get('backbone', 'resnet34')
        use_pretrained = kwargs.get('use_pretrained', True)        
        if model_name == 'MultiTaskUNet':
            model = MultiTaskUNet(encoder_name=backbone, use_pretrained=use_pretrained, in_channels=self.input_channels, num_ba_classes=self.num_ba_classes, num_lc_classes=self.num_lc_classes)
        elif model_name == 'MultiTaskSegformer':
            model = MultiTaskSegformer(encoder_name=backbone, use_pretrained=use_pretrained, in_channels=self.input_channels, num_ba_classes=self.num_ba_classes, num_lc_classes=self.num_lc_classes)
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
        mode = kwargs.pop('mode', None)
        if name == 'DL':
            loss_func = smp.losses.DiceLoss(mode=mode, **kwargs)
            return lambda logits, labels: loss_func(logits, labels)
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
    def _log_first_batch_data(self, data, labels, patch_ids, cms, dms, lcs):
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            for i, patch_id in enumerate(patch_ids[:8]):
                true_color_img = self._generate_true_color_composite(data[i])
                label_img = labels[i].squeeze().cpu().numpy()
                lc_img = lcs[i].squeeze().cpu().numpy()
                cm_img = cms[i].squeeze().cpu().numpy()

                images_to_plot = [true_color_img, label_img, lc_img, cm_img]
                titles = ['True Color', 'Label', 'Land Cover', 'Cloud Mask']

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

                # Log image with W&B or other visualization tools
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
            probs  = torch.sigmoid(logits[i]).detach().cpu().numpy()  # Ensure logits are on CPU and detached from the computation graph
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
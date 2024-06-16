config = {
    'wandb_api_key': ' ',
    'data_root_dir': r'/share/projects/erasmus/mprn/inputs/patches',
    'outputs_dir': r'/share/projects/erasmus/mprn/outputs', 
    'random_state': 44,
    'experiment_name': "40.2 Single Unet Greece DL pretrained - NDVI - With TTA - With MP",
    'batch_size': 8,
    'num_epochs': 20,
    'gradient_accumulation_steps': 1,
    'additional_features': ['NDVI'],#NDVI #BAI #NDWI #CM
    'all_stats': {
        'spectral_bands': {
            'mean': [0.1137, 0.1265,  0.1391,  0.1770],
            'std': [ 0.0839, 0.0746, 0.0667, 0.1103],
            'min': [1.0, 0.0, 0.0, 1.0],
            'max': [4596.0, 4602.0, 4271.0, 4768.0]
        },
        'NDVI': {
            'mean': 0.5001,
            'std': 0.1946,
            'min': -0.4378,
            'max': 0.7914
        },
        'BAI': {
            'mean': 6.9864e-06,
            'std': 1.1876e-05,
            'min': 2.3548e-08,
            'max': 0.5905
        },
        'NDWI': {
            'mean': 0.5424,
            'std': 0.1649,
            'min': -1.0,
            'max':  0.6581
        }
    },
    'model_name': 'UNet', # SegFormer
    'model_kwargs': {'backbone': 'resnet34','use_pretrained': True}, # nvidia/mit-b2
    'optimizer_name': 'AdamW',
    'optimizer_kwargs': {'lr': 0.0001, 'weight_decay': 1e-4},
    'scheduler_name': None, # PolynomialLR  # CosineAnnealingLR
    'scheduler_kwargs':  {}, # total_iters=55 # T_max = 55
    'loss_func_name': 'DL',  # DL_BL
    'loss_func_kwargs':  {},
    'use_tta': True,
    'export_preds': True,
    'use_mixed_precision': True,   
}
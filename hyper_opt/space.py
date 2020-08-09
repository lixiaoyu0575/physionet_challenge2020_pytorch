from hyperopt import hp, tpe, fmin, Trials

space = {
    'arch':
        hp.choice('arch', [
            {
                'type': 'InceptionTimeV2',
                'args':
                    {
                        "in_channels": 12,
                        "num_classes": 108,
                        "n_filters": hp.choice('n_filters', [32]),
                        "kernel_sizes": hp.choice('kernel_sizes', [[3, 7, 15]]),
                        "bottleneck_channels": hp.choice('bottleneck_channels', [32])
                    }
            },
            {
                'type': 'TCN',
                'args':
                    {
                        "input_size": 12,
                        "num_classes": 108,
                        "num_channels": [32, 32, 32, 32, 32],
                        "kernel_size": 7,
                        "dropout": 0.2
                    }
            }
        ]),

    'data_split':
        hp.choice('data_split', ['split1', 'split2']),

    'optimizer':
        hp.choice('optimizer', [
            {
                'type': 'Adam',
                'args':
                    {
                        'lr': hp.choice('lr', [0.01, 0.001]),
                        'weight_decay': hp.choice('weight_decay', [1e-3, 1e-4, 0]),
                        'amsgrad': True
                    }
            }
        ]),

    'loss':
        hp.choice('loss', [
            {
                'type': 'bce_with_logits_loss'
            }
        ]),

    'lr_scheduler':
        hp.choice('lr_scheduler', [
            {
                "type": "StepLR",
                "args":
                    {
                        "step_size": hp.choice('step_size', [30, 50]),
                        "gamma": hp.choice('StepLR_gamma', [0.1, 0.5])
                    }
            },
            {
                "type": "ExponentialLR",
                "args":
                    {
                        "gamma": hp.choice('ExponentialLR_gamma', [0.1, 0.5])
                    }
            },
            {
                "type": "CosineAnnealingWarmRestarts",
                "args": {
                    "T_0": 10,
                    "T_mult": 1,
                    "eta_min": 0
                }
            },
            {
                "type": "ReduceLROnPlateau",
                "args": {
                    "mode": "min",
                    "factor": 0.1,
                    "patience": 10,
                    "verbose": False,
                    "threshold": 0.0001,
                    "threshold_mode": "rel",
                    "cooldown": 0,
                    "min_lr": 0,
                    "eps": 1e-08
                }

            }
        ]),

    'trainer':
        hp.choice('trainer', [
            {
                "epochs": hp.choice('epochs', [1]),
                "monitor": hp.choice('monitor', ['min val_loss', 'max val_challenge_metric']),
                'early_stop': hp.choice('early_stop', [10, 15, 20])
            },
        ])
}
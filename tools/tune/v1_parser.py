import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    # parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
    #     of input square patches - default 16 (for 16x16 patches). Using smaller
    #     values leads to better performance but requires more memory. Applies only
    #     for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
    #     mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256. Default=0.0005""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # # Multi-crop parameters
    # parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
    #     help="""Scale range of the cropped image before resizing, relatively to the origin image.
    #     Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
    #     recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    # parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
    #     help="""Scale range of the cropped image before resizing, relatively to the origin image.
    #     Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=31, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # My additions
    parser.add_argument("--grad_accumulation_steps", default=1, type=int, help="Steps between gradient accumulation")
    parser.add_argument("--tile_size", default="auto", type=str, help="Size of tiles in slides. Do not change.")
    parser.add_argument("--n_tiles", default=10, type=int, help="How many tiles to use from a slide at each training step")
    parser.add_argument("--random_sampling", default=True, type=bool, help="If tiles should be sampled randomly")
    parser.add_argument("--stage_signal", default=False, type=bool, help="Train on stage signal? If True, ads an ABMIL stage predictor which influences the loss as well.")
    
    #parser.add_argument('--output_dir', default="lokr_dinov1/heavyaug/dino_v2_tcga_ha_lokr", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--output_dir', default="lora_dinov1/heavyaug/dino_p_s36_ha", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,    #4 ok for phikon
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument("--gpu", default=1, type=int, help="Which single GPU to use (if not torchrun)")

    parser.add_argument("--basemodel_name", default="phikon", type=str, choices=['phikon', 'phikon2', 'virchow2'], help="Which base ViT model to use.")
    parser.add_argument("--training_cohorts", default=["S36_LUSC"], type=list, help='Which dataset(s) to train on. ["TCGA_LUSC", "S36_LUSC", "UNN_LUSC"] or TCGA_NSCLC etc')
    parser.add_argument("--lora_rank", default=16, type=int, help="Rank in Low Rank configuration. Default is 16.")
    parser.add_argument("--use_dora", default=True, type=bool, help="Use dora in the lora?")
    parser.add_argument("--lora_type", default="lora", type=str, help="lora, lokr, ... Which type of low-rank adaption to use.")
    
    return parser
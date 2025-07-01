import argparse
import json
import math
import numpy as np
import os
import datetime
import pandas as pd
from pathlib import Path
import sys
import time
import timm
from timm.layers import SwiGLUPacked
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from transformers import ViTModel, AutoModel
from tqdm import tqdm

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from huggingface_hub import login
from copy import deepcopy
from peft import LoraConfig, get_peft_model
from peft import LoKrConfig, LoKrModel

import utils_v1 as utils

sys.path.append('/home/vilde/code/Phikon/HistoSSLscaling')
from rl_benchmarks.utils.augmentations import preprocess
from rl_benchmarks.constants import (
    AVAILABLE_COHORTS,
    PREPROCESSED_DATA_DIR,
    TILE_SIZES,
    HF_TOKEN,
)
from rl_benchmarks.datasets import load_dataset


def print_trainable_parameters(model: torch.nn) -> None:
    """Print number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param}"
        f" || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def load_phikon_models(model_name="phikon", use_dora=True, lora_type="lora", use_rank=16):
    if model_name=="phikon":
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        embed_dim = 768 #phikon: 768, phikon2:1024
        target_modules=["query", "key", "value", "projection"]
    elif model_name=="phikon2":
        model = AutoModel.from_pretrained("owkin/phikon-v2") #ViTModel.from_pretrained("owkin/phikon-v2", add_pooling_layer=False)
        embed_dim = 1024 #phikon: 768, phikon2:1024
        target_modules=["query", "key", "value"]
        print("Phikon2 model downloaded")
    elif model_name.lower() =="virchow2":
        login(token=HF_TOKEN, add_to_git_credential=True)
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        embed_dim = 2560
        target_modules=["qkv"]
    else:
        print(f"Model name {model_name} is not implemented for lora training yet.")
        return
    print_trainable_parameters(model)

    print(f"Building a model of type {model_name} with a {lora_type} adapter, with rank {use_rank}")
    print("Use dora?", use_dora)

    if lora_type.lower()=="lokr":
        config = LoKrConfig(
            task_type="VISION",
            r=use_rank,
            alpha=16,
            target_modules=target_modules,
            module_dropout=0.1,
            rank_dropout=0.1,
            init_weights=True,
        )
    else:
        config = LoraConfig(
            r=use_rank,   #16 is standard
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
            use_dora=use_dora,
        )
    #print(config)
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)
    return lora_model, embed_dim

"""
def load_data(tile_size=224):
    dataset_loading_fn = load_dataset
    dataset = dataset_loading_fn(
        features_root_dir=features_output_dir, tile_size=tile_size
    )
    return dataset
"""

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, n_global_views, stage_loss):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    slides_used = {}
    for it, (images, meta) in enumerate(metric_logger.log_every(data_loader, 500, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        #print("Global training iteration", it, flush=True)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True, device=args.gpu) for im in images]
        images_g = torch.stack(images[:n_global_views], dim=0)
        images_l = torch.stack(images[n_global_views:], dim=0)
        images_g = torch.permute(images_g, (1,0,2,3,4)) # New shape (bs, local_crops, 3, 98,98)
        images_l = torch.permute(images_l, (1,0,2,3,4)) # New shape (bs, local_crops, 3, 98,98)
        bs = images[0].shape[0]
        #print("local images shape", images_l.shape)

        # for s in meta["slide_name"]:
        #     if s in slides_used.keys():
        #         slides_used[s] += 1
        #     else:
        #         slides_used[s] = 1
        if args.stage_signal:
            label = meta["label"]
            # Use loss function stage_loss
            # import IPython
            # IPython.embed()

        # images_g = torch.stack([im.cuda(non_blocking=True) for im in images[0]], dim=0) # Has shape (global_crops, bs, 3,224,224)
        # images_l = torch.stack([im.cuda(non_blocking=True) for im in images[1]], dim=0)
        # images_g = torch.permute(images_g, (1,0,2,3,4))
        # images_l = torch.permute(images_l, (1,0,2,3,4))     # New shape (bs, local_crops, 3, 98,98)
        # #images = torch.concat([images_g, images_l], dim=1)
        
        # teacher and student forward passes + compute dino loss
        loss = 0

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            for b in range(bs):
                if args.stage_signal:
                    teacher_output, logits = teacher(images_g[b], True)  # only the # global views pass through the teacher
                    student_output_g, logitsg = student(images_g[b], True)
                    student_output_l, logitsl = student(images_l[b], True)
                    student_output = torch.concat([student_output_g, student_output_l], dim=0)
                    
                    batch_loss = dino_loss(student_output, teacher_output, epoch)
                    loss += batch_loss
                else:
                    # teacher_output = teacher(images[:2])
                    teacher_output = teacher(images_g[b])  # only the # global views pass through the teacher
                    student_output_g = student(images_g[b])
                    student_output_l = student(images_l[b])
                    student_output = torch.concat([student_output_g, student_output_l], dim=0)
                    
                    batch_loss = dino_loss(student_output, teacher_output, epoch)
                    loss += batch_loss

        # Which parameters are not getting gradients?
        # print(student)
        # for name, param in student.module.backbone.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is None:
        #             print(f"No grad for {name}")

        if not math.isfinite(loss.item()):
            print("Loss is {} (not finite), stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # gradient accumulation
        loss = loss / args.grad_accumulation_steps
        if args.grad_accumulation_steps > 1:
            print("OBS: gradient accumulation is not finished in implementation!! But you are using it now...")

        # Student update
        if ((it+1) % args.grad_accumulation_steps == 0) or (it+1==len(data_loader)):
            optimizer.zero_grad()
            #print("Optimizer update:", it)
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if ((it+1) % args.grad_accumulation_steps == 0) or (it+1==len(data_loader)):
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if ((it+1) % args.grad_accumulation_steps == 0) or (it+1==len(data_loader)):
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()): # student.module
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        #torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    #print("One epoch finished on device", torch.cuda.current_device())
    # print("SLIDES OVERVIEW:", slides_used)
    
    print("Stats logging", flush=True)
    #metric_logger.synchronize_between_processes() # DDP uncomment
    print("Averaged stats:", metric_logger, flush=True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @hydra.main(
#     version_base=None,
#     config_path="../../conf/tune/",
#     config_name="config",
# )
def train_dino(): #params: DictConfig,
    print("IN train dino!", flush=True)
    from v1_parser import get_args_parser
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Input Arguments:")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    print("____________________________________\n")

    model_name = args.basemodel_name.lower() #"phikon" #"virchow2"
    name_to_extractor = {'phikon': 'iBOTViTBasePANCAN', 'phikon2': 'phikon2', 'virchow2': 'virchow2'}
    feature_extractor_name = name_to_extractor[model_name]

    #ag = utils.init_distributed_mode(args)
    ag = args.gpu
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    cohorts = args.training_cohorts
    batch_size = args.batch_size_per_gpu

    tile_size = args.tile_size
    n_tiles = args.n_tiles
    n_global_views = 2 #3
    print(f"Training on GPU ID {ag}, with batch size {args.batch_size_per_gpu},  {args.num_workers} workers, using {n_tiles} tiles")
    print(f"Output directory {args.output_dir}")
    if tile_size == "auto":
        tile_size = TILE_SIZES[feature_extractor_name]
    else:
        assert (
            TILE_SIZES[feature_extractor_name] == tile_size
        ), f"Please specify a tile size (in pixels) that matches the original implementation, see constants.TILE_SIZES dictionary for details: {TILE_SIZES}"

    features_output_dir = (PREPROCESSED_DATA_DIR / "slide_classification" / "features")
    features_output_dirs = []
    for cohort in cohorts:
        if "TCGA" in cohort:
            c_dir = (features_output_dir / feature_extractor_name / "TCGA" / cohort)
        else:
            c_dir = (features_output_dir / feature_extractor_name / cohort)
        features_output_dirs.append(c_dir)

    # Load data
    if args.stage_signal:
        use_label = "STAGE"
    else:
        use_label = None
    dataset_list = []
    dataset_loading_fn = load_dataset
    for i in range(len(cohorts)):
        single_dataset = dataset_loading_fn(
            cohort=cohorts[i], features_root_dir=features_output_dir, tile_size=tile_size, label=use_label
        )
        dataset_list.append(single_dataset)
    
    dataset = pd.concat(dataset_list, axis=0)
    slides_paths = dataset.slide_path.values
    coords_paths = dataset.coords_path.values

    dataset = utils.DinoDataset(dataset, preprocess, n_tiles=args.n_tiles, random_sampling = args.random_sampling, batch_size=batch_size, num_workers=args.num_workers)
    #sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        #sampler=sampler,
        shuffle=True,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images (tiles).")
    

    # Make student and teacher
    student, embed_dim = load_phikon_models(model_name=model_name, use_dora=args.use_dora, lora_type=args.lora_type, use_rank=args.lora_rank)
    teacher, _ = load_phikon_models(model_name=model_name, use_dora=args.use_dora, lora_type=args.lora_type, use_rank=args.lora_rank)

    # Create ABMIL head if we want the added stage signal
    # And an extra loss function
    stage_abmil = None
    stage_loss = None
    if args.stage_signal:
        from rl_benchmarks.models.slide_models.abmil import ABMIL
        from rl_benchmarks.losses import BCEWithLogitsLoss
        stage_abmil = ABMIL(in_features=embed_dim-3,
                            d_model_attention=128,
                            temperature=1.0,
                            mlp_hidden=[128, 64],
                            mlp_activation=nn.ReLU(),
                            bias=True
                            )
        stage_loss = BCEWithLogitsLoss()
        print("Successfully created extra ABMIL")

    # # Consider not using these wrappers! ??
    use_bn_in_head = False
    norm_last_layer = True
    student = utils.MultiCropWrapper(student, utils.DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=use_bn_in_head,
        norm_last_layer=norm_last_layer), model_name, stage_abmil
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        utils.DINOHead(embed_dim, args.out_dim, use_bn_in_head), model_name, stage_abmil
    )

    # Put on GPU
    student, teacher = student.cuda(device=args.gpu), teacher.cuda(device=args.gpu)
    print("Models are put on gpu", torch.cuda.current_device())
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        print("Synchronizing batch norms")
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[int(args.gpu)])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    print("args gpu:", args.gpu, ", local rank", ag, type(ag))
    #student = nn.parallel.DistributedDataParallel(student, device_ids=[int(args.gpu)], find_unused_parameters=True)
    #print("DDP ok")
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.state_dict()) # student.module
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {model_name} networks.")

    #print(student)
    #print()
    #import IPython
    #IPython.embed()

    # Prepare loss
    dino_loss = utils.DINOLoss(
        args.out_dim,
        args.local_crops_number + n_global_views,  # total number of crops = # global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        model_name,
    ).cuda(device=ag)
    #koleo_loss = KoLeoLoss().cuda()

    # Optimizer
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    use_fp16 = False
    if use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # Schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr * (batch_size * utils.get_world_size()),# / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")


    start_epoch = 0
    start_time = time.time()
    print("Start DINO training")
    # Start training
    for epoch in range(start_epoch, args.epochs):
        #data_loader.sampler.set_epoch(epoch)
        # One epoch
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, n_global_views, stage_loss)

        # Log
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        #utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        #teacher.save_pretrained(str(Path(args.output_dir, "peft_checkpoints")))
        if utils.is_main_process() and ( (args.saveckp_freq and (epoch+1) % args.saveckp_freq == 0) or (epoch == args.epochs-1) ):
            epochstring = str(epoch)
            new_state_dict = teacher.backbone.state_dict()
            eval_dir = os.path.join(args.output_dir, f"eval_checkpoint_{epoch:04}")
            os.makedirs(eval_dir, exist_ok=True)
            teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
            torch.save({"teacher_backbone": new_state_dict}, teacher_ckp_path)
            print("Checkpoint saved at ", eval_dir)
            # teacher.backbone.save_pretrained(str(Path(args.output_dir, f"peft_checkpoints_{epoch:04}")))
            #utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    #world_size = torch.cuda.device_count()
    #torch.multiprocessing.spawn(train_dino, args=(world_size), nprocs=world_size)
    train_dino()
    #destroy_process_group()
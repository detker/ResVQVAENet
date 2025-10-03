import os
import shutil
from tqdm import tqdm
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from safetensors.torch import load_file, save_file
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from torchvision.models import vgg19 as pretrained_vgg19
from torchvision.transforms import Normalize

from src.utils import ImageNetDataset, transforms_training, transforms_testing
from src.model import ConvResidualVQVAE_ResNet50Backbone
from src.vgg19 import VGG19

import warnings
warnings.filterwarnings('ignore')


def add_arguments(parser):
    parser.add_argument('--experiment_name',
                        help='Name of experiment',
                        required=True,
                        type=str)

    parser.add_argument('--path_to_data',
                        help='Path to ImageNet root folder which should contain \train and \test folders',
                        required=True,
                        type=str)

    parser.add_argument('--working_directory',
                        help='Working Directory folder name where experiments'' checkpoints and logs are stored',
                        required=True,
                        type=str)

    parser.add_argument('--checkpoint_dir',
                        help='Name of the folder where checkpoints are stored, inside a \
                        folder labeled by the experiment name',
                        required=True,
                        type=str)

    parser.add_argument('--epochs',
                        help='Number of epochs',
                        default=300,
                        type=int)

    parser.add_argument('--warmup_epochs',
                        help='Number of warmup epochs',
                        default=30,
                        type=int)

    parser.add_argument('--save_checkpoint_interval',
                        help='After how many epochs to save model checkpoints',
                        default=1,
                        type=int)

    parser.add_argument('--per_gpu_batch_size',
                        help='Batch size per GPU',
                        default=256,
                        type=int)

    parser.add_argument('--gradient_accumulation_steps',
                        help='Number of gradient accumulation steps',
                        default=1,
                        type=int)

    parser.add_argument('--learning_rate',
                        help='Max learning rate for cosine scheduler',
                        default=0.003,
                        type=float)

    parser.add_argument('--weight_decay',
                        help='Weight decay for optimizer',
                        default=0.1,
                        type=float)

    parser.add_argument('--custom_weight_init',
                        help='Whether to initialize the model with truncated normal layers',
                        default=False,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('--max_grad_norm',
                        help='Maximum norm for gradient clipping',
                        default=1.0,
                        type=float)

    parser.add_argument('--img_size',
                        help='Width and Height of frames',
                        default=224,
                        type=int)

    parser.add_argument('--in_channels',
                        help='Number of channels in image',
                        default=3,
                        type=int)

    parser.add_argument('--num_workers',
                        help='Number of workers for DataLoader',
                        default=32,
                        type=int)

    parser.add_argument('--adam_beta1',
                        type=float,
                        default=0.9,
                        help='Beta1 parameter for AdamW optimizer')

    parser.add_argument('--adam_beta2',
                        type=float,
                        default=0.999,
                        help='Beta2 parameter for AdamW optimizer')

    parser.add_argument('--adam_epsilon',
                        type=float,
                        default=1e-8,
                        help='Epsilon parameter for AdamW optimizer')

    parser.add_argument('--commitment_loss_beta',
                        type=float,
                        default=0.25,
                        help='Weight for commitment loss')

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='EMA decay value')

    parser.add_argument('--use_perceptual_loss',
                        type=bool,
                        action=argparse.BooleanOptionalAction,
                        help='Use additional perceptual loss')

    parser.add_argument('--perceptual_loss_lambda',
                        help='Weight for perceptual loss',
                        default=0.0,
                        type=float)

    parser.add_argument('--log_wandb',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Log metrics to Weight & Biases')

    parser.add_argument('--resume_from_checkpoint',
                        help='Checkpoint folder for model to resume training from, inside the experiment/checkpoints folder',
                        default=None,
                        type=str)

    parser.add_argument('--max_no_of_checkpoints',
                        type=int,
                        default=10,
                        help='Max number of latest checkpoints to store on disk.')


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    experiment_path = os.path.join(args.working_directory, args.experiment_name)
    accelerator = Accelerator(project_dir=experiment_path,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with='wandb' if args.log_wandb else None)

    print(f'Device: {accelerator.device}')

    if args.log_wandb:
        experiment_config = {'epochs': args.epochs,
                             'effective_batch_size': args.per_gpu_batch_size * accelerator.num_processes,
                             'learning_rate': args.learning_rate,
                             'warmup_epochs': args.warmup_epochs,
                             'custom_weight_init': args.custom_weight_init}
        accelerator.init_trackers(args.experiment_name, config=experiment_config)

    transforms_train = transforms_training(img_wh=args.img_size)
    transforms_test = transforms_testing(img_wh=args.img_size)

    train_data = ImageNetDataset(os.path.join(args.path_to_data, 'train'),
                                 transform=transforms_train)
    test_data = ImageNetDataset(os.path.join(args.path_to_data, 'test'),
                                transform=transforms_test)

    minibatch_size = args.per_gpu_batch_size // args.gradient_accumulation_steps
    trainloader = DataLoader(train_data,
                             batch_size=minibatch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)
    testloader = DataLoader(test_data,
                            batch_size=minibatch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
    accelerator.print('Data Loaded.')

    model = ConvResidualVQVAE_ResNet50Backbone(args.in_channels)

    if args.custom_weight_init:
        model.apply(model.init_weights)
    ema_model = ConvResidualVQVAE_ResNet50Backbone(args.in_channels)
    with accelerator.main_process_first():
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad = False
    ema_model = ema_model.to(accelerator.device)

    model = model.to(accelerator.device)

    ### perceptual loss ###
    vgg_model = None
    if args.use_perceptual_loss:
        vgg_model = VGG19(in_channels=args.in_channels)
        pretrained_vgg = pretrained_vgg19(pretrained=True)
        pretrained_state_dict = pretrained_vgg.state_dict()
        my_state_dict = vgg_model.state_dict()
        new_state_dict = {}
        for p_pretrained, p in zip(pretrained_state_dict.keys(), my_state_dict.keys()):
            new_state_dict[p] = pretrained_state_dict[p_pretrained]
        vgg_model.load_state_dict(new_state_dict)
        for p in vgg_model.parameters():
            p.requires_grad = False
        vgg_model = vgg_model.to(accelerator.device)

    def perceptual_loss(recon, x):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        x = normalize(x)
        recon = normalize(recon)
        input = torch.cat([recon, x], dim=0)
        return vgg_model(input)

    def update_ema(ema_model, model, decay=0.999):
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(decay).add_(p, alpha=1 - decay)

    weight_decay_params = []
    non_weight_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name:
                non_weight_decay_params.append(param)
            elif 'bn' in name:
                non_weight_decay_params.append(param)
            elif 'embedding' in name:
                non_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

    param_config = [{'params': non_weight_decay_params, 'lr': args.learning_rate, 'weight_decay': 0.0,
                     'betas': (args.adam_beta1, args.adam_beta2), 'eps': args.adam_epsilon},
                    {'params': weight_decay_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay,
                     'betas': (args.adam_beta1, args.adam_beta2), 'eps': args.adam_epsilon}]
    optimizer = optim.AdamW(param_config)

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=args.warmup_epochs * len(
                                                    trainloader) // args.gradient_accumulation_steps,
                                                num_training_steps=args.epochs * len(
                                                    trainloader) // args.gradient_accumulation_steps)

    model, ema_model, optimizer, scheduler, trainloader, testloader = accelerator.prepare(model,
                                                                                          ema_model,
                                                                                          optimizer,
                                                                                          scheduler,
                                                                                          trainloader,
                                                                                          testloader)
    accelerator.register_for_checkpointing(scheduler)

    accelerator.print('Dependencies loaded.')

    starting_epoch = 0
    if args.resume_from_checkpoint is not None:
        path_to_checkpoint = os.path.join(experiment_path, args.checkpoint_dir, args.resume_from_checkpoint)
        accelerator.load_state(path_to_checkpoint)
        starting_epoch = int(path_to_checkpoint.split('_')[-1]) + 1
        accelerator.print('Loaded checkpoint.')

    accelerator.print(f'Starting training from epoch {starting_epoch + 1}...')

    for epoch in range(starting_epoch, args.epochs):
        train_losses = []
        test_losses = []
        accum_train_loss = 0

        pbar = tqdm(range(len(trainloader) // args.gradient_accumulation_steps),
                    disable=not accelerator.is_main_process)

        model.train()
        for data, labels in trainloader:
            data = data.to(accelerator.device)

            with accelerator.accumulate(model):
                encoded, quantized_encoded, decoded, codebook_loss, commitment_loss, _, _ = model(data)
                reconstruction_loss = torch.mean((data - decoded) ** 2)
                loss = reconstruction_loss + codebook_loss + args.commitment_loss_beta * commitment_loss
                accelerator.print(loss.item())
                if vgg_model is not None:
                    percep = args.perceptual_loss_lambda * perceptual_loss(decoded, data)
                    accelerator.print(f'recon_loss: {reconstruction_loss.item()} - percep_loss:{percep.item()}')
                    loss = loss + percep

                accum_train_loss += loss
                loss = loss / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                                max_norm=args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if ema_model is not None:
                    update_ema(ema_model, model, decay=args.ema_decay)

            if accelerator.sync_gradients:
                accum_train_loss = accum_train_loss.detach()
                gathered = accelerator.gather_for_metrics(accum_train_loss)
                train_losses.append(torch.mean(gathered).item())
                accum_train_loss = 0

                pbar.update(1)
        pbar.close()


        usage_ratios = []
        perplexities = []
        model.eval()
        for data, labels in testloader:
            data = data.to(accelerator.device)
            with torch.no_grad():
                encoded, quantized_encoded, decoded, codebook_loss, commitment_loss, codebooks_usage_ratios, codebooks_perplexities = model(data)
            reconstruction_loss = torch.mean((data - decoded) ** 2)
            loss = reconstruction_loss + codebook_loss + args.commitment_loss_beta * commitment_loss
            loss = loss.detach()
            gathered = accelerator.gather_for_metrics(loss)
            test_losses.append(torch.mean(gathered).item())

            codebooks_usage_ratios = codebooks_usage_ratios.detach().unsqueeze(0)
            codebooks_perplexities = codebooks_perplexities.detach().unsqueeze(0)
            gathered_codebooks_usage_ratio = accelerator.gather_for_metrics(codebooks_usage_ratios)
            gathered_codebooks_perplexities = accelerator.gather_for_metrics(codebooks_perplexities)
            if gathered_codebooks_usage_ratio.ndim == 1:
                gathered_codebooks_usage_ratio = gathered_codebooks_usage_ratio.unsqueeze(0)
            if gathered_codebooks_perplexities.ndim == 1:
                gathered_codebooks_perplexities = gathered_codebooks_perplexities.unsqueeze(0)
            gathered_codebooks_perplexities = torch.mean(gathered_codebooks_perplexities, dim=0).tolist()
            gathered_codebooks_usage_ratio = torch.mean(gathered_codebooks_usage_ratio, dim=0).tolist()
            usage_ratios.append(gathered_codebooks_usage_ratio)
            perplexities.append(gathered_codebooks_perplexities)


        epoch_train_loss = np.mean(train_losses).item()
        epoch_test_loss = np.mean(test_losses).item()
        epoch_usage_ratio = np.mean(usage_ratios, axis=0).tolist()
        epoch_perplexity = np.mean(perplexities, axis=0).tolist()

        accelerator.print(
            f'Epoch: {epoch+1} | Training Loss: {epoch_train_loss:.5f} | Testing Loss: {epoch_test_loss:.5f}.')
        accelerator.print(f'Codebooks usage ratio: {epoch_usage_ratio}')
        accelerator.print(f'Codebooks perplexity: {epoch_perplexity}')

        if args.log_wandb:
            accelerator.log({'training_loss': epoch_train_loss,
                             'testing_loss': epoch_test_loss,
                             'lr': scheduler.get_last_lr()[0],
                             'codebooks_usage_ratio_min': min(epoch_usage_ratio),
                             'codebooks_perplexity_min': min(epoch_perplexity)},
                            step=epoch)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if epoch % args.save_checkpoint_interval == 0:
                checkpoints_path = os.path.join(experiment_path, args.checkpoint_dir)
                os.makedirs(checkpoints_path, exist_ok=True)
                listdirs = [file for file in os.listdir(checkpoints_path) if file.startswith('checkpoint')]
                if len(listdirs) >= args.max_no_of_checkpoints:
                    listdirs_sorted = sorted(listdirs, key=lambda x: int(x.split('_')[-1]))
                    dirs_to_delete = listdirs_sorted[:-args.max_no_of_checkpoints + 1]
                    for directory in dirs_to_delete:
                        shutil.rmtree(os.path.join(checkpoints_path, directory))
                save_path = os.path.join(checkpoints_path, f'checkpoint_{epoch}')
                accelerator.save_state(save_path)
                accelerator.print('State saved.')

        accelerator.print(f'End of epoch {epoch + 1}.')

    accelerator.print('End of training loop. Saving final merged weights.')

    if accelerator.is_main_process:
        if not args.use_lora:
            checkpoints_path = os.path.join(experiment_path, args.base_weights_no_lora)
            os.makedirs(checkpoints_path, exist_ok=True)
            state_dict = accelerator.unwrap_model(model).state_dict()
            state_dict_ema = accelerator.unwrap_model(ema_model).state_dict()
            save_file(state_dict, filename=os.path.join(checkpoints_path, 'base_weights.safetensors'))
            save_file(state_dict_ema, filename=os.path.join(checkpoints_path, 'base_weights_ema.safetensors'))
            accelerator.print('Base weights saved.')
        else:
            checkpoint_path = os.path.join(experiment_path, args.merged_weights_from_lora)
            os.makedirs(checkpoint_path, exist_ok=True)
            accelerator.unwrap_model(model).save_weights(checkpoint_path)
            accelerator.print('Finetuned weights saved.')

    accelerator.end_training()


if __name__ == '__main__':
    main()

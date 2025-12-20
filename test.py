import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import glob 
from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel
import argparse
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import csv
import re
import warnings
from datetime import datetime
from tqdm import tqdm

# Suppress ALL warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Reduce CUDA verbosity

import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--csv")
    parser.add_argument("--outfile")
    parser.add_argument('--gpu-ids', default='0,1,2,3', type=str,
                        help='comma-separated list of GPU IDs to use (default: 0,1,2,3)')
    parser.add_argument('--batch-size', default=3, type=int,
                        help='batch size for testing (default: 3)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='save prediction outputs as numpy arrays')
    parser.add_argument('--enable-clipping', action='store_true',
                        help='enable prediction clipping (disabled by default)')
    parser.add_argument('--clipping-threshold', default=30.0, type=float,
                        help='height threshold for clipping predictions (default: 30.0 meters)')
    parser.add_argument('--disable-target-filtering', action='store_true',
                        help='disable target-based filtering of predictions (enabled by default)')
    parser.add_argument('--target-threshold', default=1.0, type=float,
                        help='target height threshold for filtering predictions (default: 1.0 meters)')
    parser.add_argument('--disable-normalization', action='store_true', default=False,
                        help='disable entire normalization pipeline (x1000, /100000, x100) to see raw model outputs (default: False)')
    parser.add_argument('--uint16-conversion', action='store_true', default=False,
                        help='use original IM2ELEVATION uint16 conversion: depth = (depth*1000).astype(np.uint16) (default: False)')
    args = parser.parse_args()
    
    # Extract dataset name from model path for preprocessing
    dataset_name = os.path.basename(args.model.rstrip('/'))
    
    # Configure GPU usage
    device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    # Check if specified GPUs are available
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if max(device_ids) >= available_gpus:
        print(f"Warning: Requested GPU {max(device_ids)} not available. Using available GPUs: {list(range(available_gpus))}")
        device_ids = list(range(min(len(device_ids), available_gpus)))
    
    # Determine GPU mode automatically based on number of devices
    if len(device_ids) == 1:
        print(f"Using single GPU: {device_ids[0]}")
    else:
        print(f"Using multiple GPUs: {device_ids}")

    md = glob.glob(args.model+'/*.tar')
    
    if not md:
        print("No checkpoint files found!")
        return
    
    # Prioritize best checkpoint for evaluation
    best_checkpoints = [x for x in md if 'best_epoch_' in x]
    
    if best_checkpoints:
        # Sort by epoch number and take the latest best
        best_checkpoints.sort(key=lambda x: int(x.split('best_epoch_')[1].split('.')[0]))
        selected_checkpoint = best_checkpoints[-1]
        print(f"Found best checkpoint, evaluating: {os.path.basename(selected_checkpoint)}")
    else:
        # Fallback to latest if no best checkpoint found
        latest_checkpoints = [x for x in md if 'latest' in x]
        if latest_checkpoints:
            selected_checkpoint = latest_checkpoints[0]
            print(f"No best checkpoint found, using latest: {os.path.basename(selected_checkpoint)}")
        else:
            # Use any available checkpoint
            selected_checkpoint = sorted(md, key=natural_keys)[-1]
            print(f"Using available checkpoint: {os.path.basename(selected_checkpoint)}")
    
    print("=" * 60)
    
    checkpoint_name = os.path.basename(selected_checkpoint)
    
    # Create predictions directory if saving predictions
    predictions_dir = None
    if args.save_predictions:
        predictions_dir = os.path.join(args.model, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        print(f"Predictions will be saved to: {predictions_dir}")
    
    # Create model with suppressed output
    f = StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda(device_ids[0])
            print(f"Model wrapped with DataParallel using GPUs: {device_ids}")
        else:
            model = model.cuda(device_ids[0])
            print(f"Model moved to single GPU: {device_ids[0]}")
        state_dict = torch.load(selected_checkpoint, map_location=f'cuda:{device_ids[0]}')['state_dict']
        
        # Handle DataParallel state dict mismatch
        if len(device_ids) > 1 and not any(key.startswith('module.') for key in state_dict.keys()):
            # Model is DataParallel but checkpoint is not - add 'module.' prefix
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif len(device_ids) == 1 and any(key.startswith('module.') for key in state_dict.keys()):
            # Model is not DataParallel but checkpoint is - remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict quietly without printing parameter names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.load_state_dict(state_dict, strict=False)

    test_loader = loaddata.getTestingData(args.batch_size, args.csv, dataset_name, disable_normalization=args.disable_normalization, use_uint16_conversion=args.uint16_conversion)
    result = test(test_loader, model, args, checkpoint_name, predictions_dir, args.csv)
    
    print("=" * 60)


def test(test_loader, model, args, checkpoint_name="", predictions_dir=None, csv_file=None):
    losses = AverageMeter()
    model.eval()
    
    # Get the device from model
    device = next(model.parameters()).device
    
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'SSIM': 0, 'R2': 0}

    # Read CSV file to get image names for saving predictions
    image_names = []
    if csv_file and predictions_dir:
        with open(csv_file, 'r') as f:
            for line in f:
                rgb_path = line.strip().split(',')[0]
                image_name = os.path.basename(rgb_path)
                # Remove file extension properly (handles .png, .tif, .jpg, etc.)
                image_name = os.path.splitext(image_name)[0]
                image_names.append(image_name)

    prediction_idx = 0
    
    # Create progress bar
    total_items = len(image_names) if image_names else len(test_loader.dataset)
    pbar = tqdm(total=total_items, desc="Processing test items", unit="item", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']
        depth = depth.to(device, non_blocking=True)
        image = image.to(device)
        output = model(image)

        # *MODIFICADO para adaptar ao tamanho do DFC2019
        # output = torch.nn.functional.interpolate(output, size=(440,440), mode='bilinear')
        output = torch.nn.functional.interpolate(output, size=(512,512), mode='bilinear')


        # Save predictions if requested
        if predictions_dir:
            batch_size = output.size(0)
            for j in range(batch_size):
                if prediction_idx < len(image_names):
                    # Convert prediction back to original DSM scale (default behavior)
                    if not args.disable_normalization:
                        pred_array = output[j, 0].cpu().detach().numpy() * 100  # Restore to original DSM scale
                    else:
                        pred_array = output[j, 0].cpu().detach().numpy()  # Raw model output for debugging
                    pred_filename = f"{image_names[prediction_idx]}_pred.npy"
                    pred_path = os.path.join(predictions_dir, pred_filename)
                    np.save(pred_path, pred_array)
                    prediction_idx += 1
                    # Update progress bar for each saved prediction
                    pbar.update(1)
        else:
            # Update progress bar based on batch size when not saving predictions
            batchSize = depth.size(0)
            pbar.update(batchSize)

        batchSize = depth.size(0)
        testing_loss(depth, output, losses, batchSize)

        totalNumber = totalNumber + batchSize

        errors = util.evaluateError(output, depth, i, batchSize, 
                                    disable_normalization=args.disable_normalization,
                                    enable_clipping=args.enable_clipping, 
                                    clipping_threshold=args.clipping_threshold,
                                    enable_target_filtering=not args.disable_target_filtering,
                                    target_threshold=args.target_threshold)

        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)

    # Close progress bar
    pbar.close()
    
    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    loss = float(losses.avg)

    # Enhanced output with checkpoint identification
    checkpoint_info = f" ({checkpoint_name})" if checkpoint_name else ""
    print(f'Results{checkpoint_info}:')
    print('  Loss: {loss:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | SSIM: {ssim:.4f} | R²: {r2:.4f}'.format(
        loss=loss, mse=averageError['MSE'], rmse=averageError['RMSE'], 
        mae=averageError['MAE'], ssim=averageError['SSIM'], r2=averageError['R2']))
    
    if predictions_dir:
        print(f"Saved {prediction_idx} predictions to {predictions_dir}")
    
    return {
        'checkpoint': checkpoint_name,
        'loss': loss,
        'MSE': averageError['MSE'],
        'RMSE': averageError['RMSE'], 
        'MAE': averageError['MAE'],
        'SSIM': averageError['SSIM'],
        'R2': averageError['R2']
    }


def testing_loss(depth, output, losses, batchSize):
    device = depth.device
    ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().to(device)
    get_gradient = sobel.Sobel().to(device)
    cos = nn.CosineSimilarity(dim=1, eps=0)
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    loss = loss_depth + loss_normal + (loss_dx + loss_dy)
    losses.update(loss.item(), batchSize)


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]  


if __name__ == '__main__':
    main()

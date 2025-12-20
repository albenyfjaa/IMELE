#!/usr/bin/env python3
"""
GPU Memory Management Utility for IM2ELEVATION Pipeline

This script provides utilities to:
1. Monitor GPU memory usage
2. Clear GPU cache
3. Check available GPUs
4. Optimize memory allocation
"""

import torch
import subprocess
import sys
import argparse
import time
import os


def clear_gpu_cache():
    """Clear PyTorch GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared successfully")
        return True
    else:
        print("No CUDA available")
        return False


def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        print("No CUDA available")
        return
    
    print("=" * 60)
    print("GPU Memory Information")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        cached_memory = torch.cuda.memory_reserved(i)
        free_memory = total_memory - allocated_memory
        
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory:     {total_memory / 1024**3:.2f} GB")
        print(f"  Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
        print(f"  Cached Memory:    {cached_memory / 1024**3:.2f} GB")
        print(f"  Free Memory:      {free_memory / 1024**3:.2f} GB")
        print(f"  Memory Usage:     {(allocated_memory / total_memory) * 100:.1f}%")
        print()


def monitor_gpu_usage(duration=30, interval=2):
    """Monitor GPU usage over time."""
    print(f"Monitoring GPU usage for {duration} seconds (interval: {interval}s)")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    
    try:
        for _ in range(0, duration, interval):
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}]", end=" ")
            
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                usage_pct = (allocated / total) * 100
                print(f"GPU{i}: {usage_pct:.1f}% ({allocated:.1f}GB)", end="  ")
            
            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def optimize_memory_settings():
    """Set optimal memory settings for IM2ELEVATION."""
    print("Setting optimal memory configuration...")
    
    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    print("Memory optimization settings applied:")
    print("  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("  - CUDA_LAUNCH_BLOCKING=0")


def kill_gpu_processes(gpu_id=None):
    """Kill processes using specified GPU(s)."""
    try:
        if gpu_id is not None:
            cmd = ['nvidia-smi', '--id=' + str(gpu_id), '--query-compute-apps=pid', '--format=csv,noheader,nounits']
        else:
            cmd = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"Found {len(pids)} GPU processes to terminate")
            
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid], check=True)
                    print(f"Terminated process {pid}")
                except subprocess.CalledProcessError:
                    print(f"Failed to terminate process {pid}")
        else:
            print("No GPU processes found")
            
    except FileNotFoundError:
        print("nvidia-smi not found. Cannot kill GPU processes.")
    except Exception as e:
        print(f"Error killing GPU processes: {e}")


def suggest_batch_size(gpu_memory_gb):
    """Suggest optimal batch size based on available GPU memory."""
    if gpu_memory_gb >= 24:
        return 4
    elif gpu_memory_gb >= 16:
        return 3
    elif gpu_memory_gb >= 11:
        return 2
    elif gpu_memory_gb >= 8:
        return 1
    else:
        return 1


def main():
    parser = argparse.ArgumentParser(description='GPU Memory Management for IM2ELEVATION')
    parser.add_argument('--clear', action='store_true', help='Clear GPU cache')
    parser.add_argument('--info', action='store_true', help='Show GPU memory information')
    parser.add_argument('--monitor', type=int, metavar='SECONDS', help='Monitor GPU usage for specified seconds')
    parser.add_argument('--optimize', action='store_true', help='Set optimal memory settings')
    parser.add_argument('--kill-processes', action='store_true', help='Kill all GPU processes')
    parser.add_argument('--gpu-id', type=int, help='Specific GPU ID for operations')
    parser.add_argument('--suggest-batch-size', action='store_true', help='Suggest optimal batch size')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # Default behavior: show info and clear cache
        get_gpu_memory_info()
        clear_gpu_cache()
        return
    
    if args.optimize:
        optimize_memory_settings()
    
    if args.clear:
        clear_gpu_cache()
    
    if args.info:
        get_gpu_memory_info()
    
    if args.monitor:
        monitor_gpu_usage(duration=args.monitor)
    
    if args.kill_processes:
        kill_gpu_processes(args.gpu_id)
    
    if args.suggest_batch_size:
        print("Suggested batch sizes based on GPU memory:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            suggested_bs = suggest_batch_size(memory_gb)
            print(f"GPU {i} ({memory_gb:.1f}GB): batch_size = {suggested_bs}")


if __name__ == "__main__":
    main()

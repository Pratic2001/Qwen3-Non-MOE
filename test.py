import torch

# 1. Check if CUDA (GPU support) is available
gpu_available = torch.cuda.is_available()
print(f"Is GPU available? {gpu_available}")

if gpu_available:
    # 2. Get the number of available GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # 3. Get the name of the current GPU
    print(f"Current GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 4. Create a tensor and move it to the GPU
    device = torch.device("cuda")
    x = torch.ones(3, 3, device=device)
    print("\nSuccessfully created a tensor on device:", x.device)
else:
    print("PyTorch is currently using the CPU.")
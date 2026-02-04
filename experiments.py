# reproducibility code from https://github.com/IOAI-official/IOAI-2025
import random
import numpy as np
import torch
def initialize_seeds(seed):
    #seed = 42
    
    random.seed(seed)                  # Python built-in random
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch (CPU)
    torch.cuda.manual_seed(seed)       # PyTorch (single GPU)
    torch.cuda.manual_seed_all(seed)   # PyTorch (all GPUs)
    
    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
initialize_seeds(42)
import os
import joblib
from data import load_data
import model, train
import segmentation_models_pytorch as smp
from data import load_data
config = {'filter': 'none', 'cross_validation': True}
train_ds, val_ds, test_ds, trainloader, valloader, testloader = load_data(config)
model_names = ["ternausnet-vgg16", "smp-densenet121"]
filters = ["none", "median", "bilateral"]
losses = ["focalloss", "crossentropy"]
for model_name in model_names:
    for fltr in filters:
        for loss in losses:
            records = []
            config = {"model_name": model_name, "filter": fltr, "loss": loss}
            fnstring = f"{model_name}_{fltr}_{loss}"
            log_filename = fnstring + ".joblib"
            model_filename = fnstring + ".pt"
            for item in range(1):
                initialize_seeds(item)
                net = None
                if model_name == "smp-densenet121":
                    net = smp.Unet(
                        encoder_name="densenet121",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=2,                      # model output channels (number of classes in your dataset)
                    )
                    #'''
                elif model_name == "ternausnet-vgg16":
                    net = model.UNet16(num_classes=2)
                train_ds, val_ds, test_ds, trainloader, valloader, testloader = load_data(config)
                net, record = train.train(net, 1.0, 5e-06, 60, trainloader, valloader, testloader, f"best_model_{fnstring}_{item}.pt", loss)
                records.append(record)
                print(records)
                joblib.dump(records, log_filename)
                records = np.array(records)
                idx = np.argmin(np.abs(records - np.mean(records)))
                fn = f"best_model_{fnstring}_{idx}.pt"
                os.system(f"cp {fn} {model_filename}")
                print(f"Selected and copied model {idx}") 
            
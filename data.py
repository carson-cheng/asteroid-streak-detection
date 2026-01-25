import os
import joblib
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.restoration import denoise_wavelet, denoise_bilateral
import random
from PIL import Image
import numpy as np
import torch
from skimage.segmentation import flood
import joblib
image, coords = None, None
transform = transforms.Compose([transforms.ToTensor()])
class AsteroidImageDataset(Dataset):
    def __init__(self, files, transform, train, config, negative=False):
        #self.files = files
        self.transform = transform
        self.train = train
        self.negative = negative
        self.cache = os.listdir(f"./data")
        self.cache = [x for x in self.cache if (".joblib" in x and f"{self.train}" in x)]
        self.config = config
        self.filter = self.config['filter']
        print(len(self.cache))
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        #fn = self.files[idx]
        cache_fn = os.path.join("./data", self.cache[idx])
        if os.path.exists(cache_fn):
            (patch, mask) = joblib.load(cache_fn)
            
            # Apply transform first
            transformed = self.transform(patch)  # Should be a tensor
            if self.filter != "none":
                transformed_np = transformed.numpy()
                transformed_np = transformed_np.squeeze(0)
                # Apply median filter
                from skimage.filters import median
                from skimage.morphology import square
                if self.filter == "median":
                    transformed_np = median(transformed_np, square(3))
                elif self.filter == "bilateral":
                    transformed_np = denoise_bilateral(transformed_np)
                else:
                    raise ValueError("filter in config should be either 'none', 'median', or 'bilateral'.")
                # Add channel dimension back
                transformed_np = transformed_np[np.newaxis, :, :]
                transformed = torch.tensor(transformed_np)
            return transformed, mask
        (image, coords) = joblib.load(fn)
        # normalize the image
        if (self.train or not self.train): # use same sampling technique for train & validation for now
            # in training: randomly select positive or negative samples
            # if positive: randomly sample a patch where the asteroid can appear anywhere within the central 84 * 84 box
            # if negative: randomly sample a patch at least 100 pixels away from the asteroid (to prevent long-running trails from
            # contaminating the training data)
            if random.random() < 0.5 or self.negative:
                # negative case
                range_y = [y for y in range(image.shape[0]) if (abs(y - coords[0]) > 384 and abs(y) > 128 and abs(y - image.shape[0]) > 128)]
                range_x = [x for x in range(image.shape[1]) if (abs(x - coords[1]) > 384 and abs(x) > 128 and abs(x - image.shape[1]) > 128)]
                if len(range_y) == 0 or len(range_x) == 0:
                    return self.__getitem__(idx - 1)
                y_center = random.choice(range_y)
                x_center = random.choice(range_x)
                patch = image[x_center-128:x_center+128,y_center-128:y_center+128]
                patch[patch > np.nanpercentile(patch, 99)] = np.nanpercentile(patch, 99)
                patch[patch < np.nanpercentile(patch, 1)] = np.nanpercentile(patch, 1)
                #patch = (patch - patch.min()) / (patch.max() - patch.min())
                #patch = torch.tensor(patch).unsqueeze(0)
                patch = Image.fromarray(patch)
                patch = self.transform(patch)
                mask = torch.zeros_like(patch)
                if patch.shape != torch.Size([1, 256, 256]) or mask.shape != torch.Size([1, 256, 256]):
                    return self.__getitem__(idx - 1)
                #print(patch.max(), patch.min(), patch.std(), patch.mean())
                joblib.dump((patch, mask), cache_fn)
                return patch, mask
            else:
                #print("Positive!")
                range_y = [y for y in range(image.shape[0]) if (abs(y - coords[0]) < 108 and abs(y) > 128 and abs(y - image.shape[0]) > 128)]
                range_x = [x for x in range(image.shape[1]) if (abs(x - coords[1]) < 108 and abs(x) > 128 and abs(x - image.shape[1]) > 128)]
                if len(range_y) == 0 or len(range_x) == 0:
                    return self.__getitem__(idx - 1)
                y_center = random.choice(range_y)
                x_center = random.choice(range_x)
                patch = image[x_center-128:x_center+128,y_center-128:y_center+128]
                patch[patch > np.nanpercentile(patch, 99)] = np.nanpercentile(patch, 99)
                patch[patch < np.nanpercentile(patch, 1)] = np.nanpercentile(patch, 1)
                #patch = (patch - patch.min()) / (patch.max() - patch.min())
                y_coord = int(round(coords[0] - y_center + 128, 2))
                x_coord = int(round(coords[1] - x_center + 128, 2))
                mask = np.zeros_like(patch)
                mask[x_coord-5:x_coord+6,y_coord-5:y_coord+6] = 1.0
                #patch = torch.tensor(patch).unsqueeze(0)
                patch = Image.fromarray(patch)
                patch = self.transform(patch)
                mask1 = torch.tensor(mask).unsqueeze(0)
                if patch.shape != torch.Size([1, 256, 256]) or mask1.shape != torch.Size([1, 256, 256]):
                    print("Falling back")
                    return self.__getitem__(idx - 1)
                #print(patch.max(), patch.min(), patch.std(), patch.mean())
                idx = patch[0][np.array(mask1[0] == 1)].argmax()
                value = patch[0][np.array(mask1[0] == 1)].max()
                seeds_y, seeds_x = np.where(np.array(mask1[0] == 1))
                #print(seeds_y, seeds_x)
                seed_points = [(seeds_y[i], seeds_x[i]) for i in range(len(seeds_y))]
                mask = np.array(patch[0] > (value - 0.2)).astype(np.uint8)
                region = flood(mask, seed_points[idx], connectivity=2)
                mask = torch.tensor([region]).to(torch.float32)
                #print(mask.sum())
                if mask.sum() > 1000 or mask.sum() < 10: # prevent overflowing or underflowing regions
                    #print(mask1.sum())
                    #return self.__getitem__(idx - 1)
                    patch1 = image[x_center-128:x_center+128,y_center-128:y_center+128]
                    mask1 = np.zeros_like(patch1)
                    mask1[x_coord-2:x_coord+3,y_coord-2:y_coord+3] = 1.0
                    mask1 = torch.tensor(mask1).unsqueeze(0)
                    joblib.dump((patch, mask1), cache_fn)
                    return patch, mask1
                joblib.dump((patch, mask), cache_fn)
                return patch, mask
        print("Error!")
        
def load_data(config):
    from skimage.filters import median
    from skimage.morphology import square
    root = "./near-earth-asteroids"
    transform = transforms.Compose([
        lambda x: x  # The identity function
    ])
    #transform = transforms.Compose([transforms.GaussianBlur(5)])
    files = [os.path.join(root, x) for x in os.listdir(root) if "joblib" in x]
    threshold = int(len(files) * 0.8)
    train_files, val_files = files[:threshold], files[threshold:]
    train_ds = AsteroidImageDataset(train_files, transform, True, config)
    val_ds = AsteroidImageDataset(val_files, transform, False, config)
    trainloader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    valloader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    return train_ds, val_ds, trainloader, valloader
if __name__ == '__main__':
    import random
    import numpy as np
    import torch
    
    seed = 42
    
    random.seed(seed)                  # Python built-in random
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch (CPU)
    torch.cuda.manual_seed(seed)       # PyTorch (single GPU)
    torch.cuda.manual_seed_all(seed)   # PyTorch (all GPUs)
    
    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_ds, val_ds, trainloader, valloader = load_data()
    for item in range(2):
        for i, (x, y) in enumerate(trainloader):
            pass
    for item in range(2):
        for i, (x, y) in enumerate(valloader):
            pass
import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', augArgs=None):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality, augArgs)
        self.output_size = (228, 304)

    def train_transform(self, rgb, depth):
        if(self.augArgs.scale_min is None):
            scaleMin = 1.0
        else:
            scaleMin = self.augArgs.scale_min
        if(self.augArgs.scale_max is None):
            scaleMax = 1.5
        else:
            scaleMax = self.augArgs.scale_max
        s = np.random.uniform(scaleMin, scaleMax) # random scaling factor

        if(self.augArgs.varFocus): #Variable focal length simulation
            depth_np = depth
        else:
            depth_np = depth / s #Correct for focal length

        if(self.augArgs.varScale): #Variable global scale simulation
            #Sample a depth group from a gaussian
            idx = np.random.randint(0,len(self.augArgs.scaleMeans),1)
            mean = self.augArgs.scaleMeans[idx]
            variance = self.augArgs.scaleVars[idx]
            scale = np.random.normal(mean,variance,1)
            depth_np = depth*scale

        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

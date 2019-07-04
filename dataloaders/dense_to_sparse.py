import numpy as np
import cv2

def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass

class ProjectiveSampling(DenseToSparse):
    name = "projsam"
    def __init__(self, pixx=114, pixy=152):
        DenseToSparse.__init__(self)
        self.pixx = pixx
        self.pixy = pixy

    def __repr__(self):
        return "%s{ns=%d,md=%f,pixx=%d,pixy=%d}" % (self.name, self.pixx, self.pixxy)

    def dense_to_sparse(self, rgb, depth):
        """
        Applies the projective equations to the depth map to calculate where the physical sensor would
        read a pixel from, then returns a boolean mask with a 1 in this position and zeros elsewhere.
        """
        
        F = 518.86; #The focal length of the camera in pixels (value in NYU toolbox is 518.86 @ 640x480)
        X = -26.75; #The X offset of the ToF from camera in mm (default -26.75mm)
        Y = 26.81; #The Y offset of the ToF from camera in mm (default 26.81mm)
        
        #Correct for resized image
        F = F*(depth.shape[1]/640)
        
        Ox = depth.shape[1]/2
        Oy = depth.shape[0]/2
        
        maxDepth = int(np.amax(depth)*1000)
        depthInc = int(10) #1cm increments
        
        for testDepth in range(depthInc, maxDepth, depthInc):
            #For each possible depth, check if the projection of ToF at that depth is greater than the measured depth
            u = F*X/testDepth + Ox
            v = F*Y/testDepth + Oy
        
            if((u > 2*Ox) or (v > 2*Oy)):
                continue
        
            u = int(u)
            v = int(v)
            
            measuredDepth = 1000*depth[v,u]
            
            #Skip unfilled depth values        
            if(measuredDepth < depthInc):
                continue
            
            if(measuredDepth < testDepth):
                mask = np.zeros((depth.shape), dtype=bool)
                mask[v,u] = True
                return mask
        
        
        print("ERROR: Reached Max Projection Depth")
        return np.zeros((depth.shape), dtype=bool)

class StaticSampling(DenseToSparse):
    name = "statsam"
    def __init__(self, pixx=114, pixy=152):
        DenseToSparse.__init__(self)
        self.pixx = pixx
        self.pixy = pixy

    def __repr__(self):
        return "%s{pixx=%d,pixy=%d}" % (self.name, self.pixx, self.pixy)

    def dense_to_sparse(self, rgb, depth):
        """
        Returns a boolean mask with 1 at [pixy,pixx] and zeros everywhere else
        """
        mask = np.zeros((depth.shape), dtype=bool)
        mask[self.pixy,self.pixx] = True
        return mask

class UniformSampling(DenseToSparse):
    name = "uar"
    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)


class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    # We do not use cv2.Canny, since that applies non max suppression
    # So we simply do
    # RGB to intensitities
    # Smooth with gaussian
    # Take simple sobel gradients
    # Threshold the edge gradient
    # Dilatate
    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

        edge_fraction = float(self.num_samples) / np.size(depth)

        mag = cv2.magnitude(gx, gy)
        min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
        mag_mask = mag >= min_mag

        if self.dilate_iterations >= 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
            cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)

        mask = np.bitwise_and(mag_mask, depth_mask)
        return mask

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
    def __init__(self):
        DenseToSparse.__init__(self)

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

class NearestSampling(DenseToSparse):
    name = "nearsam"
    def __init__(self, pixx=114, pixy=152):
        DenseToSparse.__init__(self)
        self.pixx = pixx
        self.pixy = pixy

    def __repr__(self):
        return "%s{pixx=%d,pixy=%d}" % (self.name, self.pixx, self.pixy)

    def dense_to_sparse(self, rgb, depth):
        """
        Returns a boolean mask with 1 at the nearest depth pixel to [pixy,pixx] 
        which is non zero, and zeros everywhere else
        """
        #The index of non zero points
        nonZeros = np.nonzero(depth) 

        #Squared distance from [pixx,pixy] to non zero elements
        dists = (abs(nonZeros[0]-self.pixy))**2 + (abs(nonZeros[1]-self.pixx))**2
        minDist = np.argmin(dists)

        mask = np.zeros((depth.shape), dtype=bool)
        pixy = nonZeros[0][minDist]
        pixx = nonZeros[1][minDist]

        mask[pixy,pixx] = True
        return mask

class ORBSampling(DenseToSparse):
    name = "orb"

    def __init__(self, num_samples, max_depth=np.inf, apply_kinect_noise=False):
        DenseToSparse.__init__(self, apply_kinect_noise)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.orb_lock = Lock()
        self.orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
        self.n_success = 0
        self.n_failed = 0


    def __repr__(self):
        return "%s{ns=%d,md=%.4f}" % (self.name, self.num_samples,
                                      self.max_depth)

    def depth_mask(self, rgb, depth):
        """
        Use ORB descriptor to determine depth points 
       """
        depth_mask = np.zeros_like(depth, dtype=np.bool)
        # return depth_mask
        _rgb = (rgb*255.0).astype('uint8')

        # self.orb_lock.acquire()
        # gray = rgb2grayscale(rgb)
        kps = self.orb.detect(_rgb)

        if len(kps) == 0:
            self.n_failed += 1
        else:
            self.n_success += 1
            # img2 = cv2.drawKeypoints(_rgb,kps,None,color=(0,255,0), flags=0)
            # plt.imshow(img2),plt.show()
        # print("ORB failed %d times, succeeded %d times" % (self.n_failed, self.n_success))
        # self.orb_lock.release()

        # TODO: randomize keypoints for better training
        # TODO: ORB slam provides around 100 points randomize exact number
        # TODO: add noise on depth estimates
        samples = 0
        for kp in kps:
            c,r = int(kp.pt[0]), int(kp.pt[1])
            if depth[r,c] != 0.0 and depth[r,c] <= self.max_depth:
                depth_mask[r,c] = True
                samples+=1
                if samples == self.num_samples:
                    break
        # if samples < self.num_samples:
        #     print("Less then required samples: %d/%d, %d available" % (samples, self.num_samples, len(kps)))
        #     print("ORB failed %d times, succeeded %d times" % (self.n_failed, self.n_success))
        return depth_mask


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

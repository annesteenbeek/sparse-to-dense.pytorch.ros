import os
import shutil
import threading
import ros_numpy
import torch
import numpy as np
import skimage.transform as transform
import dataloaders.transforms as transforms
from dataloaders.dense_to_sparse import UniformSampling, ORBSampling
from metrics import AverageMeter, Result
from scipy import ndimage
from PIL import Image as PILImage
from PIL import ImageDraw
from scipy.spatial import ConvexHull


# ROS imports
import rospy
import tf2_ros
import tf
import message_filters
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sparse_to_dense.msg import Result as ResultMsg
from sparse_to_dense.msg import SampleMetrics
from sensor_msgs.msg import Image, CameraInfo
from dynamic_reconfigure.server import Server
from sparse_to_dense.cfg import SparseToDenseConfig 


to_tensor = transforms.ToTensor()


def get_result_msg(result, count=1):
    msg = ResultMsg()

    msg.irmse = result.irmse
    msg.imae = result.imae
    msg.mse = result.mse
    msg.rmse = result.rmse
    msg.mae  = result.mae 
    msg.absrel = result.absrel
    msg.lg10 = result.lg10
    msg.delta1 = result.delta1
    msg.delta2 = result.delta2
    msg.delta3 = result.delta3
    msg.data_time = result.data_time
    msg.margin10 = result.margin10
    msg.gpu_time = result.gpu_time
    msg.count = count

    return msg

def get_camera_info_msg(iheight, iwidth, oheight, owidth):
    """ Generates a camera info message, and calculates the new camera
    parameters for the new info message based on resolution change.
    
    """
    # iwidth = 640
    # iheight = 480
    # owidth = 304
    # oheight = 228

    camera_info_msg = CameraInfo()
    camera_info_msg.height = oheight
    camera_info_msg.width = owidth

    # kinect params
    fx, fy = 525, 525
    cx, cy = 319.5, 239.5

    # tello params
    # fx, fy = 922.93, 926.02
    # cx, cy = 472.10, 384.04

    ratiox = owidth / float(iwidth)
    ratioy = oheight / float(iheight)
    fx *= ratiox
    fy *= ratioy
    cx *= ratiox
    cy *= ratioy

    camera_info_msg.K = [fx, 0, cx,
                    0, fy, cy,
                    0, 0, 1]
                        
    camera_info_msg.D = [0, 0, 0, 0]

    camera_info_msg.P = [fx, 0, cx, 0,
                        0, fy, cy, 0,
                        0, 0, 1, 0]

    return camera_info_msg

cvBridge = CvBridge()
def val_transform(img_msg, oheight, owidth):
    img_cv = cvBridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    resize = 240.0/480
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop((oheight, owidth)),
    ])

    if img_cv.ndim == 2: # depth image
        n_depth = np.count_nonzero(img_cv)
        if n_depth < 1000: # if less then n points, must be sparse
            # to prevent sparse point loss, dont use normal resize

            def sparse_resize(img):
                rb, cb = img_cv.shape # big
                rs, cs = int(rb*resize), int(cb*resize) # small
                sh = (rs, int(rb/rs), cs, int(cb/cs)) # new shape 
                return img_cv.reshape(sh).max(-1).max(1)

            transform = transforms.Compose([
                sparse_resize,
                transforms.CenterCrop((oheight, owidth)),
            ])

    img_np = transform(img_cv)
    if img_cv.ndim == 3: # rgb images to floats
        img_np = np.asfarray(img_np, dtype='float') / 255
    # else:
    #     if n_depth < 1000:
    #         n_new = np.count_nonzero(img_np)
    #         print("Sparse point loss, before: {}, after: {}, loss: {} |".format(
    #             n_depth, n_new, n_depth-n_new
    #         ))

    return img_np

def convex_mask(sparse_depth):
    r, c= sparse_depth.shape

    zr, zc = np.nonzero(sparse_depth)

    if len(zr) < 2:
        return sparse_depth

    points = np.stack((zc,zr), axis=-1)
    hull = ConvexHull(points)
    hull_vertice_points = hull.points[hull.vertices].flatten().tolist()

    img = PILImage.new('L', (c,r), 0)
    ImageDraw.Draw(img).polygon(hull_vertice_points, outline=1, fill=1)
    depth_mask = np.array(img, dtype=bool)

    return depth_mask

def region_mask(sparse_depth):
        r = 20
        out_shp = sparse_depth.shape
        X,Y = [np.arange(-r,r+1)]*2
        disk_mask = X[:,None]**2 + Y**2 <= r*r
        Ridx,Cidx = np.where(disk_mask)

        mask = np.zeros(out_shp,dtype=bool)

        maskcenters = np.stack(np.nonzero(sparse_depth), axis=-1)
        absidxR = maskcenters[:,None,0] + Ridx-r
        absidxC = maskcenters[:,None,1] + Cidx-r

        valid_mask = (absidxR >=0) & (absidxR <out_shp[0]) & \
                    (absidxC >=0) & (absidxC <out_shp[1])

        mask[absidxR[valid_mask],absidxC[valid_mask]] = 1

        return mask 

class FrameSaver(object):
    # Used to collect frames for 3D reconstruction in post processing

    def __init__(self, prefix, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return

        self.foldername = prefix
        self.labels = open("%s.txt" % prefix, "w+")
        self.label_lock = threading.Lock()

        self.labels.write('# cnn depth estimation imagesn\n# timestamp filename \n')

        if os.path.exists(self.foldername):
            shutil.rmtree(self.foldername) # remove old folder if it exists
        os.makedirs(self.foldername)

    def save_image(self, img_np, timestamp):
        if not self.enabled:
            return
        save_thread = threading.Thread(target=self._save_image, args=(img_np, timestamp))
        save_thread.start()

    def _save_image(self, img_np, timestamp):
        time_str = "%.6f" % timestamp.to_sec()
        img_name = "{}/{}.png".format(self.foldername, time_str)

        with self.label_lock:
            self.labels.write("{} {}\n".format(time_str, img_name))

        if img_np.ndim == 3: # rgb
            im = PILImage.fromarray((img_np*255).astype(np.uint8)) 
        else: # depth
            im = PILImage.fromarray((img_np*5000).astype(np.uint16)) # depth images are scaled by 5000 e.g. pixel value of 5000 is 1m
        im.save(img_name)

    def close(self):
        if not self.enabled:
            return

        self.labels.close()
        print("Saved frames in {}".format(self.foldername))

class SampleMetricsTracker(object):
    def __init__(self, max_depth=np.inf):
        self.metric_pub = rospy.Publisher('sample_metrics', SampleMetrics, queue_size=5)
        self.max_depth = max_depth
        self.samples = []
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_stde, self.sum_error = 0, 0
        self.count = 0

        self.save_sparse = np.array([])
        self.save_target = np.array([])

    def evaluate(self, sparse_np, target_np):

        depth_valid = sparse_np > 0
        depth_valid = np.bitwise_and(depth_valid, target_np > 0)
        if self.max_depth is not np.inf:
            depth_valid = np.bitwise_and(depth_valid, sparse_np <= self.max_depth)

        n_frame = np.count_nonzero(depth_valid)
        if n_frame <= 0:
            return

        self.count += 1
        self.samples.append(n_frame)
        error = target_np[depth_valid] - sparse_np[depth_valid]
        # error = error[~np.isnan(error)]
        abs_diff = np.absolute(error)
        self.sum_error += error.mean()
        self.sum_mae += abs_diff.mean()
        mse = (abs_diff**2).mean()
        self.sum_mse += mse
        self.sum_rmse += np.sqrt(mse)
        self.sum_stde += np.std(error)

        self.save_sparse = np.append(self.save_sparse, sparse_np[depth_valid])
        self.save_target = np.append(self.save_target, target_np[depth_valid])

        self.publish(n_frame)

    def publish(self, n_frame):
        msg = SampleMetrics()

        msg.max_depth = self.max_depth
        msg.n_frame = int(n_frame)
        msg.n_avg = np.mean(self.samples)
        msg.min = min(self.samples)
        msg.max = max(self.samples)
        msg.stddev = np.std(self.samples)
        msg.mse = self.sum_mse / self.count
        msg.rmse = self.sum_rmse / self.count
        msg.mae = self.sum_mae / self.count
        msg.me = self.sum_error / self.count
        msg.stde = self.sum_stde /self.count
        msg.count = self.count

        self.metric_pub.publish(msg)

    def save(self):
        np.save("sparse_samples.npy", self.save_sparse)
        np.save("target_samples.npy", self.save_target)
        print("saved samples")



class ROSNode(object):

    def __init__(self, model, oheight=228, owidth=304):
        self.model = model


        self.oheight = oheight
        self.owidth = owidth

        self.sparsifier = ORBSampling(num_samples=100, max_depth=5.0)

        self.model.eval()
        self.img_lock = threading.Lock()

        self.est_frame_saver = FrameSaver(prefix="est", enabled=False)
        self.rgb_frame_saver = FrameSaver(prefix="rgb_est", enabled=False)

        target_topic = rospy.get_param("~target_topic", "")
        self.emulate_sparse_depth = rospy.get_param("~emulate_sparse_depth", False)
        self.scale_samples = rospy.get_param("~scale_samples", 20)
        self.frame = rospy.get_param("~frame", "openni_link")
        self.max_depth = rospy.get_param("~max_depth", 2)
        self.rate = rospy.Rate(rospy.get_param("~rate", 10))

        self.depth_est_pub = rospy.Publisher('depth_est', Image, queue_size=5)
        self.sparse_debug_pub = rospy.Publisher('debug/sparse_depth', Image, queue_size=5)
        self.debug_cam_info_pub = rospy.Publisher('debug/camera_info', CameraInfo, queue_size=5)
        self.cam_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=5)
        self.avg_res_pub = rospy.Publisher('average_results', ResultMsg, queue_size=5)
        self.sample_metrics_tracker = SampleMetricsTracker(self.max_depth)

        self.rgb_sub = message_filters.Subscriber('rgb_in', Image)
        self.sparse_sub = message_filters.Subscriber('depth_sparse', Image)
        if target_topic == "":
            self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.sparse_sub], queue_size=20, slop=0.02)
        else:
            self.target_debug_pub = rospy.Publisher('debug/target_depth', Image, queue_size=5)

            self.target_sub = message_filters.Subscriber(target_topic, Image)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.sparse_sub, self.target_sub], queue_size=20, slop=0.02)
        self.ts.registerCallback(self.sync_img_callback)
        
        self.average_meter = AverageMeter()

        self.tf_pub = tf.TransformBroadcaster()

        self.gradient_cutoff = 0.05
        self.config_srv = Server(SparseToDenseConfig, self.config_callback)

        self.scale_ratios = np.array([])
        self.target_scale_ratios = np.array([])
        self.scale_est = None

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)

    def config_callback(self, config, level):
        rospy.loginfo("Gradient cutoff: {gradient_cutoff}".format(**config))
        self.gradient_cutoff = config['gradient_cutoff']
        return config

    def publish_optical_transform(self, time, optical_frame):
        self.tf_pub.sendTransform((0,0,0), 
            tf.transformations.quaternion_from_euler(-np.pi/2, 0, -np.pi/2),
            time,
            optical_frame,
            self.frame)

    def publish_scaled_transform(self, time):
        if self.scale_est == None:
            rospy.logwarn_once("Tried to publish scaled transform without known scale")
            return

        try:
            orb_trans = self.tfBuffer.lookup_transform('world', 'orb_frame', time)
            scaled_translation = (orb_trans.transform.translation.x*self.scale_est,
                                    orb_trans.transform.translation.y*self.scale_est,
                                    orb_trans.transform.translation.z*self.scale_est)
            quaternion = (orb_trans.transform.rotation.x,
                            orb_trans.transform.rotation.y,
                            orb_trans.transform.rotation.z,
                            orb_trans.transform.rotation.w)

            self.tf_pub.sendTransform(scaled_translation, 
                                        quaternion,
                                        time,
                                        self.frame,
                                        'world')

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logerr("TF error: {0}".format(ex))


    def gradient_filter_depth(self, depth_np):
        gx, gy = np.gradient(depth_np)
        gxy = np.hypot(gx,gy)
        depth_np[gxy > self.gradient_cutoff] = np.nan
        return depth_np


    def sync_img_callback(self, rgb_msg, depth_msg, target_msg=None):

        if self.img_lock.acquire(False):
            header = rgb_msg.header 
            if self.scale_ratios.size < self.scale_samples:
                    depth_pred, sparse_debug = self.predict_scale(rgb_msg, depth_msg, target_msg)
            else:
                if self.emulate_sparse_depth:
                    # use rgbd for sparse pointcloud
                    header = target_msg.header
                    depth_pred, sparse_debug = self.emulate_predict_depth(rgb_msg, target_msg)
                else:
                    depth_pred, sparse_debug = self.predict_depth(rgb_msg, depth_msg, target_msg)

            # depth_filtered = self.gradient_filter_depth(depth_pred)
            depth_filtered = depth_pred

            pred_msg = ros_numpy.msgify(Image, depth_filtered, encoding=depth_msg.encoding)
            sparse_debug_msg = ros_numpy.msgify(Image, sparse_debug, encoding=depth_msg.encoding)


            image_frame = header.frame_id
            # image_frame = "openni_rgb_optical_frame"
            pred_msg.header.stamp = header.stamp
            pred_msg.header.frame_id = image_frame

            sparse_debug_msg.header.stamp = header.stamp
            sparse_debug_msg.header.frame_id = header.frame_id

            camera_info_msg = get_camera_info_msg(rgb_msg.height, rgb_msg.width, self.oheight, self.owidth)
            camera_info_msg.header.stamp = header.stamp

            debug_camera_info_msg = get_camera_info_msg(rgb_msg.height, rgb_msg.width, self.oheight, self.owidth)
            # debug_camera_info_msg.header.stamp = header.stamp
            debug_camera_info_msg.header.stamp = header.stamp

            self.depth_est_pub.publish(pred_msg)
            self.cam_info_pub.publish(camera_info_msg)
            self.sparse_debug_pub.publish(sparse_debug_msg)
            self.debug_cam_info_pub.publish(debug_camera_info_msg)

            # for depth target pointcloud
            if target_msg is not None:
                target_np = val_transform(target_msg, self.oheight, self.owidth)
                debug_target_msg = ros_numpy.msgify(Image, target_np, encoding=depth_msg.encoding)
                # debug_target_msg.header.stamp = target_msg.header.stamp
                debug_target_msg.header.stamp = header.stamp
                debug_target_msg.header.frame_id = image_frame

                # debug_camera_info_msg = get_camera_info_msg(target_msg.height, target_msg.width, debug_target_msg.height, debug_target_msg.width)
                debug_camera_info_msg.header.stamp = target_msg.header.stamp

                self.target_debug_pub.publish(debug_target_msg)
                self.debug_cam_info_pub.publish(debug_camera_info_msg)

            if not self.scale_ratios.size < self.scale_samples:
                self.publish_scaled_transform(header.stamp)

            self.publish_optical_transform(header.stamp, image_frame)

            # self.rate.sleep()
            self.img_lock.release()

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.sparsifier.dense_to_sparse(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def predict_scale(self, rgb_msg, sparse_msg, target_msg=None):
        rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
        sparse_np = val_transform(sparse_msg, self.oheight, self.owidth)

        # ignore far away mappoints
        sparse_np[sparse_np > self.max_depth ] = 0

        empty_depth = np.zeros((self.oheight,self.owidth))
        rgbd = np.append(rgb_np, np.expand_dims(empty_depth, axis=2), axis=2)

        input_tensor = to_tensor(rgbd)
        # 4, to emualte batch size 1
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
 
        input_tensor = input_tensor.cuda()
        input_var = torch.autograd.Variable(input_tensor)

        depth_pred = self.model(input_var)
        depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

        # compare known points to predicted scale
        depth_valid = np.where(np.logical_and(sparse_np>0.0, sparse_np<=self.max_depth))

        pixel_ratio = depth_pred_cpu[depth_valid] / sparse_np[depth_valid]
        pixel_ratio = pixel_ratio[~np.isnan(pixel_ratio)]
        est_scale_ratio_median = np.median(pixel_ratio)
        if not np.isnan(est_scale_ratio_median):
            self.scale_ratios = np.append(self.scale_ratios, est_scale_ratio_median)
            self.scale_est = self.scale_ratios.mean()
            print("Frame#: %d \t frame scale median: %.2f \t  avg scale: %.2f" % (self.scale_ratios.size, est_scale_ratio_median, self.scale_est))

        if target_msg is not None: # compare to target scale
            target_np = val_transform(target_msg, self.oheight, self.owidth)
            pixel_ratio = target_np[depth_valid] / sparse_np[depth_valid]
            pixel_ratio = pixel_ratio[~np.isnan(pixel_ratio)]
            target_scale_ratio_median = np.median(pixel_ratio)
            if not np.isnan(target_scale_ratio_median):
                self.target_scale_ratios = np.append(self.target_scale_ratios, target_scale_ratio_median)
                print("est/target scale: %.2f/%.2f" % (self.scale_est, self.target_scale_ratios.mean()))

                self.scale_est = self.target_scale_ratios.mean()

        return depth_pred_cpu, sparse_np 

    def predict_depth(self, rgb_msg, sparse_msg, target_msg=None):
        start_time = rospy.Time.now()
        rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
        sparse_np = val_transform(sparse_msg, self.oheight, self.owidth)
 
        if self.scale_est is not None:
            sparse_np = sparse_np*self.scale_est

        # ignore far away mappoints
        sparse_np[sparse_np > self.max_depth ] = 0

        # use amount of samples the network was trained for
        # n_samples = 200 
        # ni,nj = np.nonzero(sparse_np)
        # if len(ni) > n_samples:
        #     ri = np.random.choice(len(ni), n_samples, replace=False)
        #     inv_mask = np.ones((self.oheight, self.owidth), dtype=np.bool)
        #     inv_mask[ni[ri], nj[ri]] = 0
        #     sparse_np[inv_mask] = 0.0
 
        #     assert len(sparse_np[sparse_np>0.0]) == n_samples, "Not exactly n_points!"
        # elif len(ni) < 5:
        #     rospy.logwarn("Less then 5 points in sparse depth map: %d, returning nothing" % len(ni))
        #     return np.zeros_like(sparse_np), sparse_np
        # else: 
        #     rospy.logwarn("sparse depth map has less then %d points: %d" % (n_samples, len(ni)))
 
        # sparse_np = np.zeros((self.oheight,self.owidth), dtype=np.float32)

        rgbd = np.append(rgb_np, np.expand_dims(sparse_np, axis=2), axis=2)

        input_tensor = to_tensor(rgbd)
        # 4, to emualte batch size 1
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
 
        input_tensor = input_tensor.cuda()
        input_var = torch.autograd.Variable(input_tensor)

        depth_pred = self.model(input_var)

        # add preknown points to prediction
        in_depth = input_tensor[:, 3:, :, :]
        in_valid = in_depth > 0.0
        depth_pred[in_valid] = in_depth[in_valid]

        depth_pred_np = np.squeeze(depth_pred.data.cpu().numpy())

        # rmask= region_mask(sparse_np)
        # cmask = convex_mask(rmask)

        # depth_pred_np[~cmask] = np.nan

        # remove points too far away
        depth_pred_np[depth_pred_np > self.max_depth] = np.nan

        if target_msg is not None:
            data_time = (rospy.Time.now()-start_time).to_sec()
            target_np = val_transform(target_msg, self.oheight, self.owidth)
            target_np[target_np > self.max_depth] = np.nan
            # target_np[~cmask] = np.nan # also remove these points in target to evaluate
            target_tensor = to_tensor(target_np)
            target_tensor = target_tensor.unsqueeze(0)
            self.evaluate_results(depth_pred, target_tensor, data_time)
            self.sample_metrics_tracker.evaluate(sparse_np, target_np)

        self.est_frame_saver.save_image(depth_pred_np, rgb_msg.header.stamp)
        self.rgb_frame_saver.save_image(rgb_np, rgb_msg.header.stamp)

        return depth_pred_np, sparse_np 

    def emulate_predict_depth(self, rgb_msg, target_msg):
        rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
        target_np = val_transform(target_msg, self.oheight, self.owidth)
        input_np = self.create_rgbd(rgb_np, target_np)

        input_tensor = to_tensor(input_np)
        # 4, to emualte batch size 1
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
 
        target_tensor = to_tensor(target_np)
        target_tensor = target_tensor.unsqueeze(0)

        input_tensor, target_tensor = input_tensor.cuda(), target_tensor.cuda()
        input_var = torch.autograd.Variable(input_tensor)

        depth_pred = self.model(input_var)

        self.evaluate_results(depth_pred, target_tensor)

        # add preknown points to prediction
        in_depth = input_tensor[:, 3:, :, :]
        # in_valid = in_depth > 0.0
        # depth_pred[in_valid] = in_depth[in_valid]

        in_depth_np = np.squeeze(in_depth.data.cpu().numpy())
        depth_pred_np = np.squeeze(depth_pred.data.cpu().numpy())

        self.sample_metrics_tracker.evaluate(in_depth_np, target_np)

        self.est_frame_saver.save_image(target_np, rgb_msg.header.stamp)
        self.rgb_frame_saver.save_image(rgb_np, rgb_msg.header.stamp)

        return depth_pred_np, in_depth_np

    def evaluate_results(self, depth_pred, target_tensor, time=0, sparse=None):
        result = Result()
        output = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        target_tensor = target_tensor.unsqueeze(0)
        result.evaluate(output, target_tensor)
        self.average_meter.update(result, 0, time)
        avg_msg = get_result_msg(self.average_meter.average(), self.average_meter.count)
        self.avg_res_pub.publish(avg_msg)

    def run(self):
        rospy.loginfo("Started %s node" % rospy.get_name())
        rospy.spin()
        self.est_frame_saver.close()
        self.rgb_frame_saver.close()
        # self.sample_metrics_tracker.save()
        return

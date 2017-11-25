#!/usr/bin/env python
import argparse
import struct
import sys
import copy
import numpy
import math
import os
import string
import time
import random
import tf
import rospy
import rospkg
import roslib

import cv2;
import cv_bridge 

from sensor_msgs.msg import Image
from moveit_commander import conversions

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)
 
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
 
import baxter_interface

# directory used to save analysis images
image_directory = os.getenv("HOME") + "/Golf/"

class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True, distance):
        global image_directory

        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        
        # image directory
        self.image_dir = image_directory

        # flag to control saving of analysis images
        self.save_images = True

        # required position accuracy in metres
        self.block_tolerance = 0.005

        ## number of blocks found
        #self.blocks_found = 0

        # start positions
        # self.ball_tray_x = 0.50                        # x     = front back
        # self.ball_tray_y = 0.30                        # y     = left right
        # self.ball_tray_z = 0.15                        # z     = up down
        self.block_x = 0.50                        # x     = front back
        self.block_y = 0.00                        # y     = left right
        self.block_z = 0.15                        # z     = up down
        self.roll        = -1.0 * math.pi              # roll  = horizontal
        self.pitch       = 0.0 * math.pi               # pitch = vertical
        self.yaw         = 0.0 * math.pi               # yaw   = rotation

        self.pose = [self.block_x, self.block_y, self.block_z,     \
                     self.roll, self.pitch, self.yaw]

        # camera parameters (NB. other parameters in open_camera)
        self.cam_calib    = 0.0025                     # meters per pixel at 1 meter
        self.cam_x_offset = 0.045                      # camera gripper offset
        self.cam_y_offset = -0.01
        self.width        = 1280                        # Camera resolution
        self.height       = 800

        # colours
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        # # canny image
        # self.canny = np.zeros((self.height, self.width), np.uint8)

        # # Canny transform parameters
        # self.canny_low  = 45
        # self.canny_high = 150

        # block places
        self.block_place = [((0, 0), 0.0)]  #block center (x,y) and angle

        # set speed as a ratio of maximum speed
        self._limb.set_joint_position_speed(0.5)

        # create image publisher to head monitor
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)

        # calibrate the gripper
        self.gripper.calibrate()

        # display the start splash screen
        self.splash_screen("Visual Servoing", "Pick and Place")

        # close all cameras
        self.close_camera("left")
        self.close_camera("right")
        self.close_camera("head")

        # reset cameras
        self.reset_cameras()

        # open required camera
        self.open_camera(self.limb, self.width, self.height)

        # subscribe to required camera
        self.subscribe_to_camera(self.limb)

        # distance of arm to table and ball tray
        self.distance      = distance
        self.block_distance = distance - 0.06

        # move other arm out of harms way
        if limb == "left":
            self.baxter_ik_move("right", (0.25, -0.50, 0.2, math.pi, 0.0, 0.0))
        else:
            self.baxter_ik_move("left", (0.25, 0.50, 0.2, math.pi, 0.0, 0.0))


        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    # camera call back function
    def camera_callback(self, data, camera_name):
        # Convert image from a ROS image message to a CV image
        try:
            self.cv_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
        except cv_bridge.CvBridgeError, e:
            print e

        # 3ms wait
        cv2.waitKey(3)

    # left camera call back function
    def left_camera_callback(self, data):
        self.camera_callback(data, "Left Hand Camera")

    # right camera call back function
    def right_camera_callback(self, data):
        self.camera_callback(data, "Right Hand Camera")

    # head camera call back function
    def head_camera_callback(self, data):
        self.camera_callback(data, "Head Camera")

    # create subscriber to the required camera
    def subscribe_to_camera(self, camera):
        if camera == "left":
            callback = self.left_camera_callback
            camera_str = "/cameras/left_hand_camera/image"
        elif camera == "right":
            callback = self.right_camera_callback
            camera_str = "/cameras/right_hand_camera/image"
        elif camera == "head":
            callback = self.head_camera_callback
            camera_str = "/cameras/head_camera/image"
        else:
            sys.exit("ERROR - subscribe_to_camera - Invalid camera")

        camera_sub = rospy.Subscriber(camera_str, Image, callback)

    # convert Baxter point to image pixel
    def baxter_to_pixel(self, pt, dist):
        x = (self.width / 2)                                                         \
          + int((pt[1] - (self.pose[1] + self.cam_y_offset)) / (self.cam_calib * dist))
        y = (self.height / 2)                                                        \
          + int((pt[0] - (self.pose[0] + self.cam_x_offset)) / (self.cam_calib * dist))

        return (x, y)

    # convert image pixel to Baxter point
    def pixel_to_baxter(self, px, dist):
        x = ((px[1] - (self.height / 2)) * self.cam_calib * dist)                \
          + self.pose[0] + self.cam_x_offset
        y = ((px[0] - (self.width / 2)) * self.cam_calib * dist)                 \
          + self.pose[1] + self.cam_y_offset

        return (x, y)

    # reset all cameras (incase cameras fail to be recognised on boot)
    def reset_cameras(self):
        reset_srv = rospy.ServiceProxy('cameras/reset', std_srvs.srv.Empty)
        rospy.wait_for_service('cameras/reset', timeout=10)
        reset_srv()

    # open a camera and set camera parameters
    def open_camera(self, camera, x_res, y_res):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - open_camera - Invalid camera")

        # close camera
        cam.close()

        # set camera parameters
        cam.resolution          = (int(x_res), int(y_res))
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # open camera
        cam.open()

    # close a camera
    def close_camera(self, camera):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - close_camera - Invalid camera")

        # set camera parameters to automatic
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # close camera
        cam.close()

    # display message on head display
    def splash_screen(self, s1, s2):
        splash_array = np.zeros((self.height, self.width, 3), np.uint8)
        #font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 3.0, 3.0, 9)

        ((text_x, text_y), baseline) = cv2.getTextSize(s1, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
        org = (int((self.width - text_x) / 2), int((self.height / 3) + (text_y / 2)))
        cv2.putText(splash_array, s1, org, cv2.FONT_HERSHEY_SIMPLEX, 3.0,          \
                    (255,255,255), thickness = 7)

        ((text_x, text_y), baseline) = cv2.getTextSize(s2, cv2.FONT_HERSHEY_SIMPLEX, 3,3)
        org = (int((self.width - text_x) / 2), int(((2 * self.height) / 3) + (text_y / 2)))
        cv2.putText(splash_array, s2, org, cv2.FONT_HERSHEY_SIMPLEX, 3,          \
                    (255,255,255), thickness = 7)

        # 3ms wait
        cv2.waitKey(3)

        msg = cv_bridge.CvBridge().cv2_to_imgmsg(splash_image, encoding="bgr8")
        self.pub.publish(msg)

    def baxter_ik_move(self, rpy_pose):
        pose = conversions.list_to_pose(rpy_pose)
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = {'left_w0': 0,
                             'left_w1': 0,
                             'left_w2': 0,
                             'left_e0': 0,
                             'left_e1': 0,
                             'left_s0': 0,
                             'left_s1': 0}
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            # display invalid move message on head display
            self.splash_screen("Invalid", "move")
            # little point in continuing so exit with error message
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x 
        ik_pose.position.y = current_pose['position'].y 
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x 
        ik_pose.orientation.y = current_pose['orientation'].y 
        ik_pose.orientation.z = current_pose['orientation'].z 
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        # self._limb.set_joint_position_speed(0.1)
        self._guarded_move_to_joint_position(joint_angles)
        # self._limb.set_joint_position_speed(0.3)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        # self._limb.set_joint_position_speed(0.1)
        self._guarded_move_to_joint_position(joint_angles)
        # self._limb.set_joint_position_speed(0.3)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()

    # use contour to find block centers and orientations
    def contour_it(self, n_block, iteration):
        #gray scale image of blocks
        gray_image = cv2.cvtColor(self.cv_image[int(y/3):int(2*y/3), int(x/3):int(2*x/3)], cv2.COLOR_BGR2GRAY)

        #find contours
        blurred = cv2.GaussianBlur(gray, (5,5),0)
        thresh = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        blocks = []
        block_image = self.cv_image.copy()
        #check to see if any blocks found
        if cnts is None:
            #display no blocks found message
            sef.splash_screen("no balls", "found")
            sys.exit("ERROR - contour_it - no blocks found")
        for c in cnts:   
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            M = cv2.moments(c)
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
            v1 = box[0]
            box = sorted(box, key=lambda x: np.linalg.norm(x-v1))
            v2 = box[1]
            if v1[1] < v2[1]:
                v1, v2 = v2, v1
            theta = np.arctan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1]))
            cv2.drawContours(block_image[int(y/3):int(2*y/3), int(x/3):int(2*x/3)],[box],0,(0,0,255),2)

            #convert to baxter coordinates
              ## baxter_box = map(self.pixel_to_baxter, box)
            baxter_centre = self.pixel_to_baxter((cX,cY), self.block_distance)
            blocks.append((baxter_centre, theta))

        #display image on monitor
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(block_image, encoding="bgr8")
        self.pub.publish(msg)

        if self.save_images:
            # save image
            file_name = self.image_dir + "contour_" + str(n_block) + "_" + str(iteration) + ".jpg"
            cv2.imwrite(file_name, block_image)

        blocks = sorted(blocks, key=lambda block: block[1][1])
        return blocks

    def locate_blocks(self):
        n_block  = 0
        while n_block < 4:      # no more than 4 blocks
            n_block += 1
            iteration = 0
            angle = 0.0

            # contour to find blocks
            angle

    def move_block(self, block):


def main():
    print("main started")
    rospy.init_node("pick_and_place")
    print("node initited")
    limb = 'left'
    hover_distance = 0.20
    pnp = PickAndPlace(limb, hover_distance)
    gripper_orientation = Quaternion(
                            x = -0.004,
                            y = 1.000,
                            z = 0.013,
                            w = 0.008)
    block_pose = Pose(
                    Point(
                        x = .704,
                        y = 0.121,
                        z = -0.152),
                    gripper_orientation)
    #pnp.move_to_start()
    print("picking")
    pnp.pick(block_pose)
    print("placing")
    pnp.place(block_pose)


if __name__ == '__main__':
    print("hello")
    main()
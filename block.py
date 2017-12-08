#!/usr/bin/env python
import rospy
import roslib

import cv2;
import cv_bridge

import numpy as np
import math
import os
import sys
import string
import time
import random
import tf
from sensor_msgs.msg import Image
import baxter_interface
from moveit_commander import conversions
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header, String
import std_srvs.srv
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

# initialise ros node
rospy.init_node("pick_and_place", anonymous = True)

# directory used to save analysis images
image_directory = os.getenv("HOME") + "/Block/"

# locate class
class locate():
    def __init__(self, arm, distance):
        global image_directory
        # arm ("left" or "right")
        self.limb           = arm
        self.limb_interface = baxter_interface.Limb(self.limb)

        if arm == "left":
            self.other_limb = "right"
        else:
            self.other_limb = "left"

        self.other_limb_interface = baxter_interface.Limb(self.other_limb)

        # gripper ("left" or "right")
        self.gripper = baxter_interface.Gripper(arm)

        # image directory
        self.image_dir = image_directory

        # flag to control saving of analysis images
        self.save_images = True

        # required position accuracy in metres
        self.block_tolerance = 0.003

        # number of blocks found
        self.blocks_found = 0

        # start positions
            # drop off position
        self.block_tray_x = 0.492                        # x     = front back
        self.block_tray_y = -.855                        # y     = left right
        self.block_tray_z = 0.084                        # z     = up down
            # initial block position
        self.block_x = 0.602                        # x     = front back
        self.block_y = -.173                        # y     = left right
        self.block_z = -0.09                        # z     = up down
        self.roll        = -1.0 * math.pi              # roll  = horizontal
        self.pitch       = 0.0 * math.pi               # pitch = vertical
        self.yaw         = 0.0 * math.pi               # yaw   = rotation

        self.pose = [self.block_x, self.block_y, self.block_z,     \
                     self.roll, self.pitch, self.yaw]

        # camera parameters
        self.cam_calib    = 0.0025                     # meters per pixel at 1 meter
        self.cam_x_offset = 0.045                      # camera gripper offset
        self.cam_y_offset = -0.01
        self.width        = 960                        # Camera resolution
        self.height       = 600

        # ROI (region of interest) in image
        self.tray_x_1 = 270                        # x     = front back
        self.tray_x_2 = 835                        # x     = front back
        self.tray_y_1 = 50                        # y     = left right        
        self.tray_y_2 = self.height                        # y     = left right

        # attribute to store images from camera
        self.cv_image = None

        # colours
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        # Enable the robot
        baxter_interface.RobotEnable().enable()

        # set speed
        self.limb_interface.set_joint_position_speed(0.5)
        self.other_limb_interface.set_joint_position_speed(0.5)

        # create image publisher to head monitor
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)

        # calibrate the gripper
        self.gripper.calibrate()

        # display the start splash screen
        self.splash_screen("Visual Servoing", "Pick and Place")

        # subscribe to required camera
        self.subscribe_to_camera(self.limb)

        #Create an instance of the rospy.Publisher object which we can 
        #use to publish messages to a topic. This publisher publishes 
        #messages of type std_msgs/String to the topic /signal_bot
        self.bot_pub = rospy.Publisher('signal_bot', String, queue_size=10)

        # distance of arm to table and block tray
        self.distance      = distance
        self.block_distance = distance - 0.055

        # move other arm out of harms way
        if arm == "left":
            self.baxter_ik_move("right", (0.25, -0.50, 0.2, math.pi, 0.0, 0.0))
        else:
            self.baxter_ik_move("left", (0.25, 0.50, 0.2, math.pi, 0.0, 0.0))

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

    # find next object of interest
    def find_next_block(self, block_data, iteration):
        # if only one object then object found
        if len(block_data) == 1:
            return block_data[0]

        # return the block with the smallest y coordinate
        return min(block_data.items(), key=lambda x: x[1][1])[1]

    # Use contours to find block centres and orientations
    def contour_it(self, n_block, iteration):
        # create gray scale image of blocks
        gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        # blur the image using a Gaussian filter
        blurred = cv2.GaussianBlur(gray_image, (5,5),0)
        # threshold the image to make it binary (white objects on a black background)
        thresh = cv2.threshold(gray_image[self.tray_y_1:self.tray_y_2,self.tray_x_1:self.tray_x_2], 200, 255,cv2.THRESH_BINARY)[1]

        # find contours in opencv 3
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        # find contours in opencv 2
        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        block_image = self.cv_image.copy()
        
        self.splash_screen("detecting", "contours")
        # Check for at least one block found
        if cnts is None:
            #display no blocks found message
            self.splash_screen("no blocks", "found")
            sys.exit("ERROR - contour_it - no blocks found")

        block_data = {}
        n_blocks = 0

        # find blocks from contours
        for c in cnts:
            # find a min area rectangle
            rect = cv2.minAreaRect(c)
            theta = rect[2]
            theta = np.deg2rad(theta)
            # get image moments
            M = cv2.moments(c)
            if (M['m00'] == 0):
                print("zero")
                continue

            # use image moments to calculate the x and y coordinate of the centre of the block
            cX = int(M['m10']/M['m00']) + self.tray_x_1
            cY = int(M['m01']/M['m00']) + self.tray_y_1

            # convert to baxter coordinates
            block = self.pixel_to_baxter((cX, cY), self.block_distance)

            # mark the centre with a green dot
            cv2.circle(block_image,(cX, cY),5,(0,255,0),-1)

            block_data[n_blocks]  = (cX, cY, theta)
            n_blocks            += 1


        # 3ms wait
        cv2.waitKey(3)

        # show image of blocks and centres on head display
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(block_image, encoding="bgr8")
        self.pub.publish(msg)

        if self.save_images:
            # save image of block centres on raw image
            file_name = self.image_dir                                                 \
                      + "block_centre" + str(n_block) + "_" + str(iteration) + ".jpg"
            cv2.imwrite(file_name, block_image)

        # Check for at least one block found
        if n_blocks == 0:                    # no blocks found
            # display no blocks found message on head display
            self.splash_screen("no blocks", "found")
            # less than 12 blocks found, no point in continuing, exit with error message
            sys.exit("ERROR - contour_it - No blocks found")

        # select next block and find it's position
        next_block = self.find_next_block(block_data, iteration)
        angle = next_block[2]
        next_block = next_block[:2]

        # return next block position and pickup angle
        return next_block, angle

    # move a limb
    def baxter_ik_move(self, limb, rpy_pose):
        quaternion_pose = conversions.list_to_pose_stamped(rpy_pose, "base")

        node = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        ik_service = rospy.ServiceProxy(node, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")

        ik_request.pose_stamp.append(quaternion_pose)
        try:
            rospy.wait_for_service(node, 5.0)
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException), error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
            sys.exit("ERROR - baxter_ik_move - Failed to append pose")

        if ik_response.isValid[0]:
            print("PASS: Valid joint configuration found")
            # convert response to joint position control dictionary
            limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
            # move limb
            if self.limb == limb:
                self.limb_interface.move_to_joint_positions(limb_joints)
            else:
                self.other_limb_interface.move_to_joint_positions(limb_joints)
        else:
            # display invalid move message on head display
            self.splash_screen("Invalid", "move")
            # little point in continuing so exit with error message
            print "requested move =", rpy_pose
            sys.exit("ERROR - baxter_ik_move - No valid joint configuration found")

        if self.limb == limb:               # if working arm
            quaternion_pose = self.limb_interface.endpoint_pose()
            position        = quaternion_pose['position']

            # if working arm remember actual (x,y) position achieved
            self.pose = [position[0], position[1],                                \
                         self.pose[2], self.pose[3], self.pose[4], self.pose[5]]

    # update pose in x and y direction
    def update_pose(self, dx, dy):
        x = self.pose[0] + dx
        y = self.pose[1] + dy
        pose = [x, y, self.pose[2], self.roll, self.pitch, self.yaw]
        self.baxter_ik_move(self.limb, pose)

    # used to place camera over block
    def block_iterate(self, n_block, iteration, block_data):
        # print iteration number
        print "block", n_block, "ITERATION ", iteration

        # find displacement of block from centre of image
        pixel_dx    = (self.width / 2) - block_data[0]
        pixel_dy    = (self.height / 2) - block_data[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.block_distance)

        x_offset = - pixel_dy * self.cam_calib * self.block_distance
        y_offset = - pixel_dx * self.cam_calib * self.block_distance

        # update pose and find new block data
        self.update_pose(x_offset, y_offset)
        block_data, angle = self.contour_it(n_block, iteration)

        # find displacement of block from centre of image
        pixel_dx    = (self.width / 2) - block_data[0]
        pixel_dy    = (self.height / 2) - block_data[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.block_distance)

        return block_data, angle, error

    # find all the blocks and place them in the block tray
    def pick_and_place(self):
        n_block = 0
        while True and n_block < 2:              # assume no more than 2 blocks
            n_block          += 1
            iteration        = 0
            angle            = 0.0

            # use Hough circles to find blocks and select one block
            next_block, angle = self.contour_it(n_block, iteration)

            error     = 2 * self.block_tolerance

            print
            print "Block number ", n_block
            print "==============="

            ###########################################################################################
            x1 = self.tray_x_1
            x2 = self.tray_x_2
            y1 = self.tray_y_1
            y2 = self.tray_y_2

            self.tray_x_1 = self.width / 3
            self.tray_x_2 = self.width * 2 / 3
            self.tray_y_1 = self.height / 3
            self.tray_y_2 = self.height *2 / 3

            # iterate to find next block
            # if hunting to and fro accept error in position
            while error > self.block_tolerance and iteration < 10:
                iteration               += 1
                next_block, angle, error  = self.block_iterate(n_block, iteration, next_block)

            self.tray_x_1 = x1
            self.tray_x_2 = x2
            self.tray_y_1 = y1
            self.tray_y_2 = y2
            ##########################################################################################

            s        = "Picking up block"
            ((text_x, text_y), baseline) = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            position = (30, 60)
            cv2.putText(self.cv_image, s, position, cv2.FONT_HERSHEY_SIMPLEX, 3,          \
                    (255,255,255), thickness = 7)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)
            print "DROPPING BLOCK ANGLE =", angle * (math.pi / 180)

            # slow down to reduce scattering of neighbouring blocks
            self.limb_interface.set_joint_position_speed(0.1)

            # move down to pick up block
            pose = (self.pose[0] + self.cam_x_offset - 0.015,
                    self.pose[1] + self.cam_y_offset ,
                    self.pose[2],
                    self.pose[3],
                    self.pose[4],
                    0)
            self.baxter_ik_move(self.limb, pose)

            # move down to pick up block
            pose = (self.pose[0],
                    self.pose[1],
                    self.pose[2] + (0.112 - self.distance),
                    self.pose[3],
                    self.pose[4],
                    self.pose[5])
            self.baxter_ik_move(self.limb, pose)

            # close the gripper
            self.gripper.close()

            s = "Moving to tray"
            ((text_x, text_y), baseline) = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            position = (30, 60)
            cv2.putText(self.cv_image, s, position, cv2.FONT_HERSHEY_SIMPLEX, 3,          \
                    (255,255,255), thickness = 7)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            pose = (self.pose[0],
                    self.pose[1],
                    self.pose[2] + 0.198,
                    self.pose[3],
                    self.pose[4],
                    self.yaw)
            self.baxter_ik_move(self.limb, pose)

            # speed up again
            self.limb_interface.set_joint_position_speed(0.5)

            # display current image on head display
            ((text_x, text_y), baseline) = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            position = (30, 60)
            cv2.putText(self.cv_image, s, position, cv2.FONT_HERSHEY_SIMPLEX, 3,          \
                    (255,255,255), thickness = 7)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)
            # move dowm
            pose = (self.block_tray_x,
                    self.block_tray_y,
                    self.block_tray_z,
                    -3.14,
                    -.029,
                    0.014)
            self.baxter_ik_move(self.limb, pose)

            pose = (self.pose[0],
                    self.pose[1],
                    self.pose[2] -0.10,
                    self.pose[3],
                    self.pose[4],
                    self.pose[5])
            self.baxter_ik_move(self.limb, pose)

            # display current image on head display
            s = "Placing block in block tray"
            cv2.putText(self.cv_image, s, position, cv2.FONT_HERSHEY_SIMPLEX, 3,          \
                    (255,255,255), thickness = 7)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            # open the gripper
            self.gripper.open()

            # prepare to look for next block
            pose = (self.block_x,
                    self.block_y,
                    self.block_z,
                    -1.0 * math.pi,
                    0.0 * math.pi,
                    0.0 * math.pi)
            self.baxter_ik_move(self.limb, pose)

        # display all blocks found on head display
        self.splash_screen("all blocks", "found")

        
        pub_string = "go, %s" % (rospy.get_time())
        # Publish our string to the 'signal_bot' topic
        self.bot_pub.publish(pub_string)

        print "All blocks found"

    # display message on head display
    def splash_screen(self, s1, s2):
        splash_array = np.zeros((self.height, self.width, 3), np.uint8)

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

        msg = cv_bridge.CvBridge().cv2_to_imgmsg(splash_array, encoding="bgr8")
        self.pub.publish(msg)

# read the setup parameters from setup.dat
def get_setup():
    global image_directory
    file_name = image_directory + "setup.dat"

    try:
        f = open(file_name, "r")
    except ValueError:
        sys.exit("ERROR: block_setup must be run before block")

    # find limb
    s = string.split(f.readline())
    if len(s) >= 3:
        if s[2] == "left" or s[2] == "right":
            limb = s[2]
        else:
            sys.exit("ERROR: invalid limb in %s" % file_name)
    else:
        sys.exit("ERROR: missing limb in %s" % file_name)

    # find distance to table
    s = string.split(f.readline())
    if len(s) >= 3:
        try:
            distance = float(s[2])
        except ValueError:
            sys.exit("ERROR: invalid distance in %s" % file_name)
    else:
        sys.exit("ERROR: missing distance in %s" % file_name)

    return limb, distance

def main():
    # get setup parameters
    limb, distance = get_setup()
    print "limb     = ", limb
    print "distance = ", distance

    # create locate class instance
    locator = locate(limb, distance)

    raw_input("Press Enter to start: ")

    # find all the blocks and place them in the block tray
    locator.pose = (locator.block_x,
                    locator.block_y,
                    locator.block_z,
                    locator.roll,
                    locator.pitch,
                    locator.yaw)
    locator.baxter_ik_move(locator.limb, locator.pose)
    locator.pick_and_place()

if __name__ == "__main__":
    main()


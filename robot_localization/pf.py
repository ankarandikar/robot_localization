#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler
from helper_functions import draw_random_sample

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """ 
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

    # TODO: define additional helper functions if needed

class ParticleFilter(Node):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            last_scan_timestamp: this is used to keep track of the clock when using bags
            scan_to_process: the scan that our run_loop should process next
    ####        occupancy_field: this helper class allows you to query the map for distance to closest obstacle
            transform_helper: this helps with various transform operations (abstracting away the tf2 module)
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            thread: this thread runs your main loop
    """
    def __init__(self):
        super().__init__('pf')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 

        self.n_particles = 200          # the number of particles to use

        self.d_thresh = 0.2             # the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # the amount of angular movement before performing an update

        # TODO: define additional constants if needed

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

    def pub_latest_transform(self):
        """ This function takes care of sending out the map to odom transform """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.
            
            You do not need to modify this function, but it is helpful to understand it.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return
        
        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        #print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        #print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:   # if particle cloud is empty, initialize it
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # we have moved far enough to do an update!
            self.update_particles_with_odom()    # update based on odometry
            self.update_particles_with_laser(r, theta)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose based on particles
            self.resample_particles()               # resample particles to focus on areas of high density
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh

    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        # initialize variables to add weighted averages to
        robot_x = 0
        robot_y = 0
        robot_cos = 0
        robot_sin = 0

        for i in range(self.n_particles):   # loop through all particles
            particle = self.particle_cloud[i]

            # take weighted averages by multiplying values by particle weight
            robot_x += particle.x * particle.w
            robot_y += particle.y * particle.w
            robot_cos += np.cos(particle.theta)*particle.w
            robot_sin += np.sin(particle.theta)*particle.w
        
        # set robot pose to weighted averaged x, y, and theta derived from averaged sine and cosine
        self.robot_pose = Particle(x=robot_x, y=robot_y, theta = np.arctan2(robot_sin, robot_cos)).as_pose()

        # update offset of map
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self):
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a tuple (x,y,theta)
            that indicates the change in position and angle between the odometry
            when the particles were last updated and the current odometry.
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:  # wait until we get an x y theta on startup
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])
            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        # convert old and new robot positions into matrices
        new_odom_transform = self.transform_helper.convert_xy_theta_to_transform(new_odom_xy_theta[0],new_odom_xy_theta[1],new_odom_xy_theta[2])
        old_odom_transform = self.transform_helper.convert_xy_theta_to_transform(old_odom_xy_theta[0],old_odom_xy_theta[1],old_odom_xy_theta[2])
        M = np.matmul(np.linalg.inv(old_odom_transform),new_odom_transform) # get transformation matrix from robot position matrices

        # modify particles
        new_particle_cloud = []
        for particle_tuple in self.particle_cloud:
            # convert particle's x, y, theta positions into a matrix
            particle_transform = self.transform_helper.convert_xy_theta_to_transform(particle_tuple.x,particle_tuple.y,particle_tuple.theta)
            # apply transformation matrix
            new_particle = np.matmul(particle_transform,M)
            # convert resulting matrix to new x, y, theta position
            new_tuple = self.transform_helper.convert_transform_to_xy_theta(new_particle)
            # add new position to a local particle cloud list
            new_particle_cloud.append(Particle(x=new_tuple[0], y=new_tuple[1], theta=new_tuple[2], w=particle_tuple.w))
        
        self.particle_cloud = new_particle_cloud    # reassign original particle cloud list to local list

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample in helper_functions.py.
        """
        # make sure the distribution is normalized
        self.normalize_particles()

        # assign particle weights to a new list
        weight_list = []
        for i in range(self.n_particles):
            particle = self.particle_cloud[i]
            weight_list.append(particle.w)
        
        # sample n particles from particle_cloud list with a distribution defined by the weights
        new_particles = draw_random_sample(self.particle_cloud,weight_list,self.n_particles)

        self.particle_cloud = new_particles # reassign original particle cloud list to local list

        self.normalize_particles()  # make sure particles are normalized
        self.add_noise_to_particles()

    def add_noise_to_particles(self):
        """Add noise to the particle positions and orientations.
           The added noise is determined by sampling from a normal distribution centered at 0 (no noise).
        """
        sigma = 0.05    # standard deviation of noise distribution

        # add noise to x, y, and theta
        for p in self.particle_cloud:
            p.x += np.random.normal(0.0,sigma)
            p.y += np.random.normal(0.0,sigma)
            p.theta += np.random.normal(0.0,sigma)

    def update_particles_with_laser(self, r, theta):
        """ Updates the particle weights in response to the scan data
            r: the distance readings to obstacles
            theta: the angle relative to the robot frame for each corresponding reading 
        """
        scan_range = len(r)
        x_values = []
        y_values = []

        # convert scans to cartesian coordinates in robot coordinate frame
        for i in range(scan_range):
            if r[i] != np.inf:  # check for invalid scans (equal to infinity)
                x = r[i]*np.cos(theta[i])
                y = r[i]*np.sin(theta[i])
                x_values.append(x)
                y_values.append(y)

        # get a new weight for each particle
        for n in range(self.n_particles):
            particle = self.particle_cloud[n]

            # convert particle position to matrix
            particle_matrix = self.transform_helper.convert_xy_theta_to_transform(particle.x,particle.y,particle.theta)

            # store the distances between particle scan readings and the closest obstacle
            differences = []
            for j in range(len(x_values)):  # for each laser scan reading
                # get the laser scan reading in the particle coordinate frame
                results = particle_matrix @ np.array([x_values[j],y_values[j],1]) # multiply particle transform by scan coordinates

                # get the distance to the closest obstacle for each scan reading
                distance = self.occupancy_field.get_closest_obstacle_distance(results[0], results[1])

                differences.append(distance)    # list of distances for each laser scan point

            # count the number of "good" distances (within a threshold)
            count = 0
            for k in range(len(differences)):
                if differences[k] < 0.1:    # if the distance is below 0.1
                    count += 1
            particle.w = count
        
        self.normalize_particles()

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is omitted, the odometry will be used """
        if xy_theta is None:    # xy_theta is the robot's pose
            xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.particle_cloud = []

        # set the mean of the normal distributions of x, y, and theta at the robot's position
        x_mean = xy_theta[0]
        y_mean = xy_theta[1]
        theta_mean = xy_theta[2]

        # experimentally set standard deviations
        x_SD = 0.2
        y_SD = 0.2
        theta_SD = 0.05

        initial_weight = 1.0/self.n_particles   # all particles have the same weight

        # select n values of x, y, and theta from the normal distributions
        x_values = np.random.normal(x_mean,x_SD,self.n_particles)
        y_values = np.random.normal(y_mean,y_SD,self.n_particles)
        theta_values = np.random.normal(theta_mean,theta_SD,self.n_particles)

        # create Particles with the x, y, theta values and append to the particle cloud
        for i in range(self.n_particles):
            self.particle_cloud.append(Particle(x_values[i],y_values[i],theta_values[i],initial_weight))
        
        self.normalize_particles()  # make sure particle weights add to 1.0
        self.update_robot_pose()    # update robot pose after initializing particles

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        weight_sum = 0
        for i in range(self.n_particles):
            particle = self.particle_cloud[i]
            weight_sum += particle.w
        normalizer = np.divide(1.0,weight_sum)
        for m in range(self.n_particles):
            particle = self.particle_cloud[m]
            particle.w = particle.w*normalizer

    def publish_particles(self, timestamp):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop 
        if self.scan_to_process is None:
            self.scan_to_process = msg

def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

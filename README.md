# Robot Localization Project
*Anusha Karandikar & Alana MacKay-Kao* <br>
*Introduction to Computational Robotics, Fall 2023*

# Goal
The goal of this project was to learn about and utilize a new algorithm (particle filtering) in order to perform robot localization such that we can locate a robot in a known map and maintain knowledge of its location. Additionally, we wanted to increase our familiarity with ROS and get comfortable with transforming points between different reference frames.
[Watch a live demo of our particle filter here.](https://youtu.be/9u_ldvQo8aA)

# Methodology
Our particle filter works in the following steps:
1. Retrieve the pose of the robot in the odom frame.
2. Initialize the particle cloud: <br>
    Take the robot’s pose, convert it to the `x`, `y`, `theta` format, and use `numpy.random.normal` to draw random samples from a Gaussian distribution around those `x`, `y`, and `theta` values. Use these values to create `n_particles` particles with varying positions and orientations.
3. Transform each particle’s pose according to robot’s motion: <br>
    If the robot has moved enough, create a transformation matrix from its previous pose to its current pose. Loop through the list of particles in `particle_cloud` and apply this transformation matrix to each particle (each particle undergoes the same relative position and orientation change as the robot has, just in its own coordinate frame).
4. Check for good matches: <br>
    Use a transformation matrix for each particle’s pose to project the points of the robot’s laser scans onto each particle’s coordinate frame. Compare the distances between where each particle expects the obstacles in the map to be to the actual obstalces in the map. For each particle, count the number of transformed laser scan points with a difference below a certain threshold to act as the weight. For example, the weight for a particle with no "good" matches is 0. Normalize the weights by dividing them all by the sum of all weights.
5. Update our estimation of the robot’s pose based on our "good" particle poses: <br>
    Multiply the weight of each particle by its `x` value and sum them to get a weighted average of all the `x` values. Do the same for y values. Multiply the particle weights by the cosine and sine of their `theta` values to find the mean `cos(theta)` and `sin(theta)` values. Find the actual average `theta` by taking the inverse tangent of the mean sine and cosine values. Set the new robot pose to the mean `x`, `y`, and `theta` values just found.
6. Resample the particle cloud: <br>
    Sample particles from the `particle_cloud` list with probabilities based on each particle’s weight. Normalize the weights of the new particles and add noise to the particles with another Gaussian distribution so that they are not all in overlapping poses.
After resampling the particle cloud, steps 3-6 are repeated.

# Design Decisions
The most significant design decision we made was to use transformation matrices to update the particle poses instead of the provided delta tuple. While this initially seemed like the more complex option, we found that this streamlined our math with less room for error.

To perform the particle position updates, we found a transformation matrix `M` representing the change from one position (of the robot or particle) to a new position. The matrices were of the form:
<p align="center">
$$
\left(\begin{array}{cc} 
\text{cos(theta)} & \text{-sin(theta)} & \text{x}\\
\text{sin(theta)} & \text{cos(theta)} & \text{y} \\
0 & 0 & 1
\end{array}\right)
$$
</p>

Since the old and new positions of the robot were known due to its on-board odometry, we put the positions in matrix form and found `M` with the following equation:
<p align="center">
$\text{M} = \text{old\_odom\_pose}^{-1} \times \text{new\_odom\_pose}$
</p>

From the known old particle poses, we could then find the new particle poses with:
<p align="center">
$\text{new\_particle\_pose} = \text{old\_particle\_pose} \times \text{M}$
</p>

# Challenges
We faced two primary coding challenges over the course of the project.
1. Our first issue was with trying to understand the different angles at play (i.e., for the robot and particles) and applying the proper trigonometry functions. We encountered this issue when converting the matrices representing the particles’ updated poses back to `x`, `y`, and `theta` form. We did this by indexing the matrix for the `cos(theta)` value and adding `pi` to the inverse cosine of this value. However, this only produced angles in the range 0 to 180 degrees. To fix this, we looked more closely at a unit circle and utilized the sign (positive or negative) of the `sin(theta)` value in the matrix, to see if the `theta` should be in the top or bottom half of the unit circle. If `sin(theta)` was positive, we used `arccos(theta)` to retrieve the `theta` in the first two quadrants of the unit circle without modification; if `sin(theta)` was negative we subtracted `arccos(theta)` from `2*pi` to put the `theta` in the third and fourth quadrants.
2. The second challenge we encountered was understanding the correct approach to resampling the particles. Originally, we simply used the robot’s position with our existing `initialize_particle_cloud` function, thinking that since the robot’s position came from the weighted particle cloud, creating a new cloud centered on the current best guess of the robot’s position would essentially serve the same function as sampling from the particle cloud itself. However, we realized that we wanted to maintain all of the best particle matches – for example, if there are two clusters of particles with high weights in different places, the robot pose resulting from the mean of the particles would be in the middle of the two clusters. However, we want our particles to be in the locations of the best particles, and using the robot pose to create a new cloud would get rid of the valuable information we gained from the two clusters. We realized that resampling from the particles according to a distribution determined by their weights and then adding noise around each one was much more likely to help us maintain good matches.

# Future Improvements
We are pretty happy with the final state of our project. However, one possible improvement we could have implemented is exploring more efficient ways to handle the particles than just for loops. We kept using for loops in our code because it worked consistently well, but it would be interesting to see if some of the loops could be combined or if there are more efficient ways to complete the tasks.

# Lessons Learned
Over the course of this project, we learned that a lot of time will be saved by making sure you know what exactly your overall and immediate goals are. Attempting to work towards a vague end is unhelpful, and often results in code that will need to be rewritten. Instead, coming up with a solid to-do list - which for us, was essentially just our methodology - can be helpful for gauging progress and maintaining motivation. Additionally, we found that it was really helpful to brainstorm what information can be relevant and useful for later applications, because relationships between data can teach you about some other thing you may want to know. For example, in this project, thinking about how information in one frame relates to another frame was immensely helpful.

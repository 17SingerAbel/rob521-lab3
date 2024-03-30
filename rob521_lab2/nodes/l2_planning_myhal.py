#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils_myhal
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from scipy.special import comb

def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.4 #m/s (Feel free to change!)
        self.rot_vel_max = 0.5 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        #Pygame window for visualization
        self.window = pygame_utils_myhal.PygameWindow(
            "Path Planner", (1590, 490), self.occupancy_map.T.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        rand_num = np.random.uniform(0, 1)
        if rand_num < 0.1:
            return self.goal_point

        x = np.random.uniform(self.bounds[0, 0] + self.robot_radius * 2, self.bounds[0, 1] - self.robot_radius * 2)
        y = np.random.uniform(self.bounds[1, 0] + self.robot_radius * 2, self.bounds[1, 1] - self.robot_radius * 2)
        return np.array([x, y]).reshape(2, 1)

    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        # self.nodes store nodes (class Node)
        threshold = 1e-2
        for node in self.nodes:
            if np.linalg.norm(node.point.flatten()[:2] - point.flatten()[:2]) < threshold:
                return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node

        minIndex = -1
        minDistance = np.inf
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            distance = np.linalg.norm(node.point.flatten()[:2] - point.flatten()[:2])
            if distance < minDistance:
                minDistance = distance
                minIndex = i

        return minIndex
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(node_i, vel, rot_vel)

        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        dx = point_s[0] - node_i[0]
        dy = point_s[1] - node_i[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        linear_vel = dist / self.timestep

        # Define the control law
        theta_s = np.arctan2(dy, dx)
        d_theta = (theta_s - node_i[2] + np.pi) % (2 * np.pi) - np.pi
        angular_vel = d_theta / self.timestep

        angular_vel = np.clip(angular_vel, -self.rot_vel_max, self.rot_vel_max)
        linear_vel = np.clip(linear_vel, 0, self.vel_max)
        if abs(d_theta) > np.pi / 2:
            linear_vel *= 0.1

        return linear_vel, angular_vel
    
    def trajectory_rollout(self, node_i, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # based on current robot frame
        p = np.asarray([vel,rot_vel]).reshape(2,1)
        trajectory = np.zeros([3, self.num_substeps+1])
        trajectory[:, 0] = node_i.squeeze()
        substep_size = self.timestep / self.num_substeps 

        for i in range(0, self.num_substeps):
            theta = trajectory[2,i]
            G = np.asarray([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [0, 1]])
            
            q_dot = np.matmul(G,p)
            trajectory[:, i+1] = trajectory[:, i] + q_dot.squeeze()*substep_size

        return trajectory[0:3,1:]
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # align [x, y] points with map origin
        origin = np.asarray(self.map_settings_dict['origin'][:2]).reshape(2,-1)
        resolution = self.map_settings_dict['resolution']
        indices = (point - origin) / resolution # get indices
        # Occupancy map origin is top left, but map points origin bottom left
        indices[1,:] = self.map_shape[0]-indices[1,:] 
        return np.vstack([indices[1,:],indices[0,:]]).astype(int)
    
    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        
        row_indices = []
        col_indices = []
        
        resolution = self.map_settings_dict['resolution']
        centers = self.point_to_cell(points)

        radius = np.ceil(self.robot_radius / resolution).astype(int)

        for i in range(centers.shape[1]):
            rr, cc = disk(centers[:,i],radius,shape=self.map_shape)
            row_indices.extend(rr)
            col_indices.extend(cc)
        return row_indices, col_indices

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
     
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        # Lab 2 Code Starts
        linear_vel, angular_vel = self.robot_controller(node_i, point_f)
        trajectory = self.trajectory_rollout(node_i, linear_vel, angular_vel)
        return trajectory
    
    def is_point_within_trajectory(self, trajectory_o, point):
        for i in range(trajectory_o.shape[1]): 
            if np.linalg.norm(trajectory_o[:2,i].flatten() - point[:2, :].flatten()) < 1e-2:
                return True 
        return False
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        #print("TO DO: Implement a cost to come metric")
        # Lab 2 Code Starts
        # Euclidean Distance
        traj_cost = 0
        for i in range(1, trajectory_o.shape[1]):
            traj_cost = np.linalg.norm(trajectory_o[:2,i-1].flatten() - trajectory_o[:2,i].flatten())
        return traj_cost
        # Lab 2 Code Ends
    
    def update_children(self, node_id, cost_change):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        #print("TO DO: Update the costs of connected nodes after rewiring.")
        # Lab 2 Code Starts
        queue = self.nodes[node_id].children_ids
        visited = []
        while len(queue) > 0:
            curr_node_id = queue.pop(0)
            if curr_node_id not in visited:
                self.nodes[curr_node_id].cost -= cost_change
                visited.append(curr_node_id)
                for child_id in self.nodes[curr_node_id].children_ids:
                    if child_id not in visited:
                        queue.append(child_id)
            
        # Lab 2 Code Ends
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        self.window.add_point(self.goal_point.squeeze(),radius=2,color=(0,255,255))
        while True: #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            if self.check_if_duplicate(point):
                continue
            #Get the closest point
            closest_node_id = self.closest_node(point)
            #Simulate driving the robot towards the closest point 3 x N
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            new_pt = trajectory_o[:, -1].reshape(3, 1)
            # Lab 2 Code Starts
            # Check for collisions
            row_indices, col_indices = self.points_to_robot_circle(trajectory_o[:2,:])
            if np.all(self.occupancy_map[row_indices, col_indices]):
                # upade self.nodes
                # new_pt = trajectory_o[:, -1].reshape(3, 1) + self.nodes[closest_node_id].point

                if not self.check_if_duplicate(new_pt):
                    new_node = Node(new_pt , closest_node_id, 0)
                    self.nodes.append(new_node)
                    
                    self.window.add_point(np.copy(new_pt[:2].squeeze()), radius=2,color=(255,0,0))
                    self.window.add_line(np.copy(new_pt[:2].squeeze()), np.copy(self.nodes[closest_node_id].point[:2].squeeze()))
                    
            #Check if goal has been reached
            if np.linalg.norm(new_pt[:2] - self.goal_point) < self.stopping_dist:
                print("Reached the goal.")
                return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot       
        while True: #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            new_pt = trajectory_o[:, -1].reshape(3, 1)
            #Check for Collision
            # Lab 2 Code Starts
            new_node = None
            row_indices, col_indices = self.points_to_robot_circle(trajectory_o[:2,:])
            if np.all(self.occupancy_map[row_indices, col_indices]):
                if not self.check_if_duplicate(new_pt):
                    traj_cost = self.cost_to_come(trajectory_o)
                    new_node = Node(new_pt , closest_node_id, self.nodes[closest_node_id].cost + traj_cost)
                    self.nodes.append(new_node)
                    self.window.add_point(np.copy(new_node.point[:2].squeeze()),radius=2,color=(255,0,0))
                    self.window.add_line(np.copy(new_node.point[:2].squeeze()), np.copy(self.nodes[closest_node_id].point[:2].squeeze()))
            if new_node is None:
                continue
            
            new_node_id = len(self.nodes) - 1
            #Last node rewire
            radius = self.ball_radius()

            for node_id, node in enumerate(self.nodes[:-1]):
                distance = np.linalg.norm(node.point.flatten()[:2] - new_node.point.flatten()[:2])
                if distance <= radius:
                    mini_trajectory = self.connect_node_to_point(node.point, new_node.point)
                    if not self.is_point_within_trajectory(mini_trajectory, new_node.point):
                        continue
                    potential_new_cost = node.cost + self.cost_to_come(mini_trajectory)
                    if potential_new_cost < new_node.cost:
                        r_indices, c_indices = self.points_to_robot_circle(mini_trajectory[:2,:])
                        if np.all(self.occupancy_map[r_indices, c_indices]):
                            new_node.parent_id = node_id
                            node.children_ids.append(new_node_id)
                            new_node.cost = potential_new_cost
                            self.window.add_line(np.copy(new_node.point[:2].squeeze()), np.copy(self.nodes[closest_node_id].point[:2].squeeze()), color=(255, 255, 255))
                            self.window.add_line(np.copy(new_node.point[:2].squeeze()), np.copy(self.nodes[node_id].point[:2].squeeze()), color=(0, 0, 0))


            #Close node rewire
            # find neighbour with radius
            for node_id, node in enumerate(self.nodes[:-1]):
                if node_id == new_node.parent_id:
                    continue
                distance = np.linalg.norm(node.point[:2] - new_node.point[:2])
                if distance <= radius:
                    mini_trajectory = self.connect_node_to_point(new_node.point, node.point)
                    if not self.is_point_within_trajectory(mini_trajectory, node.point):
                        continue
                    potential_new_cost = new_node.cost + self.cost_to_come(mini_trajectory)
                    if potential_new_cost < node.cost:
                        rr, cc = self.points_to_robot_circle(mini_trajectory[:2,:])
                        if np.all(self.occupancy_map[rr, cc]):
                            self.window.add_line(np.copy(node.point[:2].squeeze()), np.copy(self.nodes[node.parent_id].point[:2].squeeze()), color=(255, 255, 255))
                            node.parent_id = new_node_id
                            new_node.children_ids.append(node_id)
                            cost_change = node.cost - potential_new_cost
                            node.cost= potential_new_cost
                            self.update_children(node_id, cost_change)
                            self.window.add_line(np.copy(node.point[:2].squeeze()), np.copy(new_node.point[:2].squeeze()))
            #Check for early end
            if np.linalg.norm(new_pt[:2] - self.goal_point) < self.stopping_dist:
                print("Reached the goal.")
                return self.nodes
            # Lab 2 Code Ends
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "myhal.png"
    map_setings_filename = "myhal.yaml"

    #robot information
    goal_point = np.array([[7], [0]]) #m
    stopping_dist = 5e-2 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

     # #RRT Test
    start_time = time.time()
    # nodes = path_planner.rrt_planning()
    nodes = path_planner.rrt_star_planning()
    end_time = time.time()
    print(f'Time to goal: {end_time-start_time}')
    node_path_metric = np.hstack(path_planner.recover_path())
    np.save("shortest_path_myhal.npy", node_path_metric)
    for i in range(0, node_path_metric.shape[1]):
        print(node_path_metric[:, i])
        path_planner.window.add_point(node_path_metric[:2,i], radius=5, color=(0, 0, 255))
        
    # Keep pygame running
    while True:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

if __name__ == '__main__':
    main()

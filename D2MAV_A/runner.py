import numpy as np
import ray
import numba as nb
from bluesky.tools import geo
from shapely.geometry import LineString, Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import nearest_points
import math
from gym.spaces import Discrete, Box
import os
import random
import utm
from pyproj import Transformer
import glob
import gin
from copy import copy
import yaml
from itertools import groupby
import time
import json

from inspect import currentframe
from timeit import default_timer as timer

# # Auto ATC Setup items
from D2MAV_A.qatc import TrafficManager, VehicleHelper, load_routes, BadLogic

# Load traffic manager configuration file
FILE_PREFIX = str(os.path.dirname(__file__))
TOWER_CONFIG_FILE = FILE_PREFIX + "/DFW_towers.yaml"
with open(TOWER_CONFIG_FILE, "r") as f:
    tower_config = yaml.load(f, Loader=yaml.Loader)
# Load some route data
import pickle

# with open("linestring_dict.pkl", 'rb') as file:
#     route_linestrings = pickle.load(file)


with open("new_route_data.pkl", "rb") as file:
    route_data = pickle.load(file)

## Limit GPU usage
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


## TODO: Move to a different file like "util/" or something similar
@nb.njit()
def discount(r, discounted_r, v, done, gae):
    """Compute the gamma-discounted rewards over an episode."""
    for t in range(len(r) - 1, -1, -1):
        if done[t] or t == (len(r) - 1):
            delta = r[t] - v[t][0]
            gae[t] = delta

        else:
            delta = r[t] + 0.95 * v[t + 1][0] - v[t][0]
            gae[t] = delta + 0.95 * 0.95 * gae[t + 1]

        discounted_r[t] = gae[t] + v[t][0]

    return discounted_r


## Checks the feasibility of the generated route based on
## specified threshold
def checkPoint(x1, y1, x2, y2, ls, threshold):
    for j in range(len(ls)):
        old_ls = ls[j]
        start, end = list(old_ls.coords)

        x_1, y_1 = start
        xe_1, ye_1 = end

        # distance start to old LS start
        dx = np.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2) / geo.nm

        # distance start to old LS end
        dx_1 = np.sqrt((xe_1 - x1) ** 2 + (ye_1 - y1) ** 2) / geo.nm

        # distance end to old LS end
        dx_2 = np.sqrt((xe_1 - x2) ** 2 + (ye_1 - y2) ** 2) / geo.nm

        # distance end to old LS start
        dx_3 = np.sqrt((x_1 - x2) ** 2 + (y_1 - y2) ** 2) / geo.nm

        dist = np.array([dx, dx_1, dx_2, dx_3])

        # feasible route
        if any(dist <= threshold):
            return False
    return True


@ray.remote
class Runner(object):
    import tensorflow as tf
    import bluesky as bs

    """
        Worker agent. Runs the BlueSky sim within its own process. This agent
        collects the experience and sends to the scheduler/trainer Worker

    """

    def __init__(
        self,
        actor_id,
        agent,
        max_steps=1024,
        speeds=[5, 0, 220],
        simdt=1,
        bsperf="openap",
        scenario_file=None,
        working_directory=None,
        LOS=10,
        dGoal=100,
        maxRewardDistance=100,
        intruderThreshold=750,
        rewardBeta=[0.001],
        rewardAlpha=[0.1],
        speedChangePenalty=[0.001],
        rewardLOS=[-1],
        stepPenalty=[0],
        clearancePenalty=0.005,
        config_file=None,
        gui=False,
        non_coop_tag=0,
        traffic_manager_active=True,
        run_type="train",
        n_waypoints=2,
    ):
        self.id = actor_id

        self.tf.config.threading.set_intra_op_parallelism_threads(1)
        self.tf.config.threading.set_inter_op_parallelism_threads(1)
        self.tf.compat.v1.logging.set_verbosity(self.tf.compat.v1.logging.ERROR)

        self.agents = {}
        self.agent = agent
        self.scen_file = scenario_file
        self.working_directory = working_directory
        self.speeds = np.array(speeds)
        self.altitudes = np.array([-500,0,500])
        self.max_steps = max_steps
        self.simdt = simdt
        self.bsperf = bsperf
        self.step_counter = 0
        self.LOS = LOS
        self.dGoal = dGoal
        self.maxRewardDistance = maxRewardDistance
        self.intruderThreshold = intruderThreshold
        self.rewardBeta = rewardBeta
        self.rewardAlpha = rewardAlpha
        self.speedChangePenalty = speedChangePenalty
        self.rewardLOS = rewardLOS
        self.stepPenalty = stepPenalty
        self.clearancePenalty = clearancePenalty
        
        ### Altitude and Noise Reward Parameters
        self.altChangePenalty = speedChangePenalty
        self.speed_change_counter = 0
        self.alt_change_counter = 0
        self.min_alt = 1000                    # 800
        self.max_alt = 3000                   # 2500 - 3000
        self.noise_const_a = -14.3088
        self.noise_const_b = 103.4767

        self.alpha = 0
        self.beta = 1 - self.alpha
        self.gamma = 0.01
        self.safety_train = True
        self.noise_train = False
        self.alt_changing = {}
        # print("Noise Reward Params: ", self.alpha, self.beta, self.gamma)
        self.max_noise_increase = 0
        self.average_noise_increase = []
        self.a_0 = 88.09
        self.a_1 = 3.21
        self.a_2 = -2.62

        with open('D2MAV_A/ambient_noise_dict.json', 'r') as file:
            self.ambient_noise_level = json.load(file)


        self.gui = gui
        self.traffic_manager_active = traffic_manager_active
        self.run_type = run_type
        self.n_waypoints = n_waypoints
        self.vls_active = False
        self.intersection_radius = 2700
        self.nmac_offset = 3 * 86
        self.nmac_distance = self.LOS + self.nmac_offset

        if not "SIMDT" in os.environ.keys():
            os.environ["SIMDT"] = "{}".format(self.simdt)

        ## building episode specific parameters not configured by config.gin
        self.dones = []

        self.episode_done = True  ## initialization

        self.epsg_proj = "epsg:2163"
        self.epsg_from = "epsg:4326"
        self.transformer = Transformer.from_crs(
            self.epsg_from, self.epsg_proj, always_xy=True
        )

        self.timer = 0
        self.num_ac = 0
        self.counter = 0

        self.action_key = {}

        self.min_x = 281134.8350222109  # 686785.5111184405 #np.inf
        self.max_x = 332359.3446274982  # 737690.7518448773 #-np.inf
        self.min_y = -1352500.1522055818  # 3627144.8191298996 #np.inf
        self.max_y = -1306410.5905290868  # 3673125.271272543 #-np.inf
        self.tas_min = np.round(
            self.speeds[0] * geo.nm / 3600, 4
        )  # converting knots to m/s
        self.tas_max = np.round(
            self.speeds[2] * geo.nm / 3600, 4
        )  # converting knots to m/s
        self.ax_min = -3.5
        self.ax_max = 3.5
        self.max_d = 46726.453433800954  # 0

        # Non-cooperative
        self.non_coop_tag = non_coop_tag  # 0 for cooperative. 1 for Loos of Control. 2 for loss of communication.
        self.LControl_lst = ["PNHWL0"]
        self.LComm_lst = ["PI30L0"]

        # Added Modification for Altitude Adjustments
        self.speed_only = False
        if self.speed_only:
            self.action_dim = 3 # (speed up, ascend), (speed up, maintain) ...
            self.ownship_obs_dim = 6 + self.n_waypoints * 2
            self.intruder_obs_dim = 8 + self.n_waypoints * 2
        else:
            self.action_dim = 3
            self.speed_dim = 0 # speed up, maintain, slow down
            self.alt_dim = 3 # ascend, maintain, descend
            self.ownship_obs_dim = 4 # + self.n_waypoints * 2
            self.intruder_obs_dim = 3 # + self.n_waypoints * 2

        self.action_space = Discrete(self.action_dim)

        # Initialize Traffic Manager
        self.create_traffic_manager()
        if self.speed_only:
            if self.traffic_manager_active:
                self.ownship_obs_dim += 2

        if self.gui:
            self.bs.init(
                mode="sim", configfile=self.working_directory + "/" + config_file
            )
            self.bs.net.connect()

        else:
            self.bs.init(
                mode="sim",
                detached=True,
                configfile=self.working_directory + "/" + config_file,
            )

        self.agent.initialize(
            self.tf, self.ownship_obs_dim, self.intruder_obs_dim, self.action_dim
        )

    def create_traffic_manager(self):
        route_linestrings = {}
        for route_id, gps_wp_list in route_data.items():
            rtemp = []
            for item in gps_wp_list:  # item is a tuple of (lon, lat)
                x, y = self.transformer.transform(item[0], item[1])
                rtemp.append((x, y))
            route_linestrings[route_id] = LineString(rtemp)
        self.traffic_manager = TrafficManager(tower_config)
        self.vehicle_helpers = {}  # Store vehicle helpers
        self.routes_loaded = load_routes(
            tower_config, self.traffic_manager, route_linestrings
        )  # Odd name to make sure it doesn't clash
        self.pending_requests = (
            []
        )  # TODO: Make this a dict and somehow store number of requests made
        self.pending_initial_requests = (
            []
        )  # TODO: Make this a dict and somehow store number of requests made
        self.exiting_vehicles = []

    def reset(self):
        """
        Beginning of the episode. In this function, all variables need to be reset to default.
        """

        self.agent.reset()

        self.timer = 0
        self.num_ac = 0
        self.counter = 0
        self.episode_done = False
        self.step_counter = 0
        self.time_without_traffic = 0
        self.dones = []
        self.acInfo = {}
        self.file_keeper = []
        collected_responses = {}

        # randomly sample
        if ".scn" not in self.scen_file:
            scenario_files = glob.glob(f"{self.scen_file}" + "/*.scn")
            scenario_file = np.random.choice(scenario_files, 1)[0]

        else:
            scenario_file = self.scen_file

        # Reset Traffic Manager
        self.create_traffic_manager()  # easier to just create a new one for now
        # TODO: Implement proper reset
        # self.traffic_manager.reset()

        # Starting the bluesky and sim
        self.bs.stack.stack("IC " + self.working_directory + "/" + scenario_file)
        self.bs.stack.stack("FF")
        self.bs.sim.step()  # bs.sim.simt = 0.0 AFTER the call to bs.sim.step()
        self.bs.stack.stack("FF")
        if self.speed_only:
            ownship_min_state = [0, self.tas_min, self.ax_min, 0, 0, 0]
            ownship_max_state = [self.max_d, self.tas_max, self.ax_max, 360, self.max_d, 2]

            intruder_min_state = [
                self.min_x - self.max_x,
                self.min_y - self.max_y,
                0,
                self.tas_min,
                self.ax_min,
                0,
                0,
                0,
            ]

            intruder_max_state = [
                self.max_x - self.min_x,
                self.max_y - self.min_y,
                self.max_d,
                self.tas_max,
                self.ax_max,
                360,
                self.max_d,
                2,
            ]
        else:
            # Adding Altitude State information (Current Alt, Ambient Noise) to end of state
            ownship_min_state = [0, 0, 40, 0]
            ownship_max_state = [2, self.max_alt, 60, 1]

            intruder_min_state = [
                # self.min_x - self.max_x,
                # self.min_y - self.max_y,
                # 0,
                # self.tas_min,
                # self.ax_min,
                # 0,
                0,
                0,
                0
            ]

            intruder_max_state = [
                # self.max_x - self.min_x,
                # self.max_y - self.min_y,
                # self.max_d,
                # self.tas_max,
                # self.ax_max,
                # 360,
                self.max_d,
                2,
                self.max_alt
            ]

        # clearance denied, cleared, no clearance request
        if self.speed_only:
            if self.traffic_manager_active:
                ownship_min_state += [0, 0]
                ownship_max_state += [2, self.max_d]
            ownship_min_state += [-180, 0] * self.n_waypoints
            ownship_max_state += [180, self.max_d] * self.n_waypoints
            intruder_min_state += [-180, 0] * self.n_waypoints
            intruder_max_state += [180, self.max_d] * self.n_waypoints

        ## rel utm position, dist goal, speed, acceleration, heading, distance ownship to intruder, distance intruder intersection, distance ownship to intersection
        self.context_space = Box(
            np.array(intruder_min_state),
            np.array(intruder_max_state),
            dtype=np.float64,
        )

        self.observation_space = Box(
            np.array(ownship_min_state), np.array(ownship_max_state), dtype=np.float64
        )

        ## This is a catch to make sure the time between bluesky sim steps is 1 seconds
        # Should 1 second be changed to something smaller like 0.1?
        before = self.bs.sim.simt
        self.bs.sim.step()
        after = self.bs.sim.simt
        if (after - before) == 0:
            return self.reset()

        assert (after - before) == self.simdt

        ## The first self.bs.sim.step() spawns in the initial aircraft. Sim time should be at t = 12 seconds now
        self.step_counter += 1

        response = {}
        if self.traffic_manager_active:
            # First time the auto atc system is pinged
            initial_requests = []
            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID

                # Check if the current ID exists. If not then create a new vehicle helper
                if not id_ in self.vehicle_helpers.keys():
                    # Get and reformat the route name coming from Bluesky
                    route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                    self.vehicle_helpers[id_] = VehicleHelper(
                        id_, self.routes_loaded[route_name]
                    )
                    # Add initial request to enter the system
                    initial_requests.append(id_)

            # Pass collected requests to the Traffic Manager and process them
            for id_ in initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()  # tuple
                self.traffic_manager.add_request(id_, formatted_request)

            if initial_requests:
                initial_request_response = self.traffic_manager.process_requests()
            else:
                response = None
                initial_request_response = {}
                print("No initial requests")

            collected_responses = {}
            for id_, response in initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].next_intersection == None:
                        self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].route.route_id[2:4]
                    # print(f"{id_} cleared for departure at time {self.bs.sim.simt}")
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
                    self.pending_initial_requests.append(id_)
                    k_idx = self.bs.traf.id2idx(id_)
                    self.bs.traf.ap.setclrcmd(
                        k_idx, False
                    )  # set the clearance to False (i.e., denied and hold on ground)

        state, _, _, _ = self.state_update(
            self.bs.traf, init=True, tm_response=collected_responses
        )

        if self.gui:
            self.bs.net.update()

        return state

    def step(self, actions, policy, value):
        """
        Update the environment with the actions from the agents
        """

        collected_responses = {}

        for ac_id in actions.keys():
            if actions[ac_id] == -1:
                """Uncomment if running unequiped"""
                if ac_id in self.action_override:
                    # print("Overriding action for ", ac_id)
                    speed = 0
                else:
                    speed = 40
                self.bs.stack.stack("{} SPD {}".format(ac_id, speed))
                continue
            if self.speed_only:
                speed = self.speeds[actions[ac_id]]
                new_alt = 1000
                if actions[ac_id] == 1:  # hold
                    # Convert current speed in m/s to knots
                    speed = int(
                        np.round(
                            (self.bs.traf.cas[self.bs.traf.id2idx(ac_id)] / geo.nm) * 3600
                        )
                    )
            else:
                speed = self.speeds[2]
                # print("Calculated Speed: ", ac_id, actions[ac_id], self.speeds[int(actions[ac_id] / 3)])
                alt_change = self.altitudes[actions[ac_id]]
                current_alt = self.meters_to_feet(self.bs.traf.alt[self.bs.traf.id2idx(ac_id)])
                new_alt = current_alt + alt_change
                if int(actions[ac_id] / 3) == 1:  # hold
                    # Convert current speed in m/s to knots
                    speed = int(
                        np.round(
                            (self.bs.traf.cas[self.bs.traf.id2idx(ac_id)] / geo.nm) * 3600
                        )
                    )
            if self.vls_active:
                n_ac = self.bs.traf.lat.shape[0]
                d = (
                geo.kwikdist_matrix(
                    np.repeat(self.bs.traf.lat, n_ac),
                    np.repeat(self.bs.traf.lon, n_ac),
                    np.tile(self.bs.traf.lat, n_ac),
                    np.tile(self.bs.traf.lon, n_ac),
                ).reshape(n_ac, n_ac)
                * geo.nm)
                nmac_flag = False
                # print("Checking if Shielding is nessesary: ", ac_id, self.vehicle_helpers[ac_id].within_intersection)
                if not self.vehicle_helpers[ac_id].within_intersection:
                    for other_id in self.bs.traf.id:
                        if ac_id == other_id:
                            continue
                        i_idx = self.bs.traf.id2idx(ac_id)
                        j_idx = self.bs.traf.id2idx(other_id)
                        if d[i_idx][j_idx] > self.intersection_radius:
                            continue

                        # Same Route Shielding Logic
                        route_i = self.bs.traf.ap.route[i_idx]
                        route_j = self.bs.traf.ap.route[j_idx]
                        next_wp_i = route_i.wpname[route_i.iactwp]
                        next_wp_j = route_j.wpname[route_j.iactwp]
                        next_wp_i_2 = None
                        next_wp_j_2 = None
                        if route_i.iactwp != len(route_i.wpname) - 1:
                            next_wp_i_2 = route_i.wpname[route_i.iactwp + 1]
                        if route_j.iactwp != len(route_j.wpname) - 1:
                            next_wp_j_2 = route_j.wpname[route_j.iactwp + 1]

                        if self.bs.traf.active[i_idx] and self.bs.traf.active[j_idx]:
                            if next_wp_j == next_wp_i:
                                dist_i = self.bs.traf.ap.dist2wp[i_idx] 
                                dist_j = self.bs.traf.ap.dist2wp[j_idx] 
                                
                                if dist_i != -1 and dist_j != -1 and dist_j < dist_i: #self.bs.traf.distflown[i_idx] < self.bs.traf.distflown[j_idx]:
                                    if d[i_idx][j_idx] < self.nmac_distance:
                                        if self.bs.traf.tas[j_idx] == 0:
                                            self.action_override.append(ac_id)
                                        else:
                                            if d[i_idx][j_idx] < self.LOS:
                                                self.action_override.append(ac_id)
                                            else:
                                                speed = self.speeds[0]
                            
                            if  next_wp_i_2 != None and (next_wp_i_2 == next_wp_j):
                                if d[i_idx][j_idx] < self.nmac_distance:
                                    if self.bs.traf.tas[j_idx] == 0:
                                        self.action_override.append(ac_id)
                                    else:
                                        if d[i_idx][j_idx] < self.LOS:
                                            self.action_override.append(ac_id)
                                        else:
                                            speed = self.speeds[0]
                            
                            if ac_id[1:5] == other_id[1:5]:
                                if self.bs.traf.distflown[i_idx] < self.bs.traf.distflown[j_idx]:
                                    if d[i_idx][j_idx] < self.nmac_distance:
                                        if self.bs.traf.tas[j_idx] == 0:
                                            self.action_override.append(ac_id)
                                        else:
                                            if d[i_idx][j_idx] < self.LOS:
                                                self.action_override.append(ac_id)
                                            else:
                                                speed = self.speeds[0]
            if ac_id in self.action_override:
                print("Overriding action for ", ac_id, self.vls_active)
                speed = 0
            # print("Speed Command: ", "{} SPD {}".format(ac_id, speed))
            self.bs.stack.stack("{} SPD {}".format(ac_id, speed))
            if new_alt < self.min_alt:
                new_alt = self.min_alt
            if new_alt > self.max_alt:
                new_alt = self.max_alt
            # print("Checking actions: ", ac_id, current_alt, new_alt)
            # Allow actions to persist
            if ac_id in self.alt_changing.keys():
                new_alt = self.alt_changing[ac_id]
            else:
                if current_alt < self.min_alt:
                    self.alt_changing[ac_id] = self.min_alt
                else:
                    self.alt_changing[ac_id] = round(new_alt)
            
            # print("Checking actions post: ", ac_id, current_alt, new_alt)
            
            self.bs.stack.stack("{} ALT {}".format(ac_id, new_alt))
            # print(ac_id, self.bs.traf.active[self.bs.traf.id2idx(ac_id)])

        # updates the bluesky environment by 1 simulation timestep (1 seconds)
        self.bs.sim.step()
        self.step_counter += 1

        if self.gui:
            self.bs.net.update()

        for ac_id in list(self.alt_changing.keys()):
            idx = self.bs.traf.id2idx(ac_id)
            if self.alt_changing[ac_id] == round(self.meters_to_feet(self.bs.traf.alt[idx])):
                # print("Removing ", ac_id, self.alt_changing)
                del self.alt_changing[ac_id]
        
        ### Start of traffic manager code ###
        if self.traffic_manager_active:
            new_requests = []  # requests from vehicles that are in the system
            initial_requests = []  # requests from vehicles that are not in the system

            """ Use the traffic object to determine the state of all intersections and route sections.
                1) Determine which vehicles are currently inside which intersections
                2) Determine which vehicles have transitioned:
                    a) into an intersection
                    b) out of an intersection  """

            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID

                # Skip GA aircraft
                if id_[0:2] == "GA":
                    continue

                # Skip vehicles who are waiting outside the system
                if id_ in self.pending_initial_requests:
                    continue

                """ Vehicles that have spawned during the current step might be removed if space is not available,
                    therefore we need to skip them until they are authorized to enter the system.
                    Vehicles who were in self.vehicle_helpers at the beginning of the step have already entered the system """
                if id_ not in self.vehicle_helpers.keys():
                    continue

                curr_gps = [self.bs.traf.lon[i], self.bs.traf.lat[i]]
                # if id_ == "PPPDT12":
                #     for intersection in self.traffic_manager.intersections.values():
                #         if self.traffic_manager.check_if_within_intersection(
                #                 curr_gps, intersection.tower_ID
                #         ):
                #             print("In Intersection: ", intersection.tower_ID)
                # Determine which vehicles are inside which intersections
                for intersection in self.traffic_manager.intersections.values():
                    if self.traffic_manager.check_if_within_intersection(
                        curr_gps, intersection.tower_ID
                    ):
                        # Determine if a vehicle is already accepted into the intersection
                        if (
                            id_ in intersection.accepted
                            or id_ in intersection.recently_left
                        ):
                            break
                        # Determine if authorized vehicle has transitioned into intersection
                        elif id_ in intersection.authorized:
                            # if id_ == 'PPPDT12':
                            #     print("Aircraft has entered intersection: ", id_)
                            intersection.accepted.append(id_)
                            intersection.authorized.remove(id_)
                            # Update vehicle helper
                            self.vehicle_helpers[id_].within_intersection = True
                            self.vehicle_helpers[
                                id_
                            ].current_intersection = intersection.tower_ID
                            # Update the route section that the vehicle left if not an initial request
                            crs = self.vehicle_helpers[
                                id_
                            ].current_route_section  # Will be None if initial request
                            if crs:
                                self.traffic_manager.towers[crs].accepted.remove(id_)
                            break
                        # Determine if vehicle has entered illegally
                        elif (
                            id_ not in intersection.authorized
                            and id_ not in intersection.accepted
                            and id_ not in intersection.recently_left
                        ):
                            if id_ not in intersection.illegal:
                                intersection.illegal.append(id_)
                                print(
                                    id_,
                                    "has entered",
                                    intersection.tower_ID,
                                    "illegally",
                                    self.vehicle_helpers[id_].current_route_section,
                                    self.vehicle_helpers[id_].next_intersection,
                                )
                                
                                time.sleep(100)
                                # Update vehicle helper
                                self.vehicle_helpers[id_].within_intersection = True
                                self.vehicle_helpers[
                                    id_
                                ].current_intersection = intersection.tower_ID
                                break
                            else:
                                # print(id_, "is still in", intersection.tower_ID, "illegally")
                                break
                        else:  # Raise error because it should not be possible to pass through all of the above checks
                            raise BadLogic(
                                "Bad logic when checking if vehicle is within intersection. id_: ",
                                id_,
                                "intersection: ",
                                intersection.tower_ID,
                            )
                    else:
                        # Determine if accepted vehicle has transitioned out of intersection
                        if id_ in intersection.accepted:
                            intersection.accepted.remove(id_)
                            intersection.recently_left.append(id_)
                            # Update vehicle helper to reflect that this vehicle is no longer in the intersection
                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            self.vehicle_helpers[
                                id_
                            ].enter_request_status = False  # reset
                            # Update the route section that the vehicle entered if not exiting the system
                            idx = self.bs.traf.id2idx(id_)
                            autopilot_route = self.bs.traf.ap.route[idx]
                            route_counter = 0
                            found_next_wp = False
                            current_route_section = None
                            next_route_section = None
                            while not found_next_wp and autopilot_route.iactwp + route_counter < len(autopilot_route.wpname):
                                current_waypoint_string = autopilot_route.wpname[autopilot_route.iactwp + route_counter]
                                current_route_section_temp =  current_waypoint_string[0:4]
                                if current_route_section_temp in self.routes_loaded.keys():
                                    current_route_section = current_route_section_temp
                                    found_next_wp = True 
                                else:
                                    route_counter += 1
                            found_next_wp = False
                            while not found_next_wp and autopilot_route.iactwp + route_counter < len(autopilot_route.wpname):
                                next_waypoint_string = autopilot_route.wpname[autopilot_route.iactwp + route_counter]
                                next_route_section_temp =  next_waypoint_string[0:4]
                                if next_route_section_temp in self.routes_loaded.keys() and current_route_section != next_route_section_temp:
                                    next_route_section = next_route_section_temp
                                    found_next_wp = True 
                                else:
                                    route_counter += 1

                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            if current_route_section != None:
                                self.vehicle_helpers[id_].route = self.routes_loaded[current_route_section]
                            self.vehicle_helpers[id_].next_route_section = next_route_section
                            self.vehicle_helpers[
                                id_
                            ].enter_request_status = False  # reset
                            # Remove id from shielding for the appropriate intersection
                            # for alt_key in intersection.alt_levels.keys():
                            #     if id_ in intersection.alt_levels[alt_key]:
                            #         intersection.alt_levels[alt_key].remove(id_)
                            """ THE FOLLOWING IF STATEMENT SHOULD BE REMOVABLE """
                            # nrs = self.vehicle_helpers[
                            #     id_
                            # ].next_route_section  # Will be None if exiting the system
                            # if nrs:
                            #     self.traffic_manager.towers[nrs].accepted.append(id_)
                            #     self.traffic_manager.towers[nrs].authorized.remove(id_)
                            #     self.vehicle_helpers[id_].change_route_section()
                            # break
                            if id_ == 'PPPDT12':
                                print("Aircraft Accepted: ", id_, intersection.tower_ID)
                            if current_route_section:
                                self.traffic_manager.towers[current_route_section].accepted.append(id_)
                                self.traffic_manager.towers[current_route_section].authorized.remove(id_)
                                self.vehicle_helpers[id_].current_route_section = self.vehicle_helpers[id_].route.route_id
                                self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].current_route_section[2:4] # self.traffic_manager.search_for_intersection(self.vehicle_helpers[id_].current_route_section, "inbound").tower_ID
                            break
                        # Determine if illegal vehicle has transitioned out of intersection
                        elif id_ in intersection.illegal:
                            intersection.illegal.remove(id_)
                            # Update vehicle helper to reflect that this vehicle is no longer in the intersection
                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            self.vehicle_helpers[id_].enter_request_status = False
                            print(
                                "Illegal vehicle",
                                id_,
                                "has exited",
                                intersection.tower_ID,
                            )
                else:
                    # Set vehicle helper to reflect that this vehicle is not in an intersection
                    self.vehicle_helpers[id_].within_intersection = False
                    self.vehicle_helpers[id_].current_intersection = None

            # Update the volume of each intersection
            for intersection in self.traffic_manager.intersections.values():
                intersection.set_volume()
            # Update the volume of each route section
            for tower in self.traffic_manager.towers.values():
                tower.set_volume()

            """ Collect new vehicle requests for processing """
            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID
                if id_[0:2] == "GA":
                    continue

                # Check if the current ID exists. If not then create a new vehicle helper
                if not id_ in self.vehicle_helpers.keys():
                    # Get and reformat the route name coming from Bluesky
                    route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                    self.vehicle_helpers[id_] = VehicleHelper(
                        id_, self.routes_loaded[route_name]
                    )
                    # Add initial request to enter the system
                    initial_requests.append(id_)
                else:
                    a = id_ not in self.pending_requests
                    b = id_ not in self.pending_initial_requests
                    c = (
                        id_ not in self.exiting_vehicles
                    )  # TODO: Check if this is necessary
                    d = not self.vehicle_helpers[id_].enter_request_status
                    if a and b and c and d:
                        request_eligibility = self.vehicle_helpers[
                            id_
                        ].check_if_request_eligible(
                            [self.bs.traf.lon[i], self.bs.traf.lat[i]]
                        )
                        if request_eligibility:
                            new_requests.append(id_)

            """ Process requests """
            # First process pending requests
            for id_ in self.pending_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()  # tuple
                self.traffic_manager.add_request(id_, formatted_request)
            pending_request_response = self.traffic_manager.process_requests()

            # Second process new requests
            for id_ in new_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                self.traffic_manager.add_request(id_, formatted_request)
            new_request_response = self.traffic_manager.process_requests()

            # Third process pending initial requests
            for id_ in self.pending_initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                self.traffic_manager.add_request(id_, formatted_request)
            pending_initial_request_response = self.traffic_manager.process_requests()

            # Fourth process initial requests
            for id_ in initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                self.traffic_manager.add_request(id_, formatted_request)
            initial_request_response = self.traffic_manager.process_requests()

            """ Collect responses and update request lists based on the response """
            collected_responses = {}
            # Pending in system responses
            for id_, response in pending_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response:
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.pending_requests.remove(id_)
                    if self.vehicle_helpers[id_].final_route_segment:
                        self.exiting_vehicles.append(id_)
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
            # New in system responses
            for id_, response in new_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response:
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].final_route_segment:
                        self.exiting_vehicles.append(id_)
                else:
                    self.pending_requests.append(id_)
                    self.vehicle_helpers[id_].enter_request_status = False
            # Pending initial request responses
            for id_, response in pending_initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.pending_initial_requests.remove(id_)
                    if self.vehicle_helpers[id_].next_intersection == None:
                        self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].route.route_id[2:4]
                    # set the clearance to True. Optional fields are ALT, SPD
                    k_idx = self.bs.traf.id2idx(id_)
                    self.bs.traf.ap.setclrcmd(k_idx, True, 400, 30)
                    # print(f"{id_} cleared for departure at time {self.bs.sim.simt}")
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
            # Initial request responses
            for id_, response in initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].next_intersection == None:
                        self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].route.route_id[2:4]
                    # print(f"{id_} cleared for departure at time {self.bs.sim.simt}")
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
                    self.pending_initial_requests.append(id_)
                    k_idx = self.bs.traf.id2idx(id_)
                    # set the clearance to False (i.e., denied and hold on ground)
                    self.bs.traf.ap.setclrcmd(k_idx, False)

        ### End of traffic manager code ###

        obs, reward, done, info = self.state_update(
            self.bs.traf,
            a=actions,
            policy=policy,
            value=value,
            init=False,
            tm_response=collected_responses,
        )

        if len(self.bs.traf.id) == 0:
            self.time_without_traffic += self.bs.sim.simdt
        else:
            self.time_without_traffic = 0

        if self.time_without_traffic > 1800:  # 0.5 hours
            done["__all__"] = True

        else:
            done["__all__"] = False

        return obs, reward, done, info

    def state_update(
        self,
        traf,
        a=None,
        policy=None,
        value=None,
        pp=None,
        tp=None,
        init=False,
        tm_response: dict = None,
    ):
        # current number of a/c in bluesky sim
        n_ac = traf.lat.shape[0]
        rew = {}
        state = {}
        done = {}
        info = {}

        #
        self.action_override = []

        ## creating an index for the unique aircraft
        index = np.arange(n_ac).reshape(-1, 1)

        ## calculating the distance from each aircraft to all others. Will result in a n_ac x n_ac matrix
        # d = geo.latlondist_matrix(np.repeat(traf.lat,n_ac), np.repeat(traf.lon,n_ac), np.tile(traf.lat,n_ac), np.tile(traf.lon,n_ac)).reshape(n_ac,n_ac)*geo.nm ## conver to meters
        # argsort = np.array(np.argsort(d, axis=1))
        d = (
            geo.kwikdist_matrix(
                np.repeat(traf.lat, n_ac),
                np.repeat(traf.lon, n_ac),
                np.tile(traf.lat, n_ac),
                np.tile(traf.lon, n_ac),
            ).reshape(n_ac, n_ac)
            * geo.nm
        )
        argsort = np.array(np.argsort(d, axis=1))

        # transform all aircraft lon/lat positions
        coord_transform = self.transformer.transform(traf.lon, traf.lat)

        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        traf.ap.route[i].wplon[j],
                        traf.ap.route[i].wplat[j],
                    )
                    for j in range(
                        traf.ap.route[i].iactwp,
                        len(traf.ap.route[i].wplon),
                    )
                ]
                for i in range(traf.lat.shape[0])
            ]
        )
        # geometries = MultiLineString(
        #     [
        #         [[traf.lon[i], traf.lat[i]], [traf.ap.route[i].wplon[-1], traf.ap.route[i].wplat[-1]]]
        #         for i in range(traf.lat.shape[0])
        #     ]
        # )

        # looping over the ownships
        for i in range(d.shape[0]):
            # ownship ID
            id_ = traf.id[i]

            if id_ not in self.acInfo:
                self.acInfo[id_] = {
                    "NMAC": [],
                    "Lat": [],
                    "Lon": [],
                    "Spd": [],
                    "Action": [],
                    "time": [],
                }

            # if the aircraft has not taken off, skip
            if not traf.active[i]:
                continue

            # own_glat, own_glon = traf.ap.route[i].wplat[-1], traf.ap.route[i].wplon[-1]
            # dist = geo.kwikdist(traf.lat[i], traf.lon[i], own_glat, own_glon) * geo.nm  ##meters
            ownship_obj = geometries.geoms[i]

            ownship_waypoints = list(ownship_obj.coords)
            # first entry is current location
            current_position_ownship = ownship_waypoints[0]
            remaining_waypoints = ownship_waypoints[1 : 1 + self.n_waypoints]

            rel_waypoints = []
            for waypoint in remaining_waypoints:
                # rel_waypoints += [
                #     waypoint[0] - current_position_ownship[0],
                #     waypoint[1] - current_position_ownship[1],
                # ]

                relx = waypoint[0] - current_position_ownship[0]
                rely = waypoint[1] - current_position_ownship[1]

                bearing = math.degrees(math.atan2(rely, relx))

                dist_waypoint = np.sqrt(relx**2 + rely**2)
                rel_waypoints += [
                    bearing,
                    dist_waypoint,
                ]

            if len(remaining_waypoints) < self.n_waypoints:
                rel_waypoints += [0, 0] * (self.n_waypoints - len(remaining_waypoints))
            dist = ownship_obj.length

            """ Apply the result of the vehicle request WHERE? """

            ## Converting ownship lat/lon to UTM coords
            xEast_own, yNorth_own = (
                coord_transform[0][i],
                coord_transform[1][i],
            )  # self.transformer.transform(traf.lon[i], traf.lat[i])

            prev_action_own = 1  # maintain
            if a is not None:
                if id_ in a:
                    prev_action_own = a[id_]

            rew[id_] = 0

            if self.traffic_manager_active:
                # TODO: The following check will not work for vehicles that exist but arent currently requesting.
                # TODO: tm_response is only populated with vehicles that are requesting
                # TODO: the other option is to set the distance value that indicates no request
                if id_ in tm_response:
                    # try:
                    response, distance = tm_response[id_]  # 0 = denied, 1 = approved
                    # here
                    # except:
                    #    response = tm_response[id_]
                    #    distance = -1
                    response = int(response)

                    if response == 0 and distance != -1:
                        rew[id_] += -self.clearancePenalty

                    # hard action for when aircraft is too close to the boundary and SA did not slow down
                    # TODO: Store a flag of the vehicle that needs to be overwritten
                    # print(distance, response, id_, traf.active[i])
                    if distance != -1 and distance < 250 and response == 0:
                        # print(traf.cas[i])
                        # self.bs.stack.stack(f"SPD {id_} 0")
                        rew[id_] += -10 * self.clearancePenalty
                        self.action_override.append(id_)
                else:
                    response = 2  # no request
                    distance = 0

                # TODO: Set a dist value that indicates the vehicle is not requesting
                if self.speed_only:
                    own_state = [
                        dist,
                        traf.cas[i],
                        traf.ax[i],
                        traf.hdg[i],
                        self.LOS,
                        prev_action_own,
                        response,
                        distance,
                    ]
                else:
                    # Calculate Ambient Noise
                    ambient_noise = 40
                    current_route_section_name = self.vehicle_helpers[id_].current_route_section
                    current_intersection_name = self.vehicle_helpers[id_].current_intersection
                    if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                        ambient_noise = self.ambient_noise_level[current_route_section_name]
                    else:
                        if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                            ambient_noise = self.ambient_noise_level[current_intersection_name]
                    if id_ in self.alt_changing.keys():
                        alt_changing_bool = 1
                    else:
                        alt_changing_bool = 0
                    own_state = [
                        # dist,
                        # traf.cas[i],
                        # traf.ax[i],
                        # traf.hdg[i],
                        # self.LOS,
                        prev_action_own,
                        traf.alt[i],
                        ambient_noise,
                        alt_changing_bool,
                        # response,
                        # distance,
                    ]

            else:
                if self.speed_only:
                    own_state = [
                        dist,
                        traf.cas[i],
                        traf.ax[i],
                        traf.hdg[i],
                        self.LOS,
                        prev_action_own,
                    ]
                else:
                    ambient_noise = 40
                    current_route_section_name = self.vehicle_helpers[id_].current_route_section
                    current_intersection_name = self.vehicle_helpers[id_].current_intersection
                    if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                        ambient_noise = self.ambient_noise_level[current_route_section_name]
                    else:
                        if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                            ambient_noise = self.ambient_noise_level[current_intersection_name]
                    if id_ in self.alt_changing.keys():
                        alt_changing_bool = 1
                    else:
                        alt_changing_bool = 0
                    own_state = [
                        # dist,
                        # traf.cas[i],
                        # traf.ax[i],
                        # traf.hdg[i],
                        # self.LOS,
                        prev_action_own,
                        traf.alt[i],
                        ambient_noise,
                        alt_changing_bool
                    ]
            if self.speed_only:
                own_state += rel_waypoints

            own_state = np.array(own_state).reshape(1, self.observation_space.shape[0])
            # print("Own State Example: ", id_, own_state)
            #  np.array(
            #         [
            #             dist,
            #             traf.cas[i],
            #             traf.ax[i],
            #             traf.hdg[i],
            #             self.LOS,
            #             prev_action_own,
            #         ]
            #     ).reshape(1, self.observation_space.shape[0])

            ## check normalization values
            self.normalization_check(x=xEast_own, y=yNorth_own, d=dist)

            own_state = (own_state - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low
            )

            self.acInfo[id_]["Lat"].append(traf.lat[i])
            self.acInfo[id_]["Lon"].append(traf.lon[i])
            self.acInfo[id_]["Spd"].append(traf.cas[i])
            self.acInfo[id_]["NMAC"].append(
                0
            )  # place holder 0 that is overwritten later if NMAC occurred
            self.acInfo[id_]["time"].append(self.bs.sim.simt)
            if a is not None and id_ in a:
                self.acInfo[id_]["Action"].append(a[id_])

            else:
                self.acInfo[id_]["Action"].append(1)  # "hold"

            done[id_] = False
            info[id_] = None

            # if self.traffic_manager_active:
            #     if id_ in self.exiting_vehicles:
            #         done[id_] = True
            # else:
            ## made it to the goal
            if dist < self.dGoal:
                # if self.traffic_manager_active:
                #    if id_ in self.exiting_vehicles:
                #        done[id_] = True
                #
                # else:

                if self.traffic_manager_active:
                    intersection = self.vehicle_helpers[id_].current_intersection
                    if intersection != None:
                        self.traffic_manager.intersections[
                            intersection
                        ].accepted.remove(id_)
                        self.traffic_manager.intersections[intersection].set_volume()
                done[id_] = True

            # is this a GA aircraft?
            if id_[0:2] == "GA":
                if done[id_]:
                    self.bs.stack.stack("DEL {}".format(id_))
                # should prevent non-coop from formining a state
                # but state info of non-coop will still be available
                # to the remaining coop aircraft
                continue

            reward_count = False
            closest_count = self.agent.max_agents
            intruder_state = None
            # Noise Information
            alt_own = self.bs.traf.alt[i]
            alt_own_ft = self.meters_to_feet(alt_own)
            intruder_noise_vals = {} 

            for j in range(len(argsort[i])):
                index = int(argsort[i][j])
                id_j = self.bs.traf.id[index]
                # The first entry will be: intruder == ownship so we need to skip
                if i == index:
                    continue

                if not traf.active[index]:
                    continue
                ## ALT Information


                alt_intruder = self.bs.traf.alt[index]
                alt_intruder_ft = self.meters_to_feet(alt_intruder)
                id_j = self.bs.traf.id[index]
                dist_with_alt = np.sqrt((d[i, index]**2) + ((alt_own - alt_intruder)**2)) # Replace d[i, index] with this value
                


                # -1 index so that it is the true goal location even with multiple waypoints
                # glat, glon = traf.ap.route[index].wplat[-1], traf.ap.route[index].wplon[-1]
                # dist = geo.latlondist(traf.lat[index],traf.lon[index],glat,glon) # meters
                # dist = geo.kwikdist(traf.lat[index], traf.lon[index], glat, glon) * geo.nm
                intruder_obj = geometries.geoms[index]
                dist = intruder_obj.length
                # intruder to be removed
                if dist < self.dGoal and d[i, index] > self.LOS:
                    continue

                # if the intruder is > 750 meters (0.5 nm) away, skip it.
                if (
                    d[i, index] > self.intruderThreshold
                ):  # TODO: This is a hyperparameter that needs to be moved to a config file
                    continue

                if alt_intruder > 0:
                    # print("Checking Altitude: ", alt_intruder, alt_intruder_ft)
                    intruder_noise_vals[index] = self.a_0 +  (self.a_1 * math.log10(alt_intruder_ft)) + (self.a_2 * ((math.log10(alt_intruder_ft))**2))
                else:
                    intruder_noise_vals[index] = 0

                # if the ownship and intruder do not intersect and they are not on the same route
                if not ownship_obj.intersects(intruder_obj):
                    continue

                if self.vehicle_helpers[id_].current_route_section != None and self.vehicle_helpers[id_j].current_route_section != None:
                    swapped_route = self.vehicle_helpers[id_].current_route_section[2:4] + self.vehicle_helpers[id_].current_route_section[0:2]
                    if swapped_route == self.vehicle_helpers[id_j].current_route_section:
                        continue

                ## At this point. The intruder is only considered if the routes intersect
                ## Now I need to take care of tracks on the same route/lane

                # ilon, ilat = list(ownship_obj.intersection(intruder_obj).coords)[0]
                #
                # dist_int_inter = geo.kwikdist(traf.lat[index], traf.lon[index], ilat, ilon)  # nautical miles
                # dist_own_inter = geo.kwikdist(traf.lat[i], traf.lon[i], ilat, ilon)  # nautical miles
                # print("Checking: ", id_, index, d[i, index], a)
                if self.speed_only:
                    if d[i, index] < self.LOS and not reward_count and a != None:
                        rew[id_] += self.rewardLOS
                        reward_count = True
                        info[id_] = 1
                        if id_ in a:
                            self.acInfo[id_]["NMAC"][-1] = 1
                
                    if not reward_count:
                        if d[i, index] < self.maxRewardDistance and d[i, index] > self.LOS:
                            rew[id_] += -self.rewardAlpha + self.rewardBeta * (d[i, index])
                            reward_count = True
                
                ## Converting intruder lat/lon to UTM coords
                xEast_int, yNorth_int = (
                    coord_transform[0][index],
                    coord_transform[1][index],
                )

                distIntGoal = dist

                relX = xEast_int - xEast_own
                relY = yNorth_int - yNorth_own
                prev_action_int = 1
                if a is not None:
                    if traf.id[index] in a:
                        prev_action_int = a[traf.id[index]]

                intruder_waypoints = list(intruder_obj.coords)
                current_position_intruder = intruder_waypoints[
                    0
                ]  # first entry is current location
                remaining_waypoints = intruder_waypoints[
                    1 : 1 + self.n_waypoints
                ]  # first entry is current location

                rel_waypoints = []
                for waypoint in remaining_waypoints:
                    rel_x = waypoint[0] - current_position_ownship[0]
                    rel_y = waypoint[1] - current_position_ownship[1]

                    bearing = math.degrees(math.atan2(rel_y, rel_x))

                    dist_waypoint = np.sqrt(rel_x**2 + rel_y**2)
                    rel_waypoints += [
                        bearing,
                        dist_waypoint,
                    ]
                if len(remaining_waypoints) < self.n_waypoints:
                    rel_waypoints += [0, 0] * (
                        self.n_waypoints - len(remaining_waypoints)
                    )
                # print("Example of Rel Waypoints: ", id_, id_j, rel_waypoints)
                if self.speed_only:
                    int_state = np.array(
                        [
                            relX,
                            relY,
                            distIntGoal,
                            traf.cas[index],
                            traf.ax[index],
                            traf.hdg[index],
                            d[i, index],
                            prev_action_int,

                        ]
                        + rel_waypoints
                    ).reshape(1, self.context_space.shape[0])
                else:
                    int_state = np.array(
                        [
                            # relX,
                            # relY,
                            # distIntGoal,
                            # traf.cas[index],
                            # traf.ax[index],
                            # traf.hdg[index],
                            d[i, index],
                            prev_action_int,
                            traf.alt[index],
                        ]
                        # + rel_waypoints
                    ).reshape(1, self.context_space.shape[0])
                # print("Intruder State Example: ", id_j, int_state)
                self.normalization_check(x=xEast_int, y=yNorth_int, d=distIntGoal)

                int_state = (int_state - self.context_space.low) / (
                    self.context_space.high - self.context_space.low
                )

                if intruder_state is None:
                    intruder_state = int_state

                else:
                    intruder_state = np.append(intruder_state, int_state, axis=0)

                closest_count -= 1

                if closest_count == 0:
                    break

            if closest_count != 0:
                remaining = np.zeros((closest_count, self.intruder_obs_dim))
                if intruder_state is None:
                    intruder_state = remaining

                else:
                    intruder_state = np.append(intruder_state, remaining, axis=0)

                state[id_] = {
                    "ownship_obs": own_state.reshape(1, self.ownship_obs_dim),
                    "intruder_obs": intruder_state.reshape(
                        1, self.agent.max_agents, self.intruder_obs_dim
                    ),
                }

            else:
                state[id_] = {
                    "ownship_obs": own_state.reshape(1, self.ownship_obs_dim),
                    "intruder_obs": intruder_state.reshape(
                        1, self.agent.max_agents, self.intruder_obs_dim
                    ),
                }


            ### NOISE REWARD IMPLEMENTATION ###
            ## 1: Single Event Noise Calculation
            if not self.speed_only:
                next_route_section_name = self.vehicle_helpers[id_].next_route_section
                current_route_section_name = self.vehicle_helpers[id_].current_route_section
                current_intersection_name = self.vehicle_helpers[id_].current_intersection
                ambient_noise = 40
                if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                    ambient_noise = self.ambient_noise_level[current_route_section_name]
                else:
                    if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                        ambient_noise = self.ambient_noise_level[current_intersection_name]
                if alt_own > 0:
                    current_aircraft_noise = self.a_0 +  (self.a_1 * math.log10(alt_own_ft)) + (self.a_2 * ((math.log10(alt_own_ft))**2))
                    # current_aircraft_noise = (self.noise_const_a * math.log(alt_own)) + self.noise_const_b
                else:
                    current_aircraft_noise = 0
                ## 2: Group Event Noise
                # inner_log_val = (10**(ambient_noise / 10))
                inner_log_val = 0 # (10**(current_aircraft_noise / 10))
                for single_noise_val in intruder_noise_vals.values():
                    inner_log_val += (10**(single_noise_val / 10))
                if inner_log_val != 0:
                    total_noise_impact = 10 * math.log10(inner_log_val)
                else:
                    total_noise_impact = 0
                # This is a test to see if this is a viable reward formulation
                # total_noise_factored = current_aircraft_noise - total_noise_impact

                ## Old Version:
                total_noise_factored_own = current_aircraft_noise - ambient_noise
                best_case_noise = self.a_0 +  (self.a_1 * math.log10(self.max_alt)) + (self.a_2 * ((math.log10(self.max_alt))**2))
                worst_case_noise = self.a_0 +  (self.a_1 * math.log10(0.1)) + (self.a_2 * ((math.log10(0.1))**2))
                best_case_noise_increase = best_case_noise - ambient_noise
                worst_case_noise_increase = worst_case_noise - ambient_noise
                noise_increase_over_best_case = total_noise_factored_own # - best_case_noise_increase
                noise_increase_normalized = (total_noise_factored_own - best_case_noise_increase) / (worst_case_noise_increase - best_case_noise_increase)
                total_noise_factored_int = (ambient_noise - current_aircraft_noise - (10 * math.log10(10**((total_noise_impact - current_aircraft_noise)/10)+1)))
                # Ambient noise needs to be included
                # Total noise is around ~ -26per time step. Max negative reward per time step is 
                total_noise_factored = (self.alpha * noise_increase_over_best_case) + (self.beta * total_noise_factored_int)
                if total_noise_factored >= 0:
                    total_noise_factored = 0
                # print("Ambient Noise values: ", alt_own_ft, total_noise_factored_own, best_case_noise_increase, worst_case_noise_increase)
                # print("Noise reward: ", noise_increase_normalized, total_noise_factored, self.gamma)
                rew[id_] -= noise_increase_normalized
            if self.speed_only:
                if not init and not done[id_]:
                    if a is not None:
                        if id_ in a.keys():
                            if a[id_] == 0 or a[id_] == 2:
                                rew[id_] += -self.speedChangePenalty
                            rew[id_] += -self.stepPenalty  ## step penalty
                            # print("Energy reward: ", self.speedChangePenalty, self.stepPenalty)

        return state, rew, done, info

    def run_one_iteration(self, weights):
        """
        2022/11/1 modify the policy implementation and introduce the non-cooperative behaviors
        """

        if self.agent.equipped:
            self.agent.model.set_weights(weights)
        # self.agent.data = {}
        self.agent.reset()
        self.step_counter = 0

        if self.episode_done:
            obs = self.reset()
            self.nmacs = 0
            self.nmac_time = 0
            self.total_ac = 0
        else:
            obs = self.last_obs

        while True:
            if len(obs) > 0:
                action, policy, value = self.agent.predict(
                    obs, self.non_coop_tag, self.LControl_lst, self.LComm_lst
                )

            else:
                action, policy, value = {}, {}, {}

            next_obs, rew, term, term_type = self.step(action, policy, value)

            next_obs = self.store_data(
                obs, action, rew, next_obs, term, term_type, policy, value
            )

            obs = next_obs

            if term["__all__"] or self.step_counter >= self.max_steps:
                self.last_obs = next_obs

                if term["__all__"]:
                    self.episode_done = True

                if self.step_counter >= self.max_steps:
                    self.step_counter = 0

                #     # Need to process remaining entries in self.memory to self.data
                for id_ in self.agent.memory.keys():
                    # if the id_ has already been processed then skip it
                    if id_ in self.agent.data.keys():
                        continue
                    self.agent.process_memory(id_)

                # # TODO: is this necessary?
                for key in self.agent.data.keys():
                    if type(self.agent.data[key]) == list:
                        self.agent.data[key] = np.concatenate(
                            self.agent.data[key], axis=0
                        )

                self.agent.data["nmacs"] = self.nmacs
                self.agent.data["nmac_time"] = self.nmac_time
                self.agent.data["total_ac"] = self.total_ac

                if self.episode_done:  # self.run_type == "eval":
                    self.agent.data["aircraft"] = self.acInfo

                # # TODO: Do someething ike this
                # self.agent.data["intersection_metrics"] = self.intersection_metrics
                # ************

                if not "raw_reward" in self.agent.data:
                    self.agent.data["raw_reward"] = np.array([0.0])

                self.agent.data["environment_done"] = self.episode_done
                data_ID = ray.put([self.agent.data, self.id])

                return data_ID

            # if self.step_counter >= self.max_steps:

            #     self.last_obs = next_obs

            #     # Need to process remaining entries in self.memory to self.data
            #     for id_ in self.memory.keys():

            #         # if the id_ has already been processed then skip it
            #         if id_ in self.data.keys():
            #             continue

            #         self.process_memory(id_)

            #     # convert to array from list of arrays
            #     for model in self.data.keys():
            #         for key in self.data[model].keys():
            #             if type(self.data[model][key]) == list:
            #                 self.data[model][key] = np.concatenate(self.data[model][key], axis=0)

            #     self.data_ID = ray.put([self.data, self.id])
            #     del self.data

            #     return self.data_ID  # self.data, self.id

    def store_data(self, obs, action, rew, next_obs, term, term_type, policy, value):
        obs_updated = copy(next_obs)
        for ac_id in obs.keys():
            self.agent.store_step(
                ac_id, obs, action, rew, next_obs, term, policy, value
            )

            if term[ac_id]:
                self.total_ac += 1

                # did an NMAC occur
                if 1 in self.acInfo[ac_id]["NMAC"]:
                    # if term_type[ac_id] == 1:
                    group = groupby(self.acInfo[ac_id]["NMAC"])
                    group = np.array([x[0] for x in group])

                    self.nmac_time += (
                        sum(self.acInfo[ac_id]["NMAC"]) * self.bs.sim.simdt
                    )
                    self.nmacs += sum(group)
                self.bs.stack.stack("DEL {}".format(ac_id))
                # TODO: MIGHT NEED TO REMOVE VEHICLES HERE....
                # if ac_id in self.exiting_vehicles: # TODO: NOT SUSTAINABLE. THIS WILL RESULT IN AIRSPACE VOLUME NEVER BEING REDUCED

                # if self.traffic_manager_active:

                # if self.traffic_manager_active:

                #     try: # ?????
                #         self.exiting_vehicles.remove(ac_id)

                #     except:
                #         import ipdb;ipdb.set_trace()
                del obs_updated[ac_id]

        return obs_updated

    def normalization_check(self, x=None, y=None, d=None):
        traf = self.bs.traf

        if len(traf.lat) == 0:
            return

        if traf.cas.max() > self.tas_max:
            self.tas_max = traf.cas.max()
            # print("CAS MAX ", traf.cas.max())

        if traf.cas.min() < self.tas_min:
            self.tas_min = traf.cas.min()
            # print("CAS MIN ", traf.cas.min())

        if traf.ax.max() > self.ax_max:
            self.ax_max = traf.ax.max()
            # print("AX MAX ", traf.ax.max())
        if traf.ax.min() < self.ax_min:
            self.ax_min = traf.ax.min()
            # print("AX MIN ", traf.ax.min())

        if x is not None:
            if x < self.min_x:
                self.min_x = x
                # print("X Min ", x)

            elif x > self.max_x:
                self.max_x = x
                # print("X Max ", x)

            # prevent numerical normalization errors
            if self.max_x == self.min_x:
                self.max_x += 1e-4

        if y is not None:
            if y < self.min_y:
                self.min_y = y
                # print("Y Min ", y)

            elif y > self.max_y:
                self.max_y = y
                # print("Y Max ", y)

            # prevent numerical normalization errors
            if self.max_y == self.min_y:
                self.max_y += 1e-4

        if d is not None:
            if d > self.max_d:
                self.max_d = d
                # print("Max dist  ", d)
        if self.speed_only:
            ownship_min_state = [0, self.tas_min, self.ax_min, 0, 0, 0]
            ownship_max_state = [self.max_d, self.tas_max, self.ax_max, 360, self.max_d, 2]

            intruder_min_state = [
                self.min_x - self.max_x,
                self.min_y - self.max_y,
                0,
                self.tas_min,
                self.ax_min,
                0,
                0,
                0,
            ]

            intruder_max_state = [
                self.max_x - self.min_x,
                self.max_y - self.min_y,
                self.max_d,
                self.tas_max,
                self.ax_max,
                360,
                self.max_d,
                2,
            ]
        else:
            ownship_min_state = [0, 0, 40, 0]
            ownship_max_state = [2, self.max_alt, 60, 1]

            intruder_min_state = [
                # self.min_x - self.max_x,
                # self.min_y - self.max_y,
                # 0,
                # self.tas_min,
                # self.ax_min,
                # 0,
                0,
                0,
                0
            ]

            intruder_max_state = [
                # self.max_x - self.min_x,
                # self.max_y - self.min_y,
                # self.max_d,
                # self.tas_max,
                # self.ax_max,
                # 360,
                self.max_d,
                2,
                self.max_alt
            ]

        # clearance denied, cleared, no clearance request
        # print("Checking if Traffic Manager is active: ", self.traffic_manager_active)
        if self.speed_only:
            if self.traffic_manager_active:
                ownship_min_state += [0, 0]
                ownship_max_state += [2, self.max_d]
            ownship_min_state += [-180, 0] * self.n_waypoints
            ownship_max_state += [180, self.max_d] * self.n_waypoints
            intruder_min_state += [-180, 0] * self.n_waypoints
            intruder_max_state += [180, self.max_d] * self.n_waypoints

        ## rel utm position, dist goal, speed, acceleration, heading, distance ownship to intruder, distance intruder intersection, distance ownship to intersection
        self.context_space = Box(
            np.array(intruder_min_state),
            np.array(intruder_max_state),
            dtype=np.float64,
        )

        self.observation_space = Box(
            np.array(ownship_min_state),
            np.array(ownship_max_state),
            dtype=np.float64,
        )

    def within_LOS(self, id_i):
        n_ac = self.bs.traf.lat.shape[0]
        d = (
            geo.kwikdist_matrix(
                np.repeat(self.bs.traf.lat, n_ac),
                np.repeat(self.bs.traf.lon, n_ac),
                np.tile(self.bs.traf.lat, n_ac),
                np.tile(self.bs.traf.lon, n_ac),
            ).reshape(n_ac, n_ac)
            * geo.nm
        )
        i = self.bs.traf.id2idx(id_i)
        for j in range(self.bs.traf.lat.shape[0]):
            id_j = self.bs.traf.id[j]
            # if id_j == 'PDTPP5' and id_i == 'PDTPP27':
            #     print("Checking Dist: ", d[i, j], self.bs.traf.active[j], self.LOS)
            if id_i == id_j:
                continue
            if not self.bs.traf.active[
                j
            ]:
                continue
            dist = d[i, j]
            if dist <= 1.5 * self.LOS:
                # print("HOLD IT", id_i, id_j, d[i, j], self.LOS)
                return True
        return False
    
    def meters_to_feet(self, meters):
        feet = meters * 3.28084
        return feet

import ray
import os
import gin
import argparse
from D2MAV_A.agent import Agent
from D2MAV_A.runner import Runner
from copy import deepcopy
import time
import platform
import numpy as np

os.environ["PYTHONPATH"] = os.getcwd()

import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--learn_action", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


@gin.configurable
class Driver:
    def __init__(
        self,
        cluster=False,
        run_name=None,
        scenario_file=None,
        config_file=None,
        num_workers=1,
        iterations=1000,
        simdt=1,
        max_steps=1024,
        speeds=[0, 0, 84],
        LOS=10,
        dGoal=100,
        maxRewardDistance=100,
        intruderThreshold=750,
        rewardBeta=0.001,
        rewardAlpha=0.1,
        speedChangePenalty=0.001,
        rewardLOS=-1,
        stepPenalty=0,
        clearancePenalty=0.005,
        gui=False,
        non_coop_tag=0,
        weights_file=None,
        run_type="train",
        traffic_manager_active=True,
        n_waypoints=2,
    ):
        self.cluster = cluster
        self.run_name = run_name
        self.run_type = run_type
        self.num_workers = num_workers
        self.simdt = simdt
        self.iterations = iterations
        self.max_steps = max_steps
        self.speeds = speeds
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
        self.scenario_file = scenario_file
        self.config_file = config_file
        self.weights_file = weights_file
        self.gui = gui
        self.speed_only = False
        if self.speed_only:
            self.action_dim = 3
            self.observation_dim = 6
            self.context_dim = 8
        else:
            self.action_dim = 3
            self.observation_dim = 4
            self.context_dim = 3
        self.agent = Agent()
        self.agent_template = deepcopy(self.agent)
        self.working_directory = os.getcwd()
        self.non_coop_tag = non_coop_tag
        self.traffic_manager_active = traffic_manager_active
        self.n_waypoints = n_waypoints
        if self.speed_only:
            if self.traffic_manager_active:
                self.observation_dim += 2
            self.observation_dim += 2 * self.n_waypoints
            self.context_dim += 2 * self.n_waypoints

        self.agent.initialize(
            tf, self.observation_dim, self.context_dim, self.action_dim
        )

        if self.run_name is None:
            path_results = "results"
            path_models = "models"
        else:
            path_results = f"results/{self.run_name}"
            path_models = f"models/{self.run_name}"

        os.makedirs(path_results, exist_ok=True)
        os.makedirs(path_models, exist_ok=True)

        self.path_models = path_models
        self.path_results = path_results

    def train(self):
        # Start simulations on actors
        workers = {
            i: Runner.remote(
                i,
                self.agent_template,
                scenario_file=self.scenario_file,
                config_file=self.config_file,
                working_directory=self.working_directory,
                max_steps=self.max_steps,
                simdt=self.simdt,
                speeds=self.speeds,
                LOS=self.LOS,
                dGoal=self.dGoal,
                maxRewardDistance=self.maxRewardDistance,
                intruderThreshold=self.intruderThreshold,
                rewardBeta=self.rewardBeta,
                rewardAlpha=self.rewardAlpha,
                speedChangePenalty=self.speedChangePenalty,
                rewardLOS=self.rewardLOS,
                stepPenalty=self.stepPenalty,
                clearancePenalty=self.clearancePenalty,
                gui=self.gui,
                non_coop_tag=self.non_coop_tag,
                traffic_manager_active=self.traffic_manager_active,
                n_waypoints=self.n_waypoints,
            )
            for i in range(self.num_workers)
        }

        rewards = []
        total_nmacs = []
        iteration_record = []
        total_nmac_time = []
        total_transitions = 0
        best_reward = -np.inf

        if self.agent.equipped:
            if self.weights_file is not None:
                self.agent.model.load_weights(self.weights_file)

            weights = self.agent.model.get_weights()
        else:
            weights = []

        runner_sims = [
            workers[agent_id].run_one_iteration.remote(weights)
            for agent_id in workers.keys()
        ]

        for i in range(self.iterations):
            done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
            results = ray.get(done_id)

            transitions, workers_to_remove = self.agent.update_weights(results)

            if self.agent.equipped:
                weights = self.agent.model.get_weights()

            total_reward = []
            mean_total_reward = None
            nmacs = []
            total_ac = []

            for result in results:
                data = ray.get(result)

                try:
                    total_reward.append(float(np.sum(data[0]["raw_reward"])))
                except:
                    pass

                if data[0]["environment_done"]:
                    nmacs.append(data[0]["nmacs"])

                    total_nmac_time += [data[0]["nmac_time"]]

                    total_ac.append(data[0]["total_ac"])

            if total_reward:
                mean_total_reward = np.mean(total_reward)

            for j, nmac in enumerate(nmacs):
                print(f"     Scenario Complete     ")
                print("|------------------------------|")
                print(f"| Total NMACS:      {nmac}      |")
                print(f"| Total Aircraft:   {total_ac[j]}  |")
                print("|------------------------------|")
                print(" ")
                total_nmacs.append(nmac)
                iteration_record.append(i)

            if mean_total_reward:
                rewards.append(mean_total_reward)
                np.save("{}/reward.npy".format(self.path_results), np.array(rewards))

            if len(nmacs) > 0:
                np.save("{}/nmacs.npy".format(self.path_results), np.array(total_nmacs))

                np.save(
                    "{}/nmac_time.npy".format(self.path_results),
                    np.array(total_nmac_time),
                )

                np.save(
                    "{}/iteration_record.npy".format(self.path_results),
                    np.array(iteration_record),
                )

            total_transitions += transitions

            if not mean_total_reward:
                mean_total_reward = 0

            print(f"     Iteration {i} Complete     ")
            print("|------------------------------|")
            print(f"| Mean Total Reward:   {np.round(mean_total_reward,1)}  |")
            roll_mean = np.mean(rewards[-150:])
            print(f"| Rolling Mean Reward: {np.round(roll_mean,1)}  |")
            print("|------------------------------|")
            print(" ")

            if self.agent.equipped:
                if len(rewards) > 150:
                    if np.mean(rewards[-150:]) > best_reward:
                        best_reward = np.mean(rewards[-150:])
                        self.agent.model.save_weights(
                            "{}/best_model.h5".format(self.path_models)
                        )

                self.agent.model.save_weights("{}/model.h5".format(self.path_models))

            # for agent_id in workers_to_remove:
            #     workers[agent_id] = Runner.remote(
            #         agent_id,
            #         self.agent_template,
            #         scenario_file=self.scenario_file,
            #         config_file=self.config_file,
            #         working_directory=self.working_directory,
            #         max_steps=self.max_steps,
            #         simdt=self.simdt,
            #         speeds=self.speeds,
            #         LOS=self.LOS,
            #         dGoal=self.dGoal,
            #         maxRewardDistance=self.maxRewardDistance,
            #         intruderThreshold=self.intruderThreshold,
            #         rewardBeta=self.rewardBeta,
            #         rewardAlpha=self.rewardAlpha,
            #         speedChangePenalty=self.speedChangePenalty,
            #         rewardLOS=self.rewardLOS,
            #         stepPenalty=self.stepPenalty,
            #         gui=self.gui,
            #         non_coop_tag = self.non_coop_tag,
            #     )

            # if len(workers_to_remove) > 0:
            #     time.sleep(5)

            runner_sims = [
                workers[agent_id].run_one_iteration.remote(weights)
                for agent_id in workers.keys()
            ]

    def evaluate(self):
        # Start simulations on actors
        workers = {
            i: Runner.remote(
                i,
                self.agent_template,
                scenario_file=self.scenario_file,
                config_file=self.config_file,
                working_directory=self.working_directory,
                max_steps=self.max_steps,
                simdt=self.simdt,
                speeds=self.speeds,
                LOS=self.LOS,
                dGoal=self.dGoal,
                maxRewardDistance=self.maxRewardDistance,
                intruderThreshold=self.intruderThreshold,
                rewardBeta=self.rewardBeta,
                rewardAlpha=self.rewardAlpha,
                speedChangePenalty=self.speedChangePenalty,
                rewardLOS=self.rewardLOS,
                stepPenalty=self.stepPenalty,
                clearancePenalty=self.clearancePenalty,
                gui=self.gui,
                non_coop_tag=self.non_coop_tag,
                traffic_manager_active=self.traffic_manager_active,
                n_waypoints=self.n_waypoints,
            )
            for i in range(self.num_workers)
        }

        rewards = []
        total_nmacs = []
        total_nmac_time = []
        iteration_record = []
        total_transitions = 0
        best_reward = -np.inf

        if self.agent.equipped:
            self.agent.model.load_weights(self.weights_file)
            weights = self.agent.model.get_weights()
        else:
            weights = []

        runner_sims = [
            workers[agent_id].run_one_iteration.remote(weights)
            for agent_id in workers.keys()
        ]

        for i in range(self.iterations):
            done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
            results = ray.get(done_id)

            total_reward = []

            nmacs = []
            total_ac = []

            for result in results:
                data = ray.get(result)
                total_reward.append(float(np.sum(data[0]["raw_reward"])))
                if data[0]["environment_done"]:
                    nmacs.append(data[0]["nmacs"])
                    total_nmac_time += [data[0]["nmac_time"]]
                    total_ac.append(data[0]["total_ac"])

            mean_total_reward = np.mean(total_reward)

            for j, nmac in enumerate(nmacs):
                print(f"     Scenario Complete     ")
                print("|------------------------------|")
                print(f"| Total NMACS:      {nmac}      |")
                print(f"| Total Aircraft:   {total_ac[j]}  |")
                print("|------------------------------|")
                print(" ")
                total_nmacs.append(nmac)
                iteration_record.append(i)

            rewards.append(mean_total_reward)
            np.save("{}/eval_reward.npy".format(self.path_results), np.array(rewards))

            if len(nmacs) > 0:
                np.save(
                    "{}/eval_nmacs.npy".format(self.path_results), np.array(total_nmacs)
                )

                np.save(
                    "{}/eval_nmac_time.npy".format(self.path_results),
                    np.array(total_nmac_time),
                )

                np.save(
                    "{}/eval_iteration_record.npy".format(self.path_results),
                    np.array(iteration_record),
                )

            # total_transitions += transitions

            print(f"     Iteration {i} Complete     ")
            print("|------------------------------|")
            print(f"| Mean Total Reward:   {np.round(mean_total_reward,1)}  |")
            roll_mean = np.mean(rewards[-150:])
            print(f"| Rolling Mean Reward: {np.round(roll_mean,1)}  |")
            print("|------------------------------|")
            print(" ")

            runner_sims = [
                workers[agent_id].run_one_iteration.remote(weights)
                for agent_id in workers.keys()
            ]


### Main code execution
gin.parse_config_file("conf/config.gin")

if args.cluster:
    ## Initialize Ray
    ray.init(address=os.environ["ip_head"])
    print(ray.cluster_resources())
else:
    # check if running on Mac
    if platform.release() == "Darwin":
        ray.init(_node_ip_address="0.0.0.0", local_mode=args.debug)
    else:
        ray.init(local_mode=args.debug)
    print(ray.cluster_resources())


# Now initialize the trainer with 30 workers and to run for 100k episodes 3334 episodes * 30 workers = ~100k episodes
Trainer = Driver(cluster=args.cluster)
if Trainer.run_type == "train":
    Trainer.train()
else:
    Trainer.evaluate()

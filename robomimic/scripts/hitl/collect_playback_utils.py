"""Teleoperate robot with keyboard or SpaceMouse. """

import argparse
import numpy as np
import os
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper
import time
import numpy as np
import json
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
import robomimic
import cv2
import robomimic.utils.obs_utils as ObsUtils
import copy
import h5py

import robosuite
is_v1 = (robosuite.__version__.split(".")[0] == "1")

# Change later
GOOD_EPISODE_LENGTH = None
MAX_EPISODE_LENGTH = None
SUCCESS_HOLD = None

class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.low, self.high = env.action_spec

    def get_action(self, obs):
        return np.random.uniform(self.low, self.high) / 2

class TrainedPolicy:
    def __init__(self, checkpoint):
        from robomimic.utils.file_utils import policy_from_checkpoint
        self.policy = policy_from_checkpoint(ckpt_path=checkpoint)[0]

    def get_action(self, obs):
        obs = copy.deepcopy(obs)
        di = obs
        postprocess_visual_obs = True

        ret = {}
        for k in di:
            pass
            """
            if ObsUtils.key_is_image(k):
                ret[k] = di[k][::-1]
                if postprocess_visual_obs:
                    ret[k] = ObsUtils.process_image(ret[k])
            """
        obs.update(ret)

        return self.policy(obs)

def is_empty_input_spacemouse(action):
    # empty_input1 = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -1.000])
    empty_input = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000])
    if np.array_equal(np.abs(action), empty_input):
        return True
    return False

def terminate_condition_met(time_success, timestep_count, term_cond):
    assert term_cond in ["fixed_length", "success_count", "stop"]
    if term_cond == "fixed_length":
        return timestep_count >= GOOD_EPISODE_LENGTH and time_success > 0
    elif term_cond == "success_count":
        return time_success == SUCCESS_HOLD
    elif term_cond == "stop":
        return timestep_count >= MAX_EPISODE_LENGTH

def post_process_spacemouse_action(action, grasp, last_grasp):
    """ Fixing Spacemouse Action """
    # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
    # toggle arm control and / or camera viewing angle if requested
    if last_grasp < 0 < grasp:
        if args.switch_on_grasp:
            args.arm = "left" if args.arm == "right" else "right"
        if args.toggle_camera_on_grasp:
            cam_id = (cam_id + 1) % num_cam
            env.viewer.set_camera(camera_id=cam_id)
    # Update last grasp
    last_grasp = grasp

    if is_v1:
        env_action_dim = env.action_dim
    else:
        env_action_dim = 7

    # Fill out the rest of the action space if necessary
    rem_action_dim = env_action_dim - action.size
    if rem_action_dim > 0:
        # Initialize remaining action space
        rem_action = np.zeros(rem_action_dim)
        # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
        if args.arm == "right":
            action = np.concatenate([action, rem_action])
        elif args.arm == "left":
            action = np.concatenate([rem_action, action])
        else:
            # Only right and left arms supported
            print("Error: Unsupported arm specified -- "
                  "must be either 'right' or 'left'! Got: {}".format(args.arm))
    elif rem_action_dim < 0:
        # We're in an environment with no gripper action space, so trim the action space to be the action dim
        action = action[:env_action_dim]

    """ End Fixing Spacemouse Action """
    return action, last_grasp

def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
    
    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        env.reset()
        xml = env.postprocess_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
        if not is_v1:
            # hide teleop visualization after restoring from model
            env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
            env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    if "goal" in state:
        env.set_goal(**state["goal"])
    
    return env._get_observations(force_update=True)

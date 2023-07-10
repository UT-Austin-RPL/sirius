"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import cv2

def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
    diff_camera=[],
    diff_camera_height=-1,
    diff_camera_width=-1,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    if len(diff_camera) > 0:
        for camera_name in diff_camera:
            h = obs[camera_name + "_image"].shape[0]
            w = obs[camera_name + "_image"].shape[1]
            obs[camera_name + "_image"] = cv2.resize(obs[camera_name + "_image"],
                                          (0,0),
                                          fx=float(diff_camera_height) / h,
                                          fy=float(diff_camera_width) / w,
                                          )


    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        if len(diff_camera) > 0:
            for camera_name in diff_camera:
                h = obs[camera_name + "_image"].shape[0]
                w = obs[camera_name + "_image"].shape[1]
                obs[camera_name + "_image"] = cv2.resize(obs[camera_name + "_image"],
                                              (0,0),
                                              fx=float(diff_camera_height) / h,
                                              fy=float(diff_camera_width) / w,
                                              )

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def dataset_states_to_obs(args):
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(os.path.expanduser(args.dataset), "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    if not os.path.exists(os.path.dirname(args.output_dataset)):
        os.mkdir(os.path.dirname(args.output_dataset))
    assert args.dataset != args.output_dataset
    f_out = h5py.File(os.path.expanduser(args.output_dataset), "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(args.output_dataset))

    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
            diff_camera=args.diff_size_camera,
            diff_camera_height=args.diff_size_camera_height,
            diff_camera_width=args.diff_size_camera_width,
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        if "data/{}/action_modes".format(ep) in f:
            traj["action_modes"] = f["data/{}/action_modes".format(ep)][()]
        else:
            traj["action_modes"] = args.am_value * np.ones(len(actions))

        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        ep_data_grp.create_dataset("action_modes", data=np.array(traj["action_modes"]))
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        for k in ep_data_grp["obs"]:
            if "image" in k:
                print(ep_data_grp["obs"][k].shape)

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))


    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples

    # Save second different size camera info
    env_info_raw = env.serialize()
    if len(args.diff_size_camera) > 0:
        env_info_raw["env_kwargs"]["diff_size_camera"] = args.diff_size_camera
        env_info_raw["env_kwargs"]["diff_size_camera_height"] = args.diff_size_camera_height
        env_info_raw["env_kwargs"]["diff_size_camera_width"] = args.diff_size_camera_width

    data_grp.attrs["env_args"] = json.dumps(env_info_raw, indent=4) # environment info
    for k in f["data"].attrs.keys():
        if k not in ["total", "env_args"]:
            data_grp.attrs[k] = f["data"].attrs[k]

    data_grp.attrs["raw_hdf5_dataset"] = args.dataset

    print("Wrote {} trajectories to {}".format(len(demos), args.output_dataset))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    parser.add_argument(
        "--diff_size_camera",
        type=str,
        nargs='+',
        default=[],
        help="camera name for a different size camera.",
    )

    parser.add_argument(
        "--diff_size_camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--diff_size_camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    parser.add_argument(
        "--am_value",
        default=-1,
        type=int,
        help="specify action_mode value"
    )

    args = parser.parse_args()
    for c in args.diff_size_camera:
        assert c in args.camera_names
    dataset_states_to_obs(args)

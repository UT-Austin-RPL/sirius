import os
import json
import h5py
import argparse
import imageio
import numpy as np
import math

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

import robomimic.utils.vis_utils as VisUtils
import robomimic.utils.tensor_utils as TensorUtils
            
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def prepare_tensor(tensor, device=None):
    """
    Prepare raw observation dict from environment for policy.

    Args:
        ob (dict): single observation dictionary from environment (no batch dimension, 
            and np.array values for each key)
    """
    tensor = TensorUtils.to_tensor(tensor)
    tensor = TensorUtils.to_batch(tensor)
    if device is not None:
        tensor = TensorUtils.to_device(tensor, device)
    tensor = TensorUtils.to_float(tensor)
    return tensor

def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    algo=None,
    render=False, 
    write_video=False,
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]

    adv_vals = []
    v_vals = []
    q_vals = []
    grasp_success_t = []
    video_frames = []

    for i in range(traj_len):
        env.reset_to({"states" : states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])
            
        ob = env.get_observation()
        ob = prepare_tensor(ob, device=algo.device)
        ac = prepare_tensor(actions[i], device=algo.device)
        
        v_value = algo.get_v_value(obs_dict=ob)
        v_value = np.around(v_value.item(), decimals=3)
        v_vals.append(v_value)

        adv_value = algo.get_adv_weight(obs_dict=ob, ac=ac)
        adv_value = np.around(adv_value.item(), decimals=3)
        adv_vals.append(adv_value)

        q_value = algo.get_Q_value(obs_dict=ob, ac=ac)
        q_value = np.around(q_value.item(), decimals=3)
        q_vals.append(q_value)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    proc_img = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                    video_img.append(proc_img)
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_frames.append([i, video_img])
            video_count += 1

        if first:
            break
            
    return dict(
        v_vals=np.array(v_vals),
        q_vals=np.array(q_vals),
        adv_vals=np.array(adv_vals),
        grasp_success_t=np.array(grasp_success_t),
        video_frames=video_frames,
    )


def playback_dataset(args, plot_helper=None, video_helper=None):
    if args.vis_path is None:
        args.vis_path = os.path.abspath(
            os.path.join(args.ckpt, os.pardir, os.pardir, 'vis')
        )

    if not os.path.exists(args.vis_path):
        os.makedirs(args.vis_path)

    # some arg checking
    write_video = (args.no_video is False)
    write_plot = (args.no_plot is False)

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())

    # either sort demos in order or play back randomly
    if args.sort_demos:
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
    else:
        np.random.shuffle(demos)

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(
            os.path.join(args.vis_path, 'video.mp4'),
            fps=args.video_fps,
        )
    
    algo, ckpt_dict = FileUtils.algo_from_checkpoint(ckpt_path=args.ckpt)
    
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,
        verbose=True,
    )
    
    is_robosuite_env = True

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = f["data/{}/actions".format(ep)][()]

        ep_info = playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            algo=algo,
            render=False,
            write_video=write_video,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=False,
        )

        ep_info['action_modes'] = f["data/{}/action_modes".format(ep)][()]
        rewards = f["data/{}/rewards".format(ep)][()]
        ep_info['success'] = rewards[-1]

        if write_video:
            if video_helper is not None:
                video_frames = video_helper(
                    ep_num=ind,
                    ep_info=ep_info,
                )
            else:
                video_frames = ep_info['video_frames']

            for (_, img) in video_frames:
                video_writer.append_data(img)

        if write_plot:
            assert plot_helper is not None
            plot_helper(
                ep_num=ind,
                ep_info=ep_info,
            )
            
    f.close()
    if write_video:
        video_writer.close()

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to trained model (AWAC or IQL supported)",
    )

    parser.add_argument(
        "--vis_path",
        type=str,
        default=None,
        help="(optional) where to log videos and plots. default to experiment folder containing model",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in dataset",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="num demos to play",
    )

    parser.add_argument(
        "--video_fps",
        type=int,
        default=10,
        help="video speed",
    )

    parser.add_argument(
        "--sort_demos",
        action="store_true",
        help="whether to play demos in sorted order",
    )

    parser.add_argument(
        "--no_video",
        action="store_true",
        help="disable saving videos",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="disable saving plots",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=2,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    return parser
"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
from tqdm import tqdm
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robosuite as suite
from gym.core import Env

import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.wrappers import Wrapper

from robomimic.utils.dataset import (
    SequenceDataset, IWRDataset, WeightedDataset,
    PreintvRelabeledDataset,
)
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

from PIL import Image, ImageFont, ImageDraw

from robomimic.utils.vis_utils import write_text_on_image

import cv2

import wandb

def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y-%m-%d-%H-%M-%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = config.train.output_dir
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = "n"
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    filter_by_attribute = config.train.hdf5_filter_key

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        train_filter_by_attribute = "train"
        valid_filter_by_attribute = "valid"
        if filter_by_attribute is not None:
            train_filter_by_attribute = "{}_{}".format(filter_by_attribute, train_filter_by_attribute)
            valid_filter_by_attribute = "{}_{}".format(filter_by_attribute, valid_filter_by_attribute)
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        load_next_obs=True,  # make sure dataset returns s'
        frame_stack=1,  # no frame stacking
        seq_length=config.train.seq_length,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
        num_eps=config.train.num_eps,
        sort_demo_key=config.train.sort_demo_key,
        use_gripper_history=config.train.use_gripper_history,
        use_sampler=config.train.use_sampler,
    )

    if not config.algo.hc_weights.enabled and \
            not config.train.preintv_relabeling.enabled and \
            not config.train.use_iwr_sampling and not config.train.use_weighted_sampling \
            and not config.train.bootstrap_sampling:
        return SequenceDataset(**ds_kwargs)

    if config.train.bootstrap_sampling:
        return BootstrapDataset(**ds_kwargs)

    if config.train.use_iwr_sampling:
        assert config.train.use_weighted_sampling == False
        return IWRDataset(action_mode_selection=config.algo.hc_weights.action_mode_selection, **ds_kwargs)

    hc_config = config.algo.hc_weights
    ds_kwargs.update(
        use_hc_weights=hc_config.enabled,
        weight_key=hc_config.weight_key,
        w_demos=hc_config.demos,
        w_rollouts=hc_config.rollouts,
        w_intvs=hc_config.intvs,
        w_pre_intvs=hc_config.pre_intvs,
        normalize_weights=hc_config.normalize,
    )

    if not config.train.use_weighted_sampling:
        ds_kwargs.update(
            use_weighted_sampler=hc_config.use_weighted_sampler,
            use_iwr_ratio=hc_config.use_iwr_ratio,
            iwr_ratio_adjusted=hc_config.iwr_ratio_adjusted,
            action_mode_selection=hc_config.action_mode_selection,
            same_weight_for_seq=hc_config.same_weight_for_seq,
            use_category_ratio=hc_config.use_category_ratio,
            prenormalize_weights=hc_config.prenormalize_weights,
            give_final_percentage=hc_config.give_final_percentage,
            sirius_reweight=hc_config.sirius_reweight,
            delete_rollout_ratio=hc_config.delete_rollout_ratio,
            memory_org_type=hc_config.memory_org_type,
            not_use_preintv=hc_config.not_use_preintv,
        )

    if config.train.preintv_relabeling.enabled:
        rlbling_config = config.train.preintv_relabeling
        ds_kwargs['mode'] = rlbling_config.mode
        ds_kwargs['fixed_preintv_length'] = rlbling_config.fixed_preintv_length
        ds_kwargs['model_ckpt'] = rlbling_config.model_ckpt
        ds_kwargs['model_th'] = rlbling_config.model_th
        ds_kwargs['model_eval_mode'] = rlbling_config.model_eval_mode
        ds_kwargs['base_key'] = rlbling_config.base_key

    if config.train.preintv_relabeling.enabled:
        dataset = PreintvRelabeledDataset(**ds_kwargs)
    else:
        dataset = WeightedDataset(**ds_kwargs)

    return dataset


def run_rollout(
        policy,
        env,
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        algo=None,
        use_gripper_history=False,
):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = {k: False for k in env.is_success()}  # success metrics

    try:
        for step_i in range(horizon):

            # get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            if False:  # type(algo).__name__ == "AWAC":
                for k in ob_dict:
                    ob_dict[k] = torch.FloatTensor(ob_dict[k])[None, :]
                ac = torch.FloatTensor(ac)[None, :]
                ob_dict = policy._prepare_observation(ob_dict)
                goal_dict = policy._prepare_observation(goal_dict)
                ac = policy._prepare_observation(ac)

                # obtain adv score for this (s, a)
                adv, v_pi = algo._estimate_adv(ob_dict, goal_dict, ac)

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)

                    if False:  # type(algo).__name__ == "AWAC":
                        adv_value = np.around(adv.item(), decimals=5)
                        v_value = np.around(v_pi.item(), decimals=5)
                        # write adv score on image
                        text = "Adv score: {}\nV value: {}".format(adv_value, v_value)
                        video_img_text = write_text_on_image(video_img, text)
                        assert not (video_img == video_img_text).all()
                        video_img = video_img_text

                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


class ParallelEnvWrapper(Wrapper, Env):
    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.observation_space = self.action_space = None

    def seed(self, seed=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.reward()

    def check_success(self):
        return self.env._check_success()


def rollout_parallel(
        policy,
        env_meta,
        num_parallel,
        horizon,
        num_episodes,
        disable_low_noise_eval=False,
):
    # one policy is needed for each environment (to separate rnn internal states)
    if disable_low_noise_eval:
        policy.policy.nets["policy"].low_noise_eval = False

    policy.start_episode()  # must reset before deep copying
    policies = [deepcopy(policy) for i in range(num_parallel)]
    ac_dim = policies[0].policy.ac_dim

    # create environments using the parallel wrapper
    def make_env():
        # not necessarily included
        env_meta['env_kwargs']['env_name'] = env_meta['env_name']
        env = suite.make(
            **env_meta['env_kwargs']
        )
        env = ParallelEnvWrapper(env)
        return env

    from stable_baselines3.common.vec_env import SubprocVecEnv
    envs = [lambda: make_env() for i in range(num_parallel)]
    para_env = SubprocVecEnv(envs)
    para_env.seed(0)  # ensure each environment is initialized differently

    # helper function for obtaining agent actions
    def get_actions(all_obs):
        all_obs = deepcopy(all_obs)
        all_actions = list()

        for policy_idx, obs in enumerate(all_obs):
            to_delete = list()
            for k, v in obs.items():
                if type(v) == np.float64:
                    to_delete.append(k)
            for k in to_delete:
                del obs[k]

            # process image observations if needed
            if env_meta['env_kwargs']['use_camera_obs']:

                # post-process image for customized sizes
                if "diff_size_camera" in env_meta:
                    for camera_name in env_meta["diff_size_camera"]:
                        h = obs[camera_name].shape[0]
                        w = obs[camera_name].shape[1]

                        diff_camera_height = env_meta["diff_size_camera_height"]
                        diff_camera_width = env_meta["diff_size_camera_height"]

                        obs[camera_name] = cv2.resize(obs[camera_name],
                                                      (0,0),
                                                      fx=float(diff_camera_height) / h,
                                                      fy=float(diff_camera_width) / w,
                                                      )

                if "process_first_image" in env_meta:
                    obs["agentview_image"] = obs["agentview_image"][60:, 30:210, :] # 180, 180
                
                for k in obs.keys():
                    if "_image" in k:
                        obs[k] = obs[k].transpose(2, 0, 1)
                        obs[k] = obs[k].astype('float64') / 255

            # rename key f needed
            if 'object-state' in obs.keys():
                obs['object'] = obs['object-state']

            if not success_mask[policy_idx]:
                action = policies[policy_idx](obs)
            else:
                # saves computing
                action = np.zeros(ac_dim, dtype=np.float64)
            all_actions.append(action)

        return np.asarray(all_actions)

    # helper function to step using a success mask (stop updating environment if successful)
    # def step_mask(all_actions):
    #     for i, (remote, action) in enumerate(zip(para_env.remotes, all_actions)):
    #         if not success_mask[i]:
    #             remote.send(("step", action))
    #     para_env.waiting = True
    #
    #     return para_env.step_wait()

    if num_episodes % num_parallel != 0:
        num_episodes = num_episodes - num_episodes % num_parallel
        print('WARNING: num_episodes must be a multiple of num_parallel')
        print('WARNING: performing {} rollout episodes instead'.format(num_episodes))
    rounds = num_episodes // num_parallel

    total_eps, success_eps = 0, 0
    total_reward, total_horizon = 0, 0
    round_times = []

    for i in range(rounds):
        success_mask = np.full(num_parallel, False)
        print('Performing rollouts, round {} / {}'.format(i + 1, rounds))
        total_eps += num_parallel

        # reset environment and obtain first actions
        para_obs = para_env.reset()
        para_actions = get_actions(para_obs)

        # reset policies
        for policy in policies:
            policy.start_episode()

        start_time = time.time()
        round_horizons = list()
        for step in tqdm(range(horizon), ncols=100):
            para_obs, para_reward, _, _ = para_env.step(para_actions)
            # para_obs, para_reward, _, _ = step_mask(para_actions)

            # record successes and update mask
            for j in range(num_parallel):
                # stop recording after success
                if not success_mask[j]:
                    total_reward += para_reward[j]

            # record horizons
            new_success_mask = np.asarray(para_env.env_method('check_success', indices=None))
            round_horizons.extend([step + 1] * np.sum(new_success_mask[success_mask == False]))
            # success_mask = new_success_mask
            success_mask = success_mask | new_success_mask
            print()
            print("Round Horizons: ", round_horizons)
            print("Success: ", success_mask)

            para_actions = get_actions(para_obs)

        # these trajectories were not successful
        round_horizons.extend([horizon] * np.sum(success_mask == False))

        # update logging variables
        round_times.append(time.time() - start_time)
        total_horizon += np.mean(round_horizons)
        success_eps += np.sum(success_mask)

        print("success_eps: ", success_eps)
        print("total_eps: ", total_eps)

    # construct logs, naming is consistent with rest of robomimic
    rollout_logs = dict()
    rollout_logs['Return'] = total_reward / num_episodes
    rollout_logs['Horizon'] = total_horizon / rounds
    rollout_logs['Success_Rate'] = success_eps / total_eps
    rollout_logs['Time_Episode'] = np.sum(round_times) / 60
    rollout_logs['time'] = np.mean(round_times) / num_parallel

    print('Rollout success rate: {}'.format(round(success_eps / total_eps, 5)))
    return {env_meta['env_name']: rollout_logs}, None


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        algo=None,
        use_gripper_history=False,
):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()
    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = {k: video_path for k in envs}
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = {k: video_writer for k in envs}
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_paths = {k: os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs}
        video_writers = {k: imageio.get_writer(video_paths[k], fps=20) for k in envs}

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
                algo=algo,
                use_gripper_history=use_gripper_history,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(
            rollout_logs["time"]) / 60.  # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        top_n_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

        """ Top N Success Rate Logging """
        for rank in range(len(top_n_success_rate[env_name])):
            if rollout_logs["Success_Rate"] > top_n_success_rate[env_name][rank]:
                top_n_success_rate[env_name].insert(rank, rollout_logs["Success_Rate"])
                break
        top_n_success_rate[env_name] = top_n_success_rate[env_name][:3]

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        top_n_success_rate=top_n_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):
        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)

        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0

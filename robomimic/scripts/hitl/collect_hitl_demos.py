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

from collect_playback_utils import reset_to
import h5py

import threading
import subprocess

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
        #self.policy.policy.nets["policy"].low_noise_eval = False

    def get_action(self, obs):
        obs = copy.deepcopy(obs)
        di = obs
        postprocess_visual_obs = True

        ret = {}
        for k in di:
            if "image" in k:
                ret[k] = di[k][::-1]
                ret[k] = ObsUtils.process_obs(ret[k], obs_modality='rgb')
        obs.update(ret)
        obs.pop('frame_is_assembled', None)
        obs.pop('tool_on_frame', None)
        return self.policy(obs)

    def get_dist(self, obs):
        a, dist = self.policy.get_action_with_dist(obs)
        return a, dist

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

class Renderer:
    def __init__(self, env, render_onscreen):
        self.env = env
        self.render_onscreen = render_onscreen

        if (is_v1 is False) and self.render_onscreen:
            self.env.viewer.set_camera(camera_id=2)

    def render(self, obs):
        if is_v1:
            vis_env = self.env.env
            robosuite_env = self.env.env.env
            robosuite_env.visualize(vis_settings=vis_env._vis_settings)
        else:
            robosuite_env = self.env.env

        if self.render_onscreen:
            self.env.render()
        else:
            # if is_v1:
            #     img_for_policy = obs['agentview_image']
            # else:
            #     img_for_policy = obs['image']
            # img_for_policy = img_for_policy[:,:,::-1]
            # img_for_policy = np.flip(img_for_policy, axis=0)
            # cv2.imshow('img for policy', img_for_policy)

            img = robosuite_env.sim.render(height=700, width=1000, camera_name="agentview")
            img = img[:,:,::-1]
            img = np.flip(img, axis=0)
            cv2.imshow('offscreen render', img)
            cv2.waitKey(1)

        if is_v1:
            robosuite_env.visualize(vis_settings=dict(
                env=False,
                grippers=False,
                robots=False,
            ))

def collect_trajectory(env, device, args):
    renderer = Renderer(env, args.render_onscreen)

    obs = env.reset()
    renderer.render(obs)
    
    if not args.all_demos:
        policy.policy.start_episode()

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()

    time_success = 0
    timestep_count = 0
    num_human_samples = 0
    nonzero_ac_seen = False
    discard_traj = False
    success_at_time = -1
    grasped = False

    while True:
        if_sleep = args.sleep_time > 0

        if if_sleep and not time_success:
            if is_v1:
                time.sleep(args.sleep_time)
            else:
                time.sleep(0.02)

        if is_v1:
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]
        else:
            active_robot = None

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=args.arm,
            env_configuration=args.config
        )

        if action is None:
            discard_traj = True
            break

        action, last_grasp = post_process_spacemouse_action(action, grasp, last_grasp)
        action_mode = None

        policy_action = policy.get_action(obs)

        if is_empty_input_spacemouse(action):
            if args.all_demos:
                if not nonzero_ac_seen: # if have not seen nonzero action, should not be zero action
                    continue # if all demos, no action
                # else: okay to be zero action afterwards
                num_human_samples += 1
                action_mode = -1
            else:
                action = policy_action
                action_mode = 0
        else:
            nonzero_ac_seen = True
            num_human_samples += 1
            if args.all_demos:
                action_mode = -1 # iter 0 is viewed as non-intervention
            else:
                action_mode = 1

        if grasped and action[-1] < -0.7:
            grasped = False
            grasp_timesteps = 0
            while True:
                grasp_timesteps += 1
                action_human, grasp = input2action(
                    device=device,
                    robot=active_robot,
                    active_arm=args.arm,
                    env_configuration=args.config
                )
                if grasp_timesteps > 10 or not grasp:
                    break
            if grasp:
                continue

        elif not grasped and action[-1] > 0.7:
            grasped = True
        assert action_mode is not None
        obs, _, _, _ = env.step(action, action_mode=action_mode)
        timestep_count += 1

        renderer.render(obs)

        if env._check_success():
            time_success += 1
            if time_success == 1:
                print("Success length: ", timestep_count)
                success_at_time = timestep_count

        if terminate_condition_met(time_success=time_success,
                                   timestep_count=timestep_count,
                                   term_cond=args.term_condition):
            break

        if timestep_count > MAX_EPISODE_LENGTH:
            print("discard this trial")
            discard_traj = True
            break

    ep_directory = env.ep_directory
    env.close()
    return ep_directory, timestep_count, num_human_samples, success_at_time, discard_traj

def collect_trajectory_with_playback(
        env,
        initial_state,
        states,
        rollout_actions=None,
):
    renderer = Renderer(env, args.render_onscreen)

    # load the initial state
    env.reset()
    obs = reset_to(env, initial_state)

    policy.policy.start_episode()

    renderer.render(obs)

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()

    time_success = 0
    timestep_count = 0
    num_human_samples = 0
    nonzero_ac_seen = False
    discard_traj = False
    success_at_time = -1
    grasped = False

    traj_len = states.shape[0]

    human_intervened = False
    i = 0

    while True:

        i += 1

        if_sleep = args.sleep_time > 0 

        if if_sleep and not time_success:
            if is_v1:
                time.sleep(args.sleep_time)
            else:
                time.sleep(0.02)

        if is_v1:
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]
        else:
            active_robot = None

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=args.arm,
            env_configuration=args.config
        )

        if action is None:
            discard_traj = True
            break

        action, last_grasp = post_process_spacemouse_action(action, grasp, last_grasp)
        action_mode = None

        policy_action = policy.get_action(obs)

        if is_empty_input_spacemouse(action):
            if not human_intervened:
                action = rollout_actions[i]
                action_mode = 0
            else:
                action = policy_action
                action_mode = 0
        else:
            # Human start intervene, no more playback actions
            human_intervened = True

            nonzero_ac_seen = True
            num_human_samples += 1
            if args.all_demos:
                action_mode = -1 # iter 0 is viewed as non-intervention
            else:
                action_mode = 1

        if grasped and action[-1] < -0.7:
            grasped = False
            grasp_timesteps = 0
            while True:
                grasp_timesteps += 1
                action_human, grasp = input2action(
                    device=device,
                    robot=active_robot,
                    active_arm=args.arm,
                    env_configuration=args.config
                )
                if grasp_timesteps > 10 or not grasp:
                    break
            if grasp:
                continue

        elif not grasped and action[-1] > 0.7:
            grasped = True
        assert action_mode is not None
        obs, _, _, _ = env.step(action, action_mode=action_mode)
        timestep_count += 1

        renderer.render(obs)

        if env._check_success():
            time_success += 1
            if time_success == 1:
                print("Success length: ", timestep_count)
                success_at_time = timestep_count

        if terminate_condition_met(time_success=time_success,
                                   timestep_count=timestep_count,
                                   term_cond=args.term_condition):
            break

        if timestep_count > MAX_EPISODE_LENGTH:
            print("discard this trial")
            discard_traj = True
            break

    ep_directory = env.ep_directory
    env.close()
    return ep_directory, timestep_count, num_human_samples, success_at_time, discard_traj


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir)), "datasets", "raw"),
    )

    parser.add_argument("--environment", type=str, default="NutAssemblySquare")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.8, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.8, help="How much to scale rotation user inputs")

    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-traj", type=int, default=50, help="Number of trajectories to collect / evaluate")
    parser.add_argument("--training-iter", type=int, default=-1)
    parser.add_argument("--human-sample-ratio", type=float, default=None)
    parser.add_argument("--human-sample-number", type=float, default=None)
    parser.add_argument("--all-demos", action="store_true")
    parser.add_argument("--term-condition", default="fixed_length", type=str)
    parser.add_argument("--render-onscreen", action="store_true")
    parser.add_argument("--sleep-time", default=0.08, type=float)
    parser.add_argument("--playback-dataset", type=str, default=None)

    # arguments for processing datasets
    parser.add_argument(
        "--no-processing", action='store_true',
        help='(Optional) do not process raw collected data'
    )
    parser.add_argument(
        "--im-size", type=str, default='',
        help='(Optional) What image size should the processed dataset have'
    )

    # arguments for merging datasets
    parser.add_argument(
        '--merge-with', type=str, default=None,
        help='(Optional) the path for the dataset to merge with'
    )
    parser.add_argument(
        '--merge-output-path', type=str, default=None,
        help='(Optional) if merging, where to output the merged dataset'
    )
    parser.add_argument(
        '--max-merge-size', type=int, default=-1,
        help='(Optional) if merging, the max size of the merged dataset; \
              excess trajectories are removed from the newly collected dataset'
    )

    args = parser.parse_args()

    if args.checkpoint is not None:
        assert args.training_iter > 0

    if args.merge_with is not None:
        assert not args.no_processing  # dataset must be processed to be merged
        assert args.merge_output_path is not None

    script_start_time = time.time()

    if is_v1:
        # Get controller config
        controller_config = load_controller_config(default_controller=args.controller)

        # Create argument configuration
        config = {
            "env_name": args.environment,
            "robots": args.robots,
            "controller_configs": controller_config,
        }

        # Check if we're using a multi-armed environment and use env_configuration argument if so
        if "TwoArm" in args.environment:
            config["env_configuration"] = args.config
    else:
        config = {
            "env_name": args.environment,
            "control_freq": 20, #40
        }

    if args.checkpoint is not None:
        assert args.render_onscreen is False
        from robomimic.utils.file_utils import env_from_checkpoint
        env = env_from_checkpoint(ckpt_path=args.checkpoint, render=False, render_offscreen=True)[0].env
        policy = TrainedPolicy(args.checkpoint)
    else:
        if is_v1:
            env = suite.make(
                **config,
                has_renderer=args.render_onscreen,
                has_offscreen_renderer=(not args.render_onscreen),
                render_camera=args.camera,
                ignore_done=True,
                use_camera_obs=(not args.render_onscreen),
                reward_shaping=False,
                control_freq=20,
            )
        else:
            env = suite.make(
                **config,
                has_renderer=args.render_onscreen,
                has_offscreen_renderer=(not args.render_onscreen),
                # render_camera=args.camera,
                ignore_done=True,
                use_camera_obs=(not args.render_onscreen),
                reward_shaping=False,
                gripper_visualization=True,
            )
        policy = RandomPolicy(env)

    if is_v1:
        from robosuite.wrappers import VisualizationWrapper
        # Wrap this environment in a visualization wrapper
        env = VisualizationWrapper(env)#, disable_vis=True)

    if args.environment == 'NutAssemblySquare':
        GOOD_EPISODE_LENGTH = 100
        MAX_EPISODE_LENGTH = 400
        SUCCESS_HOLD = 5
    elif args.environment == 'PandaCoffee':
        GOOD_EPISODE_LENGTH = 350
        MAX_EPISODE_LENGTH = 600
        SUCCESS_HOLD = 10
    elif args.environment == 'PandaCircus':
        GOOD_EPISODE_LENGTH = 400
        MAX_EPISODE_LENGTH = 700
        SUCCESS_HOLD = 10
    elif args.environment == 'Stack':
        GOOD_EPISODE_LENGTH = 100
        MAX_EPISODE_LENGTH = 100
        SUCCESS_HOLD = 5
    elif args.environment == 'Lift':
        GOOD_EPISODE_LENGTH = 200
        MAX_EPISODE_LENGTH = 300
        SUCCESS_HOLD = 5
    elif args.environment == 'ToolHang':
        GOOD_EPISODE_LENGTH = 2000
        MAX_EPISODE_LENGTH = 3000
        SUCCESS_HOLD = 5
    else:
        raise NotImplementedError

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)
    tmp_directory = "/tmp/{}".format(str(script_start_time).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    start = time.perf_counter()

    total_human_samples = 0
    total_samples = 0
    agent_success = 0
    success_at_time_lst = []

    excluded_eps = []

    t1, t2 = str(script_start_time).split(".")
    hdf5_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(hdf5_dir)

    num_traj_saved = 0
    num_traj_discarded = 0

    # Playback from (failure) data
    if args.playback_dataset is not None:
        f = h5py.File(args.playback_dataset, "r")
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        at_ind = 0
        discard_times = 5
        while at_ind < len(inds):
            ep = demos[at_ind]
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

            actions = f["data/{}/actions".format(ep)][()]

            ep_directory, timestep_count, num_human_samples, success_at_time, discard_traj = \
                collect_trajectory_with_playback(env, initial_state, states, rollout_actions=actions)

            print("\tkeep this traj? ", not discard_traj)
            print("\tsuccess at time: ", success_at_time)
            print()
            if discard_traj:
                excluded_eps.append(ep_directory.split('/')[-1])
                num_traj_discarded += 1
                discard_times -= 1
                if discard_times == 0:
                    # move to the next traj regardless
                    at_ind += 1
            else:
                total_human_samples += num_human_samples
                total_samples += timestep_count
                if num_human_samples == 0:
                    agent_success += 1
                success_at_time_lst.append(success_at_time)

                num_traj_saved += 1

                # move to the next traj
                at_ind += 1

            meta_stats = dict(
                training_iter = args.training_iter,
                checkpoint = 'None' if args.checkpoint is None else args.checkpoint,
                total_time = time.perf_counter() - start,

                num_traj_saved = num_traj_saved,
                num_traj_discarded = num_traj_discarded,
                num_agent_success = agent_success,
                total_human_samples = total_human_samples,
                total_samples = total_samples,
            )

            print("AGENT SUCCESS: ", agent_success)
            print("AGENT FAILURE: ", num_traj_saved - agent_success)

            gather_demonstrations_as_hdf5(tmp_directory, hdf5_dir, env_info, excluded_eps, meta_stats)
            print("At traj: ", at_ind)

            if args.human_sample_number is not None:
                threshold = args.human_sample_number
                print("progress: {}%".format(int(total_human_samples / threshold * 100)))
                if total_human_samples >= threshold:
                    break

    else:
        while True:
            print("Collecting traj # {}".format(num_traj_saved + 1))
            ep_directory, timestep_count, num_human_samples, success_at_time, discard_traj = collect_trajectory(env, device, args)
            print("\tkeep this traj? ", not discard_traj)
            print("\tsuccess at time: ", success_at_time)
            print()
            if discard_traj:
                excluded_eps.append(ep_directory.split('/')[-1])
                num_traj_discarded += 1
            else:
                total_human_samples += num_human_samples
                total_samples += timestep_count
                if num_human_samples == 0:
                    agent_success += 1
                success_at_time_lst.append(success_at_time)

                num_traj_saved += 1

            meta_stats = dict(
                training_iter = args.training_iter,
                checkpoint = 'None' if args.checkpoint is None else args.checkpoint,
                total_time = time.perf_counter() - start,

                num_traj_saved = num_traj_saved,
                num_traj_discarded = num_traj_discarded,
                num_agent_success = agent_success,
                total_human_samples = total_human_samples,
                total_samples = total_samples,
            )

            print("AGENT SUCCESS: ", agent_success)
            print("AGENT FAILURE: ", num_traj_saved - agent_success)

            gather_demonstrations_as_hdf5(tmp_directory, hdf5_dir, env_info, excluded_eps, meta_stats)

            if args.human_sample_ratio is not None:
                threshold = int(GOOD_EPISODE_LENGTH * args.num_traj * args.human_sample_ratio)
                print("# human samples: ", total_human_samples)
                print("progress: {}%".format(int(total_human_samples / threshold * 100)))
                print()
                if total_human_samples >= threshold:
                    break
            elif args.human_sample_number is not None:
                threshold = args.human_sample_number
                print("progress: {}%".format(int(total_human_samples / threshold * 100)))
                if total_human_samples >= threshold:
                    break
            else:
                if num_traj_saved == args.num_traj:
                    break

    device.thread._delete()

    def count_traj(dataset_path):
        with h5py.File(dataset_path, 'r') as f:
            return len(f['data'])

    # process data here
    if not args.no_processing:
        raw_path = os.path.join(hdf5_dir, 'demo.hdf5')
        processed_path = os.path.join(hdf5_dir, 'demo_processed.hdf5')
        print('processing dataset...')
        if len(args.im_size) > 0:
            subprocess.run(['bash', 'template_process_sim_dataset.sh', raw_path, processed_path, args.im_size])
        else:
            subprocess.run(['bash', 'template_process_sim_dataset.sh', raw_path, processed_path])

        # merge datasets here
        if args.merge_with is not None:
            sizes = [count_traj(args.merge_with), count_traj(processed_path)]
            max_size = args.max_merge_size if args.max_merge_size != -1 else sum(sizes)
            assert sizes[0] < max_size
            if sum(sizes) > max_size:
                sizes[1] -= sum(sizes) - max_size

            print('merging datasets...')
            subprocess.run([
                'python', 'merge_datasets.py',
                '--datasets', args.merge_with, processed_path,
                '--output_dataset', args.merge_output_path,
                '--num_lst', *[str(s) for s in sizes]
            ])

            print('\n### dataset merging ###')
            print('merged with:', args.merge_with)
            print('combined size:', count_traj(args.merge_output_path))
            print('merged dataset saved at:', args.merge_output_path)

            # Adding action modes to pure rollout data
            subprocess.run([
                'python', 'add_action_modes.py',
                '--dataset', args.merge_output_path,
                '--value', "0", # only add rollouts
            ])

            # Add intervention labels
            subprocess.run([
                'python', 'add_hc_weights.py',
                '--dataset', args.merge_output_path,
            ])

        if args.playback_dataset is not None:
            intv_traj_count = count_traj(processed_path)
            intv_traj_count_total = count_traj(args.playback_dataset)

            succ_traj_count = count_traj(args.merge_with)
            succ_traj_count_to_save = int(succ_traj_count * (intv_traj_count / intv_traj_count_total))
            succ_traj_count_to_delete = succ_traj_count - succ_traj_count_to_save

            print()
            print("Using playback dataset: Processing successful trajectories...")
            print("Deleting {} / {} extra successful trajectories from dataset.".format(
                succ_traj_count_to_delete,
                succ_traj_count,
            ))

            subprocess.run([
                'python', 'merge_datasets_delete_some_rollouts.py',
                '--dataset', args.merge_output_path,
                '--output_dataset', args.merge_output_path[:-5] + "_final.hdf5",
                '--delete-num', str(succ_traj_count_to_delete),
            ])

            print()
            print("Merged dataset saved at: ", args.merge_output_path[:-5] + "_final.hdf5")

            # Get intervention ratio info
            subprocess.run([
                'python', 'count_intv.py',
                '--dataset', args.merge_output_path[:-5] + "_final.hdf5",
            ])

        else:
            print()
            print("Merged dataset saved at: ", args.merge_output_path)

            # Get intervention ratio info
            subprocess.run([
                'python', 'count_intv.py',
                '--dataset', args.merge_output_path,
            ])

        print('\n### dataset processing ###')
        print('image size:', args.im_size)
        print('number of trajectory saved successfully:', count_traj(processed_path))
        print('raw dataset saved at:', raw_path)
        print('processed dataset saved at:', processed_path)

print("\n### meta stats ###")
for (k, v) in meta_stats.items():
    print("{k}: {v}".format(k=k, v=v))

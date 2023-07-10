"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

import wandb

import sys

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hdf5_path,
            obs_keys,
            dataset_keys,
            frame_stack=1,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=True,
            num_eps=None,  # for loading different num of trajectories
            sort_demo_key=None,
            use_gripper_history=False,  # if using gripper history
            hc_weights_dict=None,
            use_sampler=False,
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)
        Args:
            hdf5_path (str): path to hdf5
            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset
            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset
            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).
            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).
            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).
            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).
            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.
            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals
            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all
                non-image data. Set to None to use no caching - in this case, every batch sample is
                retrieved via file i/o. You should almost never set this to None, even for large
                image datasets.
            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.
            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.
            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load
            load_next_obs (bool): whether to load next_obs from the dataset
        """
        super(SequenceDataset, self).__init__()

        self.use_sampler = use_sampler

        self.hc_weights_dict = hc_weights_dict

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.num_eps = num_eps
        self.sort_demo_key = sort_demo_key
        self.use_gripper_history = use_gripper_history
        self.load_demo_info(filter_by_attribute=self.filter_by_attribute,
                            num_eps=self.num_eps,
                            sort_demo_key=self.sort_demo_key,
                            )

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # data to be overriden
        self._data_override = {demo_id: dict() for demo_id in self.demos}

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory
            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs,
                use_gripper_history=self.use_gripper_history,
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # debug for IWR: should not delete here, need hdf5 cache later
                if self.__class__.__name__ not in ["IWRDataset", "WeightedDataset"]:
                    self._delete_hdf5_cache()
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    def sort_demos(self, key):

        def rollout_p(d):
            action_modes = self.hdf5_file["data/" + d]["action_modes"][()]
            rollout_percentage = sum(abs(action_modes) != 1) / len(action_modes)
            return rollout_percentage

        def round_info(d):
            round_num = self.hdf5_file["data/" + d]["round"][()][0]
            if round_num == 0: 
                # human demonstration class
                # confirm round info is correct
                assert (self.hdf5_file["data/" + d]["action_modes"][()] == -1).all()
            return round_num

        assert key in ["MFI", "LFI", "FILO", "FIFO"]
       
        import random
        random.shuffle(self.demos)

        if key == "LFI": 
            self.demos.sort(key=lambda x: rollout_p(x), reverse=True)
        elif key == "MFI":
            self.demos.sort(key=lambda x: rollout_p(x))
        elif key == "FILO":
            self.demos.sort(key=lambda x: round_info(x))
        elif key == "FIFO":
            self.demos.sort(key=lambda x: round_info(x), reverse=True)

    def load_demo_info(self,
                       filter_by_attribute=None,
                       demos=None,
                       num_eps=None,
                       sort_demo_key=None,
                       ):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load
            demos (list): list of demonstration keys to load from the hdf5 file. If
                omitted, all demos in the file (or under the @filter_by_attribute
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            if type(filter_by_attribute) is str:
                self.demos = [elem.decode("utf-8") for elem in
                              np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
            elif type(filter_by_attribute) is list:
                demos_lst = []
                for filter_key in filter_by_attribute:
                    demos_lst.extend([elem.decode("utf-8") for elem in
                              np.array(self.hdf5_file["mask/{}".format(filter_key)][:])])
                assert len(demos_lst) == len(set(demos_lst))
                self.demos = demos_lst
            else:
                raise Error
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        if sort_demo_key is not None:
            self.sort_demos(sort_demo_key)

        if num_eps is not None:
            self.demos = self.demos[:num_eps]  # choose the first num_eps

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs, use_gripper_history):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.
        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset
        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            if not use_gripper_history:
                # get obs
                all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in
                                       obs_keys}
                if load_next_obs:
                    all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype('float32')
                                                for k in obs_keys}
            else:
                # get obs with gripper history information
                all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in
                                       obs_keys}
                history = self._get_gripper_history(all_data[ep]["obs"])
                all_data[ep]["obs"]["robot0_gripper_qpos"] = history
                if load_next_obs:
                    all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype('float32')
                                                for k in obs_keys}
                    history = self._get_gripper_history(all_data[ep]["next_obs"])
                    all_data[ep]["next_obs"]["robot0_gripper_qpos"] = history
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    if k == "pull_rollout":
                        if (hdf5_file["data/{}/action_modes".format(ep)][()] == 0).all():
                            all_data[ep][k] = np.array([1] * len(hdf5_file["data/{}/action_modes".format(ep)][()])).astype('float32')
                        else:
                            all_data[ep][k] = np.array([0] * len(hdf5_file["data/{}/action_modes".format(ep)][()])).astype('float32')
                        continue
                    elif k == "round":
                        all_data[ep][k] = np.array([3] * len(hdf5_file["data/{}/states".format(ep)][()])).astype('float32')
                        continue
                    raise ValueError("key {} does not exist!".format(k))
                    # all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def _get_gripper_history(self, obs_data):
        gripper = obs_data["robot0_gripper_qpos"]
        gripper_start = np.array([gripper[0] for i in range(4)])
        gripper_with_history = np.concatenate([gripper_start, gripper], axis=0)
        history = np.array([np.reshape(gripper_with_history[i:i + 5], 10) for i in range(len(gripper))])
        return history

    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations
        (per dimension and per obs key) and returns it.
        """

        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = {k: {} for k in traj_obs_dict}
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True)  # [1, ...]
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0,
                                                                                                keepdims=True)  # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("SequenceDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = {k: {} for k in merged_stats}
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
            obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.
        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            try:
                ret = self.hdf5_file[hd5key]
            except:
                import pdb; pdb.set_trace()
                self.hdf5_file[hd5key]
                print("hd5key missed: ", hd5key)

        # override as necessary
        ret = self._data_override[ep].get(key, ret)

        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )
        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(meta["obs"], obs_normalization_stats=self.obs_normalization_stats)

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )
            if self.hdf5_normalize_obs:
                meta["next_obs"] = ObsUtils.normalize_obs(meta["next_obs"],
                                                          obs_normalization_stats=self.obs_normalization_stats)

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                goal = ObsUtils.normalize_obs(goal, obs_normalization_stats=self.obs_normalization_stats)
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        meta["index"] = index

        return meta

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index].astype("float32")

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1,
                                   prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"
        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        # prepare image observations from dataset
        return ObsUtils.process_obs_dict(obs)

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        if not self.use_sampler: 
            return None
        
        weights = np.ones(len(self))

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _delete_hdf5_cache(self):
        # don't need the previous cache anymore
        del self.hdf5_cache
        self.hdf5_cache = None


class IWRDataset(SequenceDataset):
    """
    A dataset class that is useful for performing dataset operations needed by some
    human-in-the-loop algorithms, such as labeling good and bad transitions depending
    on when "interventions" occurred. This information can be used by algorithms
    that perform policy regularization.
    """

    def __init__(self, 
            action_mode_selection=0,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert "action_modes" in self.dataset_keys

        self.action_mode_selection = action_mode_selection
        assert action_mode_selection in [0, -1] # only start and end

        self.action_mode_cache = np.array([self.get_action_mode(i) for i in range(len(self))])

        if self.hdf5_cache_mode == "all":
            self._delete_hdf5_cache()

    def get_action_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=["action_modes"],
            seq_length=self.seq_length
        )

        return meta['action_modes'][self.action_mode_selection]

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        weights = np.zeros(len(self))

        for index in range(len(self)):
            # intervention (s, a) get up-weighted
            is_intervention = self.action_mode_cache[index] == 1
            if is_intervention:
                num_int = np.sum(self.action_mode_cache == 1)
                weights[index] = (len(self.action_mode_cache) - num_int) / num_int
            else:
                weights[index] = 1.

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler


class WeightedDataset(SequenceDataset):
    def __init__(
            self,
            use_hc_weights=False,
            weight_key="intv_labels",
            w_demos=10,
            w_rollouts=1,
            w_intvs=10,
            w_pre_intvs=0.1,
            normalize_weights=False,
            update_weights_at_init=True,
            use_weighted_sampler=False,
            use_iwr_ratio=False,
            iwr_ratio_adjusted=False,
            action_mode_selection=0,
            same_weight_for_seq=False,
            use_category_ratio=False,
            prenormalize_weights=False,
            give_final_percentage=False,
            sirius_reweight=False,
            delete_rollout_ratio=-1,
            memory_org_type=None,
            not_use_preintv=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_hc_weights = use_hc_weights
        self.weight_key = weight_key
        self.w_demos = w_demos
        self.w_rollouts = w_rollouts
        self.w_intvs = w_intvs
        self.w_pre_intvs = w_pre_intvs
        self.normalize_weights = normalize_weights
        self.use_weighted_sampler = use_weighted_sampler
        self.use_iwr_ratio = use_iwr_ratio
        self.iwr_ratio_adjusted = iwr_ratio_adjusted
        self.action_mode_selection = action_mode_selection
        self.same_weight_for_seq = same_weight_for_seq

        self.use_category_ratio = use_category_ratio
        self.prenormalize_weights = prenormalize_weights
        self.give_final_percentage = give_final_percentage
        self.sirius_reweight = sirius_reweight

        self.delete_rollout_ratio = delete_rollout_ratio

        self.memory_org_type = memory_org_type

        assert use_category_ratio + \
               use_iwr_ratio + \
               prenormalize_weights + \
               give_final_percentage + \
               sirius_reweight <= 1

        assert action_mode_selection in [0, -1]

        self._weights = np.ones((len(self), self.seq_length))
        if update_weights_at_init:
            self._update_weights()

        self.not_use_preintv = not_use_preintv
        if self.not_use_preintv:
            self.action_mode_selection = -1 # for sampling purpose

    def _get_rounds_labels(self):
        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.rounds_key,),
                seq_length=self.seq_length,
            )[self.rounds_key]
            labels.append(label)

        labels = np.stack(labels)
        return labels

    def _get_action_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=[self.weight_key],
            seq_length=self.seq_length
        )
        return meta[self.weight_key][self.action_mode_selection]

    def _get_iwr_ratio(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])
        
        num_int = np.sum(self.action_mode_cache == 1)
        total_num = len(self.action_mode_cache)
        weight_intv = (total_num - num_int) / num_int
        self.w_demos = 1.
        self.w_rollouts = 1.
        self.w_intvs = weight_intv
        self.w_pre_intvs = 1.

        if self.iwr_ratio_adjusted:
            if num_int != 0:
                weight_intv = total_num / (2 * num_int) 
                weight_non_intv = total_num / (2 * (total_num - num_int)) 
            else:
                weight_intv = 1
                weight_non_intv = 1
            self.w_demos = weight_non_intv
            self.w_rollouts = weight_non_intv
            self.w_intvs = weight_intv
            self.w_pre_intvs = weight_non_intv

    def _get_category_ratio(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        weight_intv = total_num / num_int
        weight_demos = total_num / num_demos

        self.w_demos = weight_demos
        self.w_intvs = weight_intv

    def _prenormalize_weights(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        weight_intv = total_num / num_int
        weight_demos = total_num / num_demos
        weight_rollouts = total_num / num_rollouts
        weight_pre_intv = total_num / num_pre_intv

        self.w_demos *= weight_demos
        self.w_intvs *= weight_intv
        self.w_rollouts *= weight_rollouts
        self.w_pre_intvs *= weight_pre_intv

    def _sirius_reweight(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        ratio_intv = num_int / total_num
        ratio_demos = num_demos / total_num
        ratio_rollouts = num_rollouts / total_num
        ratio_pre_intv = num_pre_intv / total_num

        weight_intv = 0.5
        weight_preintv = 0.002

        self.w_demos = 1
        self.w_intvs = weight_intv / ratio_intv
        self.w_rollouts = (1 - weight_intv - ratio_demos - weight_preintv) / ratio_rollouts
        self.w_pre_intvs = weight_preintv / ratio_pre_intv

        print("ratio_intv: ", ratio_intv)
        print("ratio_pre_intv: ", ratio_pre_intv)

        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _update_weights(self):
        print("Updating weights...")
        if not self.use_hc_weights or self.use_weighted_sampler:
            self._weights = np.ones(len(self))
            print("Done.")
            return

        if self.memory_org_type is not None:
            self._get_deleted_index_memory(self.memory_org_type)
            assert self.memory_save_cache is not None

        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.weight_key,),
                seq_length=self.seq_length,
            )[self.weight_key]

            if self.same_weight_for_seq:
                label = np.array([label[self.action_mode_selection]] * self.seq_length)

            labels.append(label)

        labels = np.stack(labels)

        if self.use_iwr_ratio:
            self._get_iwr_ratio()

        if self.use_category_ratio:
            self._get_category_ratio()

        if self.prenormalize_weights:
            self._prenormalize_weights()

        if self.sirius_reweight:
            self._sirius_reweight()

        weight_dict = {
            -1: self.w_demos,
            0: self.w_rollouts,
            1: self.w_intvs,
            -10: self.w_pre_intvs,
        }

        weights = np.ones((len(self), self.seq_length))
        assert weights.shape == labels.shape

        for (l, w) in weight_dict.items():
            inds = np.where(labels == l)
            weights[inds] = w

        if self.normalize_weights or self.not_use_preintv: # normalize since preintv is gone
            print("Mean weight before normalization", np.mean(weights))
            weights /= np.mean(weights)
            print("Mean weight after normalization", np.mean(weights))

        self._weights = weights
        print("Done.")

    def __getitem__(self, index):
        meta = super().__getitem__(index)

        if not self.use_weighted_sampler:
            meta["hc_weights"] = self._weights[index]

        return meta

    def _get_weights_labels(self):
        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.weight_key,),
                seq_length=1,
            )[self.weight_key][0]
            labels.append(label)
        labels = np.array(labels)
        return labels

    def _memory_org_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        print(self.memory_save_cache)
        unique, counts = np.unique(self.memory_save_cache, return_counts=True)
        print("unique value: ", unique)
        print("counts: ", counts)

        #assert list(set(self.memory_save_cache)) == [0, 1]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.memory_save_cache,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _no_preintv_sampler(self):

        self.action_mode_selection = -1

        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        ones = np.ones(self.action_mode_cache.shape)
        zeros = np.zeros(self.action_mode_cache.shape)

        self.no_preintv_sampling = np.where(self.action_mode_cache == -10, zeros, ones)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.no_preintv_sampling,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        if self.not_use_preintv:
            return self._no_preintv_sampler()

        if self.memory_org_type is not None:
            return self._memory_org_sampler()

        if self.use_sampler and not self.use_weighted_sampler: # simply sample from sampler

            weights = np.ones(len(self))

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(self),
                replacement=True,
            )
            return sampler

        if (not self.use_weighted_sampler) and (not self.rounds_resampling):
            return None

        print("Creating weighted sampler...")

        weights = np.zeros(len(self))

        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.weight_key,),
                seq_length=1,
            )[self.weight_key][0]
            labels.append(label)
        labels = np.array(labels)

        weight_dict = {
            -1: self.w_demos,
            0: self.w_rollouts,
            1: self.w_intvs,
            -10: self.w_pre_intvs,
        }

        for (l, w) in weight_dict.items():
            inds = np.where(labels == l)[0]
            if len(inds) > 0:
                weights[inds] = w / len(inds)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        print("Done.")
        return sampler


class PreintvRelabeledDataset(WeightedDataset):
    def __init__(
            self,
            mode='fixed',
            fixed_preintv_length=15,
            model_ckpt=None,
            model_th=-0.30,
            model_eval_mode='V',
            base_key="action_modes",
            *args, **kwargs
    ):
        super().__init__(*args, update_weights_at_init=False, **kwargs)

        self._label_key = "intv_labels"
        self._base_key = base_key

        # DO NOT REMOVE THIS LINE. because we are relabeling, check we are not caching __getitem__ calls
        assert self.hdf5_cache_mode is not "all"

        assert mode in ['fixed', 'model']
        self._mode = mode

        assert isinstance(fixed_preintv_length, int)
        self._fixed_preintv_length = fixed_preintv_length

        if self._mode == 'model':
            assert model_ckpt is not None
            model, _ = FileUtils.algo_from_checkpoint(ckpt_path=model_ckpt)
            self._model = model
        else:
            self._model = None

        assert model_eval_mode in ['V', 'Q', 'A']
        self._model_eval_mode = model_eval_mode

        self._model_th = model_th

        self._relabeled_values_cache = dict()

        print("Relabeling pre-interventions in dataset...")
        for demo_id in LogUtils.custom_tqdm(self.demos):
            demo_length = self._demo_id_to_demo_length[demo_id]
            ep_info = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=[self._base_key, "actions"],
                seq_length=demo_length
            )

            if self._model is not None:
                ep_info["obs"] = self.get_obs_sequence_from_demo(
                    demo_id,
                    index_in_demo=0,
                    keys=self._model.obs_key_shapes.keys(),
                    seq_length=demo_length
                )

            intv_labels = self._get_intv_labels(ep_info)
            self._data_override[demo_id][self._label_key] = intv_labels
        print("Done.")
        self._update_weights()  # update the weights after relabeling data

    def _get_intv_labels(self, ep_info):
        action_modes = ep_info[self._base_key]
        intv_labels = deepcopy(action_modes)

        intv_inds = np.reshape(np.argwhere(action_modes == 1), -1)
        intv_start_inds = [i for i in intv_inds if i > 0 and action_modes[i - 1] != 1]
        for i_start in intv_start_inds:
            for j in range(i_start - 1, -1, -1):
                if self._mode == 'fixed':
                    if j in intv_inds or i_start - j > self._fixed_preintv_length:
                        break
                elif self._mode == 'model':
                    ob = {k: ep_info["obs"][k][j] for k in ep_info["obs"].keys()}
                    ob = self._prepare_tensor(ob, device=self._model.device)

                    if self._model_eval_mode == 'V':
                        val = self._model.get_v_value(obs_dict=ob)
                    elif self._model_eval_mode == 'Q':
                        raise NotImplementedError
                    elif self._model_eval_mode == 'A':
                        raise NotImplementedError
                    else:
                        raise ValueError

                    if j in intv_inds or val > self._model_th:
                        break
                else:
                    raise ValueError

                intv_labels[j] = -10
        return intv_labels

    def _prepare_tensor(self, tensor, device=None):
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

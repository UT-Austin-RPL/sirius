# import slurm
# from slurm.util.arguments.base_args import *
from robomimic.scripts.slurm.base_args import *

import os

import robomimic
robomimic_base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

slurm_path = os.path.join(robomimic_base_path, 'robomimic/scrpts/slurm')


def add_batchrl_config_args():
    """
    Adds batchrl config args to command line arguments list
    """
    # Define namespace
    prefix = 'batchrl_config'
    actions = {
        "const": prefix,
        "action": GroupedAction
    }
    parser.add_argument(
        '--algo',
        type=str,
        default='gti',
        # choices=ALGOS,
        help='algo to use for batchRL',
        **actions
    )
    parser.add_argument(
        '--header_str',
        type=str,
        default='',
        help='optional header string to add to naming scheme (e.g.: task name)',
        **actions
    )
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        help='batch to use',
        **actions
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=slurm_path + "/log/",
        help='directory to save logs',
        **actions
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='If specified, checkpoint to load before initializing training',
        **actions
    )
    parser.add_argument(
        '--new_config_dir',
        type=str,
        default=slurm_path + "/configs/batchRL/",
        help='location to save new configs',
        **actions
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=400,
        help='Horizon for rollouts',
        **actions
    )
    parser.add_argument(
        '--n_rollouts',
        type=int,
        default=30,
        help='Number of rollouts',
        **actions
    )
    parser.add_argument(
        '--save_every_n_epochs',
        type=int,
        default=100,
        help='How often to save the current model',
        **actions
    )
    parser.add_argument(
        '--rollout_every_n_epochs',
        type=int,
        default=100,
        help='How often to rollout the current model',
        **actions
    )
    parser.add_argument(
        '--validate_every_n_epochs',
        type=int,
        default=10,
        help='How often to run validation epoch',
        **actions
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size during training',
        **actions
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=25,
        help='Sequence size for, e.g., RNNs / goal planning',
        **actions
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1001,
        help='How many epochs to run',
        **actions
    )
    parser.add_argument(
        '--use_random_crops',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use random crops when encoding images',
        **actions
    )
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=60,
        help='Crop dimension',
        **actions
    )
    parser.add_argument(
        '--n_crops',
        type=int,
        default=10,
        help='How many crops to use',
        **actions
    )
    parser.add_argument(
        '--crop_modalities',
        type=str,
        nargs='+',
        default=None,
        help='Modalities to use for observations',
        **actions
    )
    parser.add_argument(
        '--pos_enc_crops',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use positional encoding as part of random crops feature space',
        **actions
    )
    parser.add_argument(
        '--num_data_workers',
        type=int,
        default=2,
        help='How many processes to use for data grabbing if not storing HDF5 datasets in memory (RAM)',
        **actions
    )
    parser.add_argument(
        '--hdf5_in_memory',
        type=str,
        default="True",
        choices=BOOL_CHOICES,
        help='Whether to cache hdf5 in memory for fast processing',
        **actions
    )

    parser.add_argument(
        '--rebalance_dataset',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to rebalance dataset or not',
        **actions
    )
    parser.add_argument(
        '--balance_keys',
        type=str,
        nargs='+',
        default=None,
        help='Keys to match when determining which samples to reweight in dataset',
        **actions
    )
    parser.add_argument(
        '--balance_values',
        type=float,
        nargs='+',
        default=None,
        help='Values to match when determining which samples to reweight in dataset',
        **actions
    )
    parser.add_argument(
        '--balance_heuristic',
        type=float,
        default=1.0,
        help='Balancing heuristic to use if rebalancing dataset',
        **actions
    )
    parser.add_argument(
        '--seq_filter_keys',
        type=str,
        nargs='+',
        default=None,
        help='If specified, is key(s) to filter by in dataset',
        **actions
    )
    parser.add_argument(
        '--seq_filter_vals',
        type=float,
        nargs='+',
        default=None,
        help='If specified, is value(s) that key(s) should correspond to when filtering dataset',
        **actions
    )

    # Attention args
    parser.add_argument(
        '--use_attention',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use attention or not for observation encoder',
        **actions
    )
    parser.add_argument(
        '--attention_feature_dim',
        type=int,
        default=64,
        help='How many features to use for attention encoding',
        **actions
    )
    parser.add_argument(
        '--attention_layer_dims',
        type=int,
        nargs='+',
        default=[256],
        help='Hidden dimensions for mapping flat obs modalities to featurized vector for attention',
        **actions
    )
    parser.add_argument(
        '--attention_confidence_layer_dims',
        type=int,
        nargs='+',
        default=[128],
        help='Hidden dimensions for mapping attention features to confidence scores',
        **actions
    )
    parser.add_argument(
        '--attention_regularization',
        type=float,
        default=0.0,
        help='Scale of loss for regularizing attention',
        **actions
    )
    parser.add_argument(
        '--rollout_init_states_from_demos',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to initialize rollout states from demos',
        **actions
    )
    parser.add_argument(
        '--rollout_sample_random_demos',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to sample random start states from demos for rollout',
        **actions
    )
    parser.add_argument(
        '--rollout_start_states',
        type=str,
        default=None,
        help='Path to dataset for loading start states for rollouts',
        **actions
    )


def add_batchrl_algo_args():
    """
    Adds batchrl algo-specific args to command line arguments list
    """
    # Define namespace
    prefix = 'batchrl_algo'
    actions = {
        "const": prefix,
        "action": GroupedAction
    }
    parser.add_argument(
        '--rnn_hidden_dim',
        type=int,
        nargs='+',
        default=100,
        maybe_array=True,
        help='hidden dim for rnn policy network. Can specify multiple values if using tiered RNN',
        **actions
    )
    parser.add_argument(
        '--actor_lr',
        type=float,
        default=0.0001,
        help='actor learning rate',
        **actions
    )
    parser.add_argument(
        '--planner_lr',
        type=float,
        default=0.0001,
        help='planner learning rate',
        **actions
    )
    parser.add_argument(
        '--use_gmm',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use GMM distribution for policy',
        **actions
    )
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default=None,
        help='Path to pre-trained model. Used for residual training',
        **actions
    )
    parser.add_argument(
        '--action_limits',
        nargs="+",
        type=float,
        default=None,
        help='Action limits to apply. Should be list of values (e.g., [min, max])',
        **actions
    )
    parser.add_argument(
        '--target_q_gap',
        type=float,
        default=None,
        help='If specified, sets the target q gap. Used for SAC CQL',
        **actions
    )
    parser.add_argument(
        '--use_lagrange',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use automatic tuning for cql. Used for CQL',
        **actions
    )
    parser.add_argument(
        '--obs_modalities',
        type=str,
        nargs='+',
        default=["proprio", "object"],
        help='Modalities to use for observations',
        **actions
    )
    parser.add_argument(
        '--subgoal_modalities',
        type=str,
        nargs='+',
        default=["proprio", "object"],
        help='Modalities to use for subgoal',
        **actions
    )
    parser.add_argument(
        '--kl_weight',
        type=float,
        default=0.0001,
        help='KL Weight for VAE Loss',
        **actions
    )
    parser.add_argument(
        '--primitive',
        type=int,
        default=0,
        help='Primitive ID for specific training, e.g.: Primitive policies',
        **actions
    )
    parser.add_argument(
        '--aux_weight',
        type=float,
        default=0.1,
        help='Auxiliary loss weight if using auxiliary loss',
        **actions
    )
    parser.add_argument(
        '--pretrained_primitives',
        type=str,
        nargs='+',
        default=[],
        maybe_array=True,
        help='Pretrained primitives to use for, e.g., OCP',
        **actions
    )
    parser.add_argument(
        '--steps_per_metastep',
        type=int,
        default=1,
        help='How many primitive policy steps to run per meta step',
        **actions
    )
    parser.add_argument(
        '--rnn_horizon',
        type=int,
        nargs='+',
        default=10,
        maybe_array=True,
        help='Horizon for the RNN policy. Can be multiple if using tiered RNN',
        **actions
    )
    parser.add_argument(
        '--rnn_tiered',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use a tiered RNN or not',
        **actions
    )
    parser.add_argument(
        '--rnn_tier_strides',
        type=int,
        nargs='+',
        default=[1, 5],
        help='Strides to use for each tier. At least 2 should be specified, and only relevant for tiered RNNs',
        **actions
    )
    parser.add_argument(
        '--rnn_hidden_tier_output_dim',
        type=int,
        nargs='+',
        default=32,
        maybe_array=True,
        help='Output dimensions for each hidden tier; only relevant for tiered RNNs. If specified for per-layer, should'
             'have n_tiers - 1 numbers specified',
        **actions
    )

    # Args for FAN
    parser.add_argument(
        '--policy0_obs_modalities',
        type=str,
        nargs='+',
        default=["proprio", "object"],
        help='Modalities to use for policy0 obs',
        **actions
    )
    parser.add_argument(
        '--policy0_subgoal_modalities',
        type=str,
        nargs='+',
        default=["proprio", "object"],
        help='Modalities to use for policy0 subgoals',
        **actions
    )
    parser.add_argument(
        '--policy1_obs_modalities',
        type=str,
        nargs='+',
        default=["proprio", "object"],
        help='Modalities to use for policy1 obs',
        **actions
    )
    parser.add_argument(
        '--policy1_subgoal_modalities',
        type=str,
        nargs='+',
        default=["proprio", "object"],
        help='Modalities to use for policy1 subgoals',
        **actions
    )
    parser.add_argument(
        '--use_gmm_planner_prior',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use GMM distribution for planner prior (in GTI)',
        **actions
    )
    parser.add_argument(
        '--policy_ac_dims',
        type=int,
        nargs='+',
        default=[2, 8],
        help='action dimensions for each subpolicy',
        **actions
    )
    parser.add_argument(
        '--autoregressive_order',
        type=int,
        nargs='+',
        default=None,
        help='If specified, should be ordering of policies by their index for autoregressive conditioning',
        **actions
    )
    parser.add_argument(
        '--grayscale_reconstruction',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to use grayscale for reconstruction',
        **actions
    )


def add_batchrl_hp_args():
    """
    Adds batchrl hyperparameter-sweep specific args to command line arguments list
    """
    # Define namespace
    prefix = 'batchrl_hp'
    actions = {
        "const": prefix,
        "action": GroupedAction
    }

    parser.add_argument(
        '--hp_sweep_script',
        type=str,
        required=True,
        help='Path to generated hyperparamter script from hyperparameter_helper.py (in batchRL)',
        **actions
    )

    parser.add_argument(
        '--generate_debug_script',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to generate a debugging script for all of the generated configs or not',
        **actions
    )

    parser.add_argument(
        '--n_exps_per_instance',
        type=int,
        default=1,
        help='Number of experiments to run per sbatch instance',
        **actions
    )


# def add_batchrl_gti_args():
#     """
#     Adds batchrl gti-specific args to command line arguments list
#     """
#     # Define namespace
#     prefix = 'batchrl_gti'
#     actions = {
#         "const": prefix,
#         "action": GroupedAction
#     }
#
#
# def add_batchrl_bc_rnn_args():
#     """
#     Adds batchrl bc_rnn-specific args to command line arguments list
#     """
#     # Define namespace
#     prefix = 'batchrl_bc_rnn'
#     actions = {
#         "const": prefix,
#         "action": GroupedAction
#     }

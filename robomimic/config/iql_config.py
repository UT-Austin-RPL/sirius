"""
Config for CQL algorithm.
"""

from robomimic.config.base_config import BaseConfig


class IQLConfig(BaseConfig):
    ALGO_NAME = "iql"

    def train_config(self):
        """
        Update from superclass to change default batch size.
        """
        super(IQLConfig, self).train_config()

        # increase batch size to 1024 (found to work better for most manipulation experiments)
        self.train.batch_size = 1024

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(IQLConfig, self).algo_config()

        # optimization parameters
        self.algo.optim_params.critic.learning_rate.initial = 1e-4          # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.0      # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []     # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00              # L2 regularization strength

        self.algo.optim_params.vf.learning_rate.initial = 1e-4           # actor learning rate
        self.algo.optim_params.vf.learning_rate.decay_factor = 0.0       # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.vf.learning_rate.epoch_schedule = []      # epochs where LR decay occurs
        self.algo.optim_params.vf.regularization.L2 = 0.00               # L2 regularization strength

        self.algo.optim_params.actor.learning_rate.initial = 1e-4           # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = 0.0       # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.actor.learning_rate.epoch_schedule = []      # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = 0.00               # L2 regularization strength

        # target network related parameters
        self.algo.discount = 0.99                                           # discount factor to use
        self.algo.target_tau = 0.01                                        # update rate for target networks
        self.algo.ignore_dones = False
        self.algo.use_negative_rewards = False
        self.algo.use_shaped_rewards = False
        self.algo.relabel_dones_mode = None
        self.algo.relabel_rewards_mode = None

        # Actor network settings
        self.algo.actor.net.type = "gaussian"                               # Options are currently only "gaussian" (no support for GMM yet)

        # Actor network settings - shared
        self.algo.actor.net.common.std_activation = "softplus"                   # Activation to use for std output from policy net
        self.algo.actor.net.common.low_noise_eval = True                    # Whether to use deterministic action sampling at eval stage
        self.algo.actor.net.common.use_tanh = False

        # Actor network settings - gaussian
        self.algo.actor.net.gaussian.init_last_fc_weight = 0.001            # If set, will override the initialization of the final fc layer to be uniformly sampled limited by this value
        self.algo.actor.net.gaussian.init_std = 0.3                         # Relative scaling factor for std from policy net
        self.algo.actor.net.gaussian.fixed_std = False                      # Whether to learn std dev or not

        self.algo.actor.net.gmm.num_modes = 5
        self.algo.actor.net.gmm.min_std = 0.0001

        self.algo.actor.layer_dims = (300, 400)                             # actor MLP layer dimensions

        self.algo.actor.max_gradient_norm = None

        # actor residual MLP settings
        self.algo.actor.res_mlp.enabled = False
        self.algo.actor.res_mlp.num_blocks = 4
        self.algo.actor.res_mlp.hidden_dim = 1024
        self.algo.actor.res_mlp.use_layer_norm = True

        # ================== Critic Network Config ===================
        # critic ensemble parameters (TD3 trick)
        self.algo.critic.ensemble.n = 2                                     # number of Q networks in the ensemble
        self.algo.critic.layer_dims = (300, 400)                            # critic MLP layer dimensions
        self.algo.critic.use_huber = False

        # critic residual MLP settings
        self.algo.critic.res_mlp.enabled = False
        self.algo.critic.res_mlp.num_blocks = 4
        self.algo.critic.res_mlp.hidden_dim = 1024
        self.algo.critic.res_mlp.use_layer_norm = True

        self.algo.adv.filter_type = "softmax"
        self.algo.adv.use_final_clip = True
        self.algo.adv.clip_adv_value = None
        self.algo.adv.beta = 1.0

        self.algo.vf_quantile = 0.9

        self.algo.critic.max_gradient_norm = None

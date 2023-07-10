"""
Contains distribution models used as parts of other networks. These
classes usually inherit or emulate torch distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
class TanhWrappedDistribution(D.Distribution):
    """
    Class that wraps another valid torch distribution, such that sampled values from the base distribution are
    passed through a tanh layer. The corresponding (log) probabilities are also modified accordingly.
    Tanh Normal distribution - adapted from rlkit and CQL codebase
    (https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6).
    """
    def __init__(self, base_dist, scale=1.0, epsilon=1e-6):
        """
        Args:
            base_dist (Distribution): Distribution to wrap with tanh output
            scale (float): Scale of output
            epsilon (float): Numerical stability epsilon when computing log-prob.
        """
        self.base_dist = base_dist
        self.scale = scale
        self.tanh_epsilon = epsilon
        super(TanhWrappedDistribution, self).__init__()
    def log_prob(self, value, pre_tanh_value=None):
        """
        Args:
            value (torch.Tensor): some tensor to compute log probabilities for
            pre_tanh_value: If specified, will not calculate atanh manually from @value. More numerically stable
        """
        value = value / self.scale
        if pre_tanh_value is None:
            one_plus_x = (1. + value).clamp(min=self.tanh_epsilon)
            one_minus_x = (1. - value).clamp(min=self.tanh_epsilon)
            pre_tanh_value = 0.5 * torch.log(one_plus_x / one_minus_x)
        lp = self.base_dist.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value * value + self.tanh_epsilon)
        # In case the base dist already sums up the log probs, make sure we do the same
        return lp - tanh_lp if len(lp.shape) == len(tanh_lp.shape) else lp - tanh_lp.sum(-1)
    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.base_dist.sample(sample_shape=sample_shape).detach()
        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale
    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Sampling in the reparameterization case - for differentiable samples.
        """
        z = self.base_dist.rsample(sample_shape=sample_shape)
        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale
    @property
    def mean(self):
        return self.base_dist.mean
    @property
    def stddev(self):
        return self.base_dist.stddev
class DiscreteValueDistribution(object):
    """
    Extension to torch categorical probability distribution in order to keep track
    of the support (categorical values, or in this case, value atoms). This is
    used for distributional value networks.
    """
    def __init__(self, values, probs=None, logits=None):
        """
        Creates a categorical distribution parameterized by either @probs or
        @logits (but not both). Expects inputs to be consistent in shape
        for broadcasting operations (e.g. multiplication).
        """
        self._values = values
        self._categorical_dist = D.Categorical(probs=probs, logits=logits)
    @property
    def values(self):
        return self._values
    @property
    def probs(self):
        return self._categorical_dist.probs
    @property
    def logits(self):
        return self._categorical_dist.logits
    def mean(self):
        """
        Categorical distribution mean, taking the value support into account.
        """
        return (self._categorical_dist.probs * self._values).sum(dim=-1)
    def variance(self):
        """
        Categorical distribution variance, taking the value support into account.
        """
        dist_squared = (self.mean().unsqueeze(-1) - self.values).pow(2)
        return (self._categorical_dist.probs * dist_squared).sum(dim=-1)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the distribution. Make sure to return value atoms, not categorical class indices.
        """
        inds = self._categorical_dist.sample(sample_shape=sample_shape)
        return torch.gather(self.values, inds, dim=-1)
class MixtureSameFamily(D.Distribution):
    """
    NOTE: this is copied from pytorch 1.7 source code
    """
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.
    Examples::
        # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)
        # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
        # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
                     torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)
        # Construct a batch of 3 Gaussian Mixture Models in 2D each
        # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)
    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints = {}
    has_rsample = False
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        if not isinstance(self._mixture_distribution, D.Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")
        if not isinstance(self._component_distribution, D.Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")
        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                 "compatible with `component_distribution."
                                 "batch_shape`({1})".format(mdbs, cdbs))
        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km
        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)
    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = \
            self._component_distribution.expand(batch_shape_comp)
        new._mixture_distribution = \
            self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(batch_shape=batch_shape,
                                               event_shape=event_shape,
                                               validate_args=False)
        new._validate_args = self._validate_args
        return new
    # @constraints.dependent_property
    # def support(self):
    #     # FIXME this may have the wrong shape when support contains batched
    #     # parameters
    #     return self._component_distribution.support
    @property
    def mixture_distribution(self):
        return self._mixture_distribution
    @property
    def component_distribution(self):
        return self._component_distribution
    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]
    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs * self.component_distribution.variance,
                                  dim=-1 - self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1 - self._event_ndims)
        return mean_cond_var + var_cond_mean
    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs
        return torch.sum(cdf_x * mix_prob, dim=-1)
    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape
            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape
            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)
            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)
            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)
    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)
    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x
    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.mixture_distribution,
                                             self.component_distribution)
        return 'MixtureSameFamily' + '(' + args_string + ')'

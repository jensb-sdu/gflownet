from typing import Iterable, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from torchtyping import TensorType
import math
from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong, tint, tbool

def PAD_FUNC(**kwargs):

    return torch.nan 

# Example of custom functions
def Yin(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the first element of the time series"""
    return input[0]

def Yout(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the last element of the time series"""
    return input[-1]

# Statistical Measures
def median(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the median value"""
    return torch.median(input)

def std(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the standard deviation"""
    return torch.std(input)

def variance(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the variance"""
    return torch.var(input)

def mad(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the Mean Absolute Deviation"""
    return torch.mean(torch.abs(input - torch.mean(input)))

def rms(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the Root Mean Square"""
    return torch.sqrt(torch.mean(input ** 2))

def skewness(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the skewness (third moment)"""
    mean = torch.mean(input)
    std = torch.std(input)
    return torch.mean(((input - mean) / std) ** 3)

def kurtosis(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the kurtosis (fourth moment)"""
    mean = torch.mean(input)
    std = torch.std(input)
    return torch.mean(((input - mean) / std) ** 4)

# Range and Spread Measures
def value_range(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the range (max - min)"""
    return torch.max(input) - torch.min(input)

def iqr(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the Interquartile Range (Q3 - Q1) - vmap compatible"""
    sorted_input = torch.sort(input)[0]
    n = len(sorted_input)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    return sorted_input[q3_idx] - sorted_input[q1_idx]

def percentile_90(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the 90th percentile - vmap compatible"""
    sorted_input = torch.sort(input)[0]
    idx = int(len(sorted_input) * 0.90)
    return sorted_input[min(idx, len(sorted_input) - 1)]

def percentile_10(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the 10th percentile - vmap compatible"""
    sorted_input = torch.sort(input)[0]
    idx = int(len(sorted_input) * 0.10)
    return sorted_input[idx]

def median(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the median value - vmap compatible"""
    sorted_input = torch.sort(input)[0]
    n = len(sorted_input)
    if n % 2 == 1:
        return sorted_input[n // 2]
    else:
        return (sorted_input[n // 2 - 1] + sorted_input[n // 2]) / 2

# More efficient versions using kthvalue (which is vmap compatible)
def iqr_fast(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the Interquartile Range (Q3 - Q1) - faster vmap compatible version"""
    n = len(input)
    q1_idx = max(1, int(n * 0.25))
    q3_idx = max(1, int(n * 0.75))
    q1 = torch.kthvalue(input, q1_idx)[0]
    q3 = torch.kthvalue(input, q3_idx)[0]
    return q3 - q1

def percentile_90_fast(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the 90th percentile - faster vmap compatible version"""
    n = len(input)
    idx = max(1, min(n, int(n * 0.90)))
    return torch.kthvalue(input, idx)[0]

def percentile_10_fast(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the 10th percentile - faster vmap compatible version"""
    n = len(input)
    idx = max(1, int(n * 0.10))
    return torch.kthvalue(input, idx)[0]

def median_fast(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the median value - faster vmap compatible version"""
    n = len(input)
    k = (n + 1) // 2
    return torch.kthvalue(input, k)[0]

# Trend and Change Measures
def slope(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the linear trend slope"""
    n = len(input)
    x = torch.arange(n, dtype=input.dtype, device=input.device)
    x_mean = x.mean()
    y_mean = input.mean()
    numerator = torch.sum((x - x_mean) * (input - y_mean))
    denominator = torch.sum((x - x_mean) ** 2)
    return numerator / (denominator + 1e-10)

def absolute_change(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the absolute change (last - first)"""
    return input[-1] - input[0]

def relative_change(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the relative change ((last - first) / first)"""
    return (input[-1] - input[0]) / (torch.abs(input[0]) + 1e-10)

def mean_absolute_change(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the mean absolute difference between consecutive values"""
    return torch.mean(torch.abs(torch.diff(input)))

def mean_change(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the mean difference between consecutive values"""
    return torch.mean(torch.diff(input))

# Energy and Power Measures
def energy(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the sum of squared values"""
    return torch.sum(input ** 2)

def absolute_energy(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the sum of absolute values"""
    return torch.sum(torch.abs(input))

def mean_abs_value(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the mean of absolute values"""
    return torch.mean(torch.abs(input))

# Zero Crossing and Peak Measures
def zero_crossing_rate(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the rate of sign changes"""
    signs = torch.sign(input)
    sign_changes = torch.sum(torch.abs(torch.diff(signs))) / 2
    return sign_changes / (len(input) - 1)

def mean_crossing_rate(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the rate of mean crossing"""
    mean_val = torch.mean(input)
    crossings = torch.sum(torch.abs(torch.diff(torch.sign(input - mean_val)))) / 2
    return crossings / (len(input) - 1)

def peaks_count(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the number of local maxima"""
    if len(input) < 3:
        return torch.tensor(0.0)
    peaks = ((input[1:-1] > input[:-2]) & (input[1:-1] > input[2:])).sum()
    return peaks.float()

# Complexity Measures
def abs_sum_of_changes(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the absolute sum of consecutive changes"""
    return torch.sum(torch.abs(torch.diff(input)))

def longest_strike_above_mean(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the length of the longest consecutive sequence above mean"""
    mean_val = torch.mean(input)
    mask = (input > mean_val).float()
    
    # Use cumulative sum with reset logic
    # Create boundaries: 1 where streak starts, 0 elsewhere
    boundaries = torch.cat([mask[:1], mask[1:] * (1 - mask[:-1])])
    # Cumulative sum of boundaries gives group ID
    groups = torch.cumsum(boundaries, dim=0)
    # Cumulative count within each group
    counts = torch.cumsum(mask, dim=0) - torch.maximum(
        torch.zeros_like(groups),
        torch.cumsum(boundaries * (torch.cumsum(mask, dim=0) - mask), dim=0)
    )
    
    return torch.max(counts * mask)

def longest_strike_below_mean(input: TensorType, *, dtype: torch.dtype | None = None):
    """Returns the length of the longest consecutive sequence below mean"""
    mean_val = torch.mean(input)
    mask = (input < mean_val).float()
    
    boundaries = torch.cat([mask[:1], mask[1:] * (1 - mask[:-1])])
    groups = torch.cumsum(boundaries, dim=0)
    counts = torch.cumsum(mask, dim=0) - torch.maximum(
        torch.zeros_like(groups),
        torch.cumsum(boundaries * (torch.cumsum(mask, dim=0) - mask), dim=0)
    )
    
    return torch.max(counts * mask)
 


# Updated FUNCTIONS list with fast versions
FUNCTIONS = tuple([
    # Basic statistics
    torch.mean,
    torch.max,
    torch.min,
    median_fast,  # Use fast version
    torch.std,
    torch.var,
    
    # Positional
    Yin,
    Yout,
    torch.argmin,
    torch.argmax,
    
    # Distribution measures
    mad,
    rms,
    skewness,
    kurtosis,
    
    # Range measures (use fast versions)
    value_range,
    iqr_fast,
    percentile_90_fast,
    percentile_10_fast,
    
    # Integration
    torch.trapezoid,
    
    # Trend measures
    slope,
    absolute_change,
    relative_change,
    mean_change,
    mean_absolute_change,
    
    # Energy measures
    energy,
    absolute_energy,
    mean_abs_value,
    
    # Crossing and peaks
    zero_crossing_rate,
    mean_crossing_rate,
    peaks_count,
    
    # Complexity
    abs_sum_of_changes,
    longest_strike_above_mean,
    longest_strike_below_mean,
])
 


class VerificationEnv(GFlowNetEnv) :
    # shared cache for constructed action spaces across all instances
    _action_space_cache = {}

    def __init__(self,
                 functions: Iterable = None,
                 data_path: Union[str, Path] = None,
                 max_length: int = 16,
                 window_size: int = 2048,
                 min_function_width = 64,
                 internal_dtype: torch.dtype = torch.int16,
                 **kwargs):
        
        # save the path to the data set on which to evalute on
        if isinstance(data_path,Path):
            self.data_path = data_path
        else:
            self.data_path = Path(data_path)

        self.int = internal_dtype

        # Use default list of functions unless custom list is provided
        if functions is None :
            self.functions = FUNCTIONS
        else :
            self.functions = functions

        self.n_functions = len(self.functions)
        self.max_length = max_length
        self.eos_idx = -1
        self.pad_idx = 0
        self.eos = [self.eos_idx, 0, 0]
        self.padding = [self.pad_idx, 0, 0]
        self.source = torch.tensor([[self.pad_idx,0 ,0]]  * self.max_length, dtype=self.int, device=kwargs.get('device', None) )
        self.padding = torch.tensor(self.padding, dtype=self.int, device=kwargs.get('device', None) )
        self.eos = torch.tensor(self.eos, dtype=self.int, device=kwargs.get('device', None) )
        self.funcidx2token = { idx + 1 : func for idx, func in enumerate(self.functions) }
        
        self.window_size = window_size
        self.min_function_width = min_function_width
        self.action_space = None



        super().__init__(**kwargs)
        # self.source = self.source.to(self.device if self.device else 'cpu' )
        # self.padding = self.padding.to(self.device if self.device else 'cpu' )
        # self.eos = self.eos.to(self.device if self.device else 'cpu' )
        # self.action_space = self.get_action_space().to(self.device if self.device else 'cpu' )  


    def get_action_space(self) -> TensorType["action_space_dim", "action_dim"]:
        """
        Returns the action space of the environment.
        The action space is cached per-class to avoid rebuilding it repeatedly.

        Optimized implementation: vectorized construction using torch tensors and a boolean
        mask to extract valid (start,end) pairs for one function, then replicate for all
        functions. Works on CPU and stores a cached CPU tensor.
        """
        if self.action_space is not None:
            return self.action_space.clone()

        key = (self.n_functions, self.window_size, self.min_function_width)

        if key in VerificationEnv._action_space_cache:
            cached = VerificationEnv._action_space_cache[key]
            self.action_space = cached.to(self.device if self.device else "cpu")
            return self.action_space.clone()

        W = int(self.window_size)
        m = int(self.min_function_width)
        u = W - m  # number of possible start positions
        if u <= 0:
            # no valid (start,end) pairs, only EOS
            tensor_cpu = torch.tensor([[self.eos_idx, 0, 0]], dtype=torch.int16, device="cpu")
            VerificationEnv._action_space_cache[key] = tensor_cpu
            self.action_space = tensor_cpu.to(self.device if self.device else "cpu")
            return self.action_space.clone()

        # Build grid for a single function on CPU
        # start indices: 0 .. u-1 (shape u x u after broadcast)
        starts_grid = torch.arange(u, dtype=torch.int32, device="cpu").view(u, 1).expand(u, u)
        # relative offset for end: 0 .. u-1 (same shape u x u)
        rel_offsets = torch.arange(u, dtype=torch.int32, device="cpu").view(1, u).expand(u, u)
        # ends = start + min_width + rel_offsets
        ends_grid = starts_grid + m + rel_offsets
        # mask valid pairs (end must be < W)
        mask = ends_grid < W
        # extract flattened valid starts/ends for one function
        valid_starts = starts_grid[mask].to(torch.int16)
        valid_ends = ends_grid[mask].to(torch.int16)
        pairs_per_function = valid_starts.numel()

        try:
            # Check that the last and first valid pairs are correct
            assert valid_starts[0].item() == 0
            assert valid_ends[-1].item() == W - 1
        except AssertionError:
            raise ValueError(f"Error in action space construction: invalid start/end pairs {valid_starts, valid_ends}")


        # replicate for all functions
        if self.n_functions == 1:
            func_col = torch.ones(pairs_per_function, dtype=torch.int16, device="cpu")
        else:
            func_col = torch.arange(1, self.n_functions + 1, dtype=torch.int16, device="cpu").repeat_interleave(pairs_per_function)

        starts_all = valid_starts.repeat(self.n_functions)
        ends_all = valid_ends.repeat(self.n_functions)

        # stack columns [func, start, end]
        actions = torch.stack([func_col, starts_all, ends_all], dim=1).to(torch.int16)

        # append EOS action (keep dtype int16; eos_idx is negative and fits)
        eos_row = torch.tensor([[self.eos_idx, 0, 0]], dtype=torch.int16, device="cpu")
        tensor_cpu = torch.cat([actions, eos_row], dim=0).contiguous()

        # cache CPU tensor and move a copy to instance device
        VerificationEnv._action_space_cache[key] = tensor_cpu
        self.action_space = tensor_cpu.to(self.device if self.device else "cpu")
        return self.action_space


    def get_mask_invalid_actions_forward(
        self,
        state: Optional[TensorType["state_dim", "action_dim"]] = None,
        done: Optional[bool] = None,
    ) -> TensorType["action_space_dim"]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return torch.ones(self.action_space_dim, dtype=torch.bool, device=self.device)
        # If sequence is not at maximum length, all actions are valid
        seq_len = self._get_seq_length(state)
        if seq_len < self.max_length:
            mask = torch.zeros(self.action_space_dim, dtype=torch.bool, device=self.device)
            #filter out existing actions in the current state
            mask[self.actions2indices(state[:seq_len])] = True          
            return mask

        # Otherwise, only EOS is valid
        mask = torch.ones(self.action_space_dim, dtype=torch.bool, device=self.device)
        mask[self.eos_idx] = False
        return mask

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> TensorType["state_space_dim", "action_dim"]:
        """
        Returns the list of non-invalid (valid, for short) according to the mask of
        invalid actions.

        More documentation about the meaning and use of invalid actions can be found in
        gflownet/envs/README.md.
        """
        if mask is None:
            mask = self.get_mask(state, done, backward)
        return self.action_space[~mask].to(self.device).clone()

    def get_mask_invalid_actions_backward(self, state = None, done = None, parents_a = None):
        """
        Returns a list of length the action space with values:
            - True if the backward action is invalid from the current state.
            - False otherwise.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        mask = torch.ones(self.action_space_dim, dtype=torch.bool, device=self.device)
        if done:
            return mask
        # If sequence is not at maximum length, all actions are valid
        if parents_a is None:
            _, parents_a = self.get_parents(state, done)

        for pa in parents_a:
            # Compare row-wise to obtain a 1D boolean mask matching mask.shape
            # Ensure pa is a tensor with same dtype/device as action_space
            if not torch.is_tensor(pa):
                pa = torch.tensor(pa, dtype=self.action_space.dtype, device=self.action_space.device)
            eq = (self.action_space == pa).all(dim=1)
            mask[eq.to(self.device)] = False
        return mask

    def get_parents(
        self,
        state: Optional[TensorType["state_dim", "action_dim"]] = None,
        done: Optional[bool] = None,
        action: Optional[TensorType["action_dim"]] = None,
    ) -> Tuple[TensorType["state_dim", "action_dim"], TensorType["action_dim"]]:
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : TensorType["state_dim", "action_dim"]
            Tensor of parents in state format. This environment has a single parent per
            state.

        actions : TensorType["action_dim"]
            Tensor of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []
        last_func_idx = self._get_seq_length(state) - 1
        parent = copy(state)
        parent[last_func_idx] = self.padding
        p_action = state[last_func_idx]
        return [parent], [p_action]
    

    def _pre_step(
        self, action: Tuple[int], backward: bool = False, skip_mask_check: bool = False
    ) -> Tuple[bool, List[int], Tuple[int]]:
        """
        Performs generic checks shared by the step() and step_backward() (backward must
        be True) methods of all environments.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        do_step : bool
            If True, step() should continue further, False otherwise.

        self.state : list
            The sequence after executing the action

        action : int
            Action index
        """
        # If action not found in action space raise an error
        if not (action == self.action_space).all(dim=1).any():
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        # If backward and state is source, step should not proceed.
        if backward is True:
            if self.equal(self.state, self.source) and action != self.eos:
                return False, self.state, action
        # If forward and env is done, step should not proceed.
        else:
            if self.done:
                return False, self.state, action
        # If action is in invalid mask (not in valid actions), step should not proceed.
        if not (self.skip_mask_check or skip_mask_check):
            if not (action == self.get_valid_actions(backward=backward)).all(dim=1).any():
                return False, self.state, action
        return True, self.state, action


    def step(
            self, action: TensorType["action_dim"], skip_mask_check: bool = False
        ) -> Tuple[TensorType["state_dim", "action_dim"], TensorType["action_dim"], bool]:
            """
            Executes step given an action.

            Args
            ----
            action : TensorType["action_dim"]
                Action to be executed. An action is a tuple int values indicating the
                dimensions to increment by 1.

            skip_mask_check : bool
                If True, skip computing forward mask of invalid actions to check if the
                action is valid.

            Returns
            -------
            self.state : TensorType["state_dim", "action_dim"]
                The sequence after executing the action

            action : TensorType["action_dim"]
                Action executed

            valid : bool
                False, if the action is not allowed for the current state.
            """
            # Generic pre-step checks
            do_step, self.state, action = self._pre_step(
                action, skip_mask_check or self.skip_mask_check
            )
            if not do_step:
                return self.state, action, False
            valid = True
            self.n_actions += 1
            # If action is EOS, set done to True and return state as is
            if torch.equal(action, self.eos):
                self.done = True
                return self.state, action, valid
            # Update state
            self.state[self._get_seq_length()] = action
            return self.state, action, valid
    
    
    # def randomize_and_temper_sampling_distribution(
    #     self,
    #     policy_outputs: TensorType["n_states", "policy_output_dim"],
    #     probability_random_action: Optional[float] = 0.0,
    #     temperature: Optional[float] = 1.0,
    # ) -> TensorType["n_states", "policy_output_dim"]:
    #     """
    #     Replaces the rows of `policy_outputs` by a vector corresponding to a random
    #     sampling policy with the probability indicated by `probability_random_action`.

    #     Note that the tensor of policy outputs is not cloned if neither tempering nor
    #     random actions are incorporated. This implies that the original tensor of
    #     policy outputs may be modified by subsequent methods (namely
    #     sample_actions_batch()), for example to mask the invalid actions.

    #     Parameters
    #     ----------
    #     policy_outputs : tensor
    #         The original outputs of the sampling policy. For example, they may
    #         correspond to the output (logits) of the GFlowNet policy model.
    #     probability_random_action : float, optional
    #         The probability of sampling a random action. If larger than one, the logits
    #         will be replaced by a random policy vector with this probability, according
    #         to Bernoulli distribution. By default, the probability is 0.0 (no random
    #         actions).
    #     temperature : float, optional
    #         A scalar by which the logits are divided to adjust the sampling
    #         distribution. A temperature larger than one will result in a flatter
    #         distribution, favouring exploration. A temperature smaller than one will
    #         sharpen the distribution, favouring concentration around high probability
    #         actions. By default, the temperature is 1.0 (no tempering).

    #     Returns
    #     -------
    #     policy_outputs : tensor
    #         The modified policy outputs.
    #     """
    #     if not math.isclose(temperature, 1.0, abs_tol=1e-08):
    #         do_temper = True
    #     else:
    #         do_temper = False
    #     if not math.isclose(probability_random_action, 0.0, abs_tol=1e-08):
    #         do_random = True
    #     else:
    #         do_random = False
    #     if not do_temper and not do_random:
    #         return policy_outputs

    #     # Clone the sampling logits in order not to change the original tensor
    #     logits_sampling = policy_outputs.clone().detach().to('cpu')
    #     if do_temper:
    #         logits_sampling /= temperature
    #     if do_random:
    #         idx_random = tbool(
    #             Bernoulli(
    #                 probability_random_action * torch.ones(policy_outputs.shape[0])
    #             ).sample(),
    #             device='cpu',
    #         )
    #         logits_sampling[idx_random, :] = self.random_policy_output
    #     return policy_outputs

    
    # def sample_actions_batch(
    #     self,
    #     policy_outputs: TensorType["n_states", "policy_output_dim"],
    #     mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
    #     states_from: Optional[List] = None,
    #     is_backward: Optional[bool] = False,
    #     random_action_prob: Optional[float] = 0.0,
    #     temperature_logits: Optional[float] = 1.0,
    # ) -> Tuple[List[Tuple], TensorType["n_states"]]:
    #     """
    #     Samples a batch of actions from a batch of policy outputs.

    #     This implementation is generally valid for all discrete environments but
    #     continuous or mixed environments need to reimplement this method.

    #     The method is valid for both forward and backward actions in the case of
    #     discrete environments. Some continuous environments may also be agnostic to the
    #     difference between forward and backward actions since the necessary information
    #     can be contained in the mask. However, some continuous environments do need to
    #     know whether the actions are forward of backward, which is why this can be
    #     specified by the argument is_backward.

    #     Most environments do not need to know the states from which the actions are to
    #     be sampled since the necessary information is in both the policy outputs and
    #     the mask. However, some continuous environments do need to know the originating
    #     states in order to construct the actions, which is why one of the arguments is
    #     states_from.

    #     Note that methods overriding this method should randomize and temper the
    #     logits.

    #     Parameters
    #     ----------
    #     policy_outputs : tensor
    #         The output of the GFlowNet policy model.
    #     mask : tensor
    #         The mask of invalid actions. For continuous or mixed environments, the mask
    #         may be tensor with an arbitrary length contaning information about special
    #         states, as defined elsewhere in the environment.
    #     states_from : tensor
    #         The states originating the actions, in GFlowNet format. Ignored in discrete
    #         environments and only required in certain continuous environments.
    #     is_backward : bool
    #         True if the actions are backward, False if the actions are forward
    #         (default). Ignored in discrete environments and only required in certain
    #         continuous environments.
    #     random_action_prob : float, optional
    #         The probability of sampling a random action. If larger than one, the model
    #         outputs will be replaced by a random policy vector with probability
    #         `random_action_prob`, according to Bernoulli distribution.
    #     temperature_logits : float, optional
    #         A scalar by which the model outputs are divided to temper the sampling
    #         distribution.

    #     Returns
    #     -------
    #     actions : list
    #         The list of sampled actions.
    #     """
    #     # Randomize actions and temper the logits
    #     logits_sampling = self.randomize_and_temper_sampling_distribution(
    #         policy_outputs, random_action_prob, temperature_logits
    #     )

    #     # Make the logits of invalid actions equal to -inf.
    #     if mask is not None:
    #         if torch.all(mask, dim=1).any():
    #             raise RuntimeError(
    #                 "All actions in the mask are invalid for some states in the batch."
    #             )
    #         logits_sampling[mask] = -torch.inf

    #     # Sample actions from the Categorical distributions defined by the logits
    #     action_indices = Categorical(logits=logits_sampling).sample().to(self.device)
    #     # Build actions
    #     actions = self.action_space[action_indices]
    #     return actions
    

    def state2readable(self, state: TensorType["state_dim", "action_dim"] = None) -> str:
        """
        Converts a state into a human-readable string.

        The output string contains the letter corresponding to each index in the state,
        separated by spaces.

        Args
        ----
        states : tensor
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated letters.
        """
        state = self._get_state(state)
        state = self._unpad(state)

        # Normalize into a flat list of rows
        if torch.is_tensor(state):
            rows = state.tolist()
        else:
            rows = state

        entries = []
        for r in rows:
            # Expect each row to be [func_idx, start, end]
            try:
                func_idx = int(r[0])
                start = int(r[1])
                end = int(r[2])
            except Exception:
                # Fallback: skip malformed rows
                continue

            # skip padding
            if func_idx == self.pad_idx:
                continue

            # handle EOS explicitly
            if func_idx == self.eos_idx:
                entries.append("EOS")
                continue

            token = self.funcidx2token.get(func_idx, None)
            if callable(token) and hasattr(token, "__name__"):
                name = token.__name__
            else:
                name = str(token)

            entries.append(f"{name}[{start},{end}]")

        return " ".join(entries)
    

    def action2index(self, action):
        """
        Converts an action into its index in the action space.

        Args
        ----
        action : TensorType["action_dim"]
            The action to convert.

        Returns
        -------
        The index of the action in the action space.
        """
        return (self.action_space == action).all(dim=1).nonzero(as_tuple=True)[0].item()

    def states2policy(
        self,
        states: Union[
            List[TensorType["height", "width"]], TensorType["height", "width", "batch"]
        ],
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "environment format" for the policy model.

        See states2proxy().

        Args
        ----
        states : list of 2D tensors or 3D tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tint(states, device=self.device, int_type=self.int)
        return self.states2proxy(states).flatten(start_dim=1).to(self.float)


    def readable2state(self, readable):
        import re
        matches = re.findall(r'([A-Za-z0-9_+-]+)\[(\d+),\s*(\d+)\]', readable)
        state_list = []
        for name, a, b in matches:
            a = int(a)
            b = int(b)
            func_idx = None
            for i, f in enumerate(self.functions, start=1):
                if callable(f) and hasattr(f, "__name__"):
                    fname = f.__name__
                else:
                    fname = str(f)
                if fname == name:
                    func_idx = i
                    break
            if func_idx is not None:
                state_list.append([func_idx, a, b])
        
        state_tensor = torch.tensor([state_list], dtype=torch.int16, device=self.device)

        return state_tensor


    def _pad(self, state: TensorType["state_dim", "action_dim"]):
        """
        Pads a sequence represented as a list of indices.

        Args
        ----
        seq_list : list
            The input sequence. A list containing a list of indices.

        Returns
        -------
        The input list padded by the end with self.pad_idx.
        """


        seq_len = self._get_seq_length(state)
        n_pads = self.max_length - seq_len
        if n_pads <= 0:
            return state
        pad_tensor = self.padding.repeat(n_pads, 1)
        return torch.cat([state, pad_tensor], dim=0)

    def _unpad(self, state: TensorType["state_dim", "action_dim"]):
        """
        Removes the padding from the end off a sequence represented as a list of
        indices.

        Args
        ----
        seq_list : list
            The input sequence. A list containing a list of indices, including possibly
            padding indices.

        Returns
        -------
        The input list padded by the end with self.pad_idx.
        """
        seq_len = self._get_seq_length(state)
        return state[:seq_len]

    def _get_seq_length(self, state: TensorType["state_dim", "action_dim"] = None):
        """
        Returns the effective length of a state, that is ignoring the padding.

        Args
        ----
        state : list
            The input sequence. If None, self.state is used.

        Returns
        -------
        Length of the sequence, without counting the padding.
        """
        state = self._get_state(state)
        # If state equals the source (all padding), return length 0
        if self.equal(state, self.source):
            return 0

        # Extract the function-index column (handle 1D or 2D)
        if state.dim() > 1:
            func_indices = state[:, 0]
        else:
            func_indices = state

        # Find the first padding occurrence; if none, sequence is full length
        pads = (func_indices == self.pad_idx).nonzero(as_tuple=True)[0]
        if pads.numel() == 0:
            return self.max_length
        return int(pads[0].item())
from typing import Iterable, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

def PAD_FUNC(**kwargs):

    return torch.nan 

# list of ucntion to chose from
FUNCTIONS = tuple(
    [torch.mean,
     torch.max,
     torch.min,
     torch.std,
     torch.var,
     torch.trapezoid,
     torch.argmin,
     torch.argmax,

     ]
)




class VerificationEnv(GFlowNetEnv) :
    def __init__(self,
                 functions: Iterable = None,
                 data_path: Union[str, Path] = None,
                 max_length: int = 16,
                 window_size: int = 2048,
                 min_function_width = 64,
                 **kwargs):
        
        # save the path to the data set on which to evalute on
        if isinstance(data_path,Path):
            self.data_path = data_path
        else:
            self.data_path = Path(data_path)

        

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
        self.source = torch.tensor([[self.pad_idx,0 ,0]  * self.max_length])
        self.funcidx2token = { idx + 1 : func for idx, func in enumerate(self.functions) }
        
        self.window_size = window_size
        self.min_function_width = min_function_width
        self.action_space = None



        super().__init__(**kwargs)


    def get_action_space(self):
        if self.action_space is not None:
            return self.action_space.clone()
        action_space = []
        for func_idx in range(1, len(self.functions) + 1) :
            for start in range(0, self.window_size - self.min_function_width) :
                for end in range(start + self.min_function_width, self.window_size ) :
                    action_space.append( [func_idx, start, end] )
            
        action_space.append( (self.eos_idx, 0, 0) )
        self.action_space = torch.tensor(action_space, dtype=torch.int16, device=self.device if self.device else 'cpu' )
        return self.action_space.clone()


    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
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
            return torch.ones(self.action_space_dim, dtype=torch.bool, device=self)
        # If sequence is not at maximum length, all actions are valid
        if state[-1] == self.padding:
            return torch.zeros(self.action_space_dim, dtype=torch.bool, device=self)
        # Otherwise, only EOS is valid
        mask = torch.ones(self.action_space_dim, dtype=torch.bool, device=self)
        mask[self.eos_idx] = False
        return mask
    


    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
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
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
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
        parent[last_func_idx] = self.pad_idx
        p_action = (state[last_func_idx],)
        return [parent], [p_action]
    


    def step(
            self, action: Tuple[int], skip_mask_check: bool = False
        ) -> [List[int], Tuple[int], bool]:
            """
            Executes step given an action.

            Args
            ----
            action : tuple
                Action to be executed. An action is a tuple int values indicating the
                dimensions to increment by 1.

            skip_mask_check : bool
                If True, skip computing forward mask of invalid actions to check if the
                action is valid.

            Returns
            -------
            self.state : list
                The sequence after executing the action

            action : tuple
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
            if action == self.eos:
                self.done = True
                return self.state, action, valid
            # Update state
            self.state[self._get_seq_length()] = action[0]
            return self.state, action, valid
    

    def state2readable(self, state: List[int] = None) -> str:
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
        return "".join([str(self.funcidx2token[int(idx)].__name__) + " " for idx in state])[:-1]
    



    def _pad(self, seq_list: list):
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
        return seq_list + [self.pad_idx] * (self.max_length - len(seq_list))

    def _unpad(self, seq_list: list):
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
        if self.pad_idx not in seq_list:
            return seq_list
        return seq_list[: seq_list.index(self.pad_idx)]

    def _get_seq_length(self, state: List[int] = None):
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
        if state[-1] == self.pad_idx:
            return state.index(self.pad_idx)
        else:
            return len(state)
"""
Scrabble environment: starting from an emtpy sequence, letters are added one by one up
to a maximum length.
"""

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

LETTERS = tuple(
    [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
)


class Scrabble(GFlowNetEnv):
    """
    Scrabble environment: sequences are constructed starting from an empty sequence and
    adding one letter at a time.

    States are represented by a list of indices corresponding to each letter, starting
    from 1, and are padded with index 0.

    Actions are represented by a single-element tuple with the index of the letter to
    be added. The EOS action is by (-1, ).

    Attributes
    ----------
    letters : tuple
        An tuple containing the letters to form words. By default, LETTERS is used.

    max_length : int
        Maximum length of the sequences. Default is 7, like in the standard game.

    pad_token : str
       PAD token. Default: "0".
    """

    def __init__(
        self,
        letters: Iterable = None,
        max_length: int = 7,
        pad_token: str = "0",
        **kwargs,
    ):
        # Main attributes
        if letters is None:
            self.letters = LETTERS
        else:
            self.letters = letters
        self.pad_token = pad_token
        self.n_letters = len(self.letters)
        self.max_length = max_length
        self.eos_idx = -1
        self.pad_idx = 0
        # Dictionaries / tensor representations
        # Keep a python list for readable token lookup (strings can't be stored in a numeric tensor)
        self.idx2token = [self.pad_token] + list(self.letters)  # index -> token string

        # Fast python lookup from token->index (used when converting readable -> state)
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}

        # Torch tensors for numeric operations: indices 0..n_letters
        self.idx2token_idx = torch.arange(len(self.idx2token), dtype=torch.long)
        # tensor of token indices in the same order as idx2token (useful for batch ops)
        self.token_idxs = torch.tensor([self.token2idx[t] for t in self.idx2token], dtype=torch.long)

        # Source state: list of length max_length filled with pad token
        self.source = torch.tensor([[self.pad_idx]] * self.max_length)
        # End-of-sequence action
        self.eos = torch.tensor([[self.eos_idx]])
        # Base class init
        super().__init__(**kwargs)
        self.source = self.source.to(self.device)
        self.token_idxs = self.token_idxs.to(self.device)
        self.eos = self.eos.to(self.device)
        self.idx2token_idx = self.idx2token_idx.to(self.device)


    def get_action_space(self) -> TensorType["action_space_dim", "action_dim"]:
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a single-element tuple indicating the index of the
        letter to be added to the current sequence (state).

        The action space of this parent class is:
            action_space: [(0,), (1,), (-1,)]
        """
        return torch.tensor([[self.token2idx[token]] for token in self.letters] + [[self.eos_idx]], device=self.device)

    def get_mask_invalid_actions_forward(
        self,
        state: Union[List[int], List[int]] = None,
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
        if (state == self.pad_idx).any():
            return torch.zeros(self.action_space_dim, dtype=torch.bool, device=self.device)
        # Otherwise, only EOS is valid
        mask = torch.ones(self.action_space_dim, dtype=torch.bool, device=self.device)
        mask[-1] = False
        if ~mask.any():
            raise ValueError("No valid actions found, there is likely a bug.")
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
            return state, torch.tensor([self.eos], device=self.device)
        if self.equal(state, self.source):
            return [], []
        pos_last_letter = self._get_seq_length(state) - 1
        parent = copy(state)
        parent[pos_last_letter] = self.pad_idx
        p_action = state[pos_last_letter]
        return parent, p_action
    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[TensorType["state_dim","action_dim"], TensorType["action_dim"], bool]:
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
        self.state : TensorType["state_dim","action_dim"]
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
        if action == self.eos:
            self.done = True
            return self.state, action, valid
        # Update state
        self.state[self._get_seq_length()] = action[0]
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum trajectory lenght is the maximum sequence length (self.max_length)
        plus one (EOS action).
        """
        return self.max_length + 1



    def action2index(self, action):
        """
        Converts an action into its index in the action space.

        Args
        ----
        action : tuple
            Action to be converted.

        Returns
        -------
        Index of the action in the action space.
        """
        if action == (self.eos_idx,):
            return self.action_space_dim - 1
        return self.token2idx[self.idx2token[action[0]]]




    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        return tlong(states, device=self.device)

    def states2policy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)

        # Replace any negative class values (e.g. EOS = -1) with pad_idx so one_hot
        # receives non-negative class indices. Also clamp upper bound to n_letters.


        return (
            F.one_hot(states, self.n_letters + 1)
            .reshape(states.shape[0], -1)
            .to(self.float)
        )

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
        return "".join([str(self.idx2token[idx]) + " " for idx in state])[:-1]

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a state in readable format into the "environment format" (tensor)

        Args
        ----
        readable : str
            A state in readable format - space-separated letters.

        Returns
        -------
        A tensor containing the indices of the letters.
        """
        if readable == "":
            return self.source
        return self._pad([self.token2idx[token] for token in readable.split(" ")])

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Args
        ----
        n_states : int
            The number of states to sample.

        seed : int
            Random seed.
        """
        n_letters = len(self.letters)
        n_per_length = n_letters ** torch.arange(
            1, self.max_length + 1, dtype=torch.long, device=self.device
        )
        lengths = Categorical(logits=n_per_length.repeat(n_states, 1), device = self.device).sample() + 1
        samples = torch.randint(
            low=1, high=n_letters + 1, size=(n_states, self.max_length), device=self.device
        )
        positions = torch.arange(self.max_length, device=self.device).unsqueeze(0)
        mask = positions >= lengths.unsqueeze(1)
        samples.masked_fill_(mask, 0)
        return samples

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

    def _unpad(self, state: TensorType):
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
        if torch.is_tensor(state):
            seq_list = state.flatten().tolist()


        if self.pad_idx not in seq_list:
            return seq_list
        return seq_list[: seq_list.index(self.pad_idx)]

    def _get_seq_length(self, state: TensorType["state_dim"] = None):
        """
        Returns the effective length of a state, that is ignoring the padding.

        Args
        ----
        state : list or tensor
            The input sequence. If None, self.state is used.

        Returns
        -------
        Length of the sequence, without counting the padding.
        """
        state = self._get_state(state)
        # ensure tensor on correct device / dtype
        state_t = tlong(state, device=self.device).flatten()
        # find first pad index (pad_idx == 0). If none, sequence is full.
        pad_positions = (state_t == self.pad_idx).nonzero(as_tuple=False)
        if pad_positions.numel() == 0:
            return int(state_t.numel())
        return int(pad_positions[0].item())

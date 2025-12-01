import torch
from gflownet.envs.verification_env import VerificationEnv
from gflownet.utils.verification_utils import ForceDataset, ForceDisplacementDataset
from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

class VerificationProxy(Proxy) :
    def __init__(self,
        production_data = False,
        reward_min: float = 0.1,
        do_clip_rewards: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reward_min = reward_min
        self.do_clip_rewards = do_clip_rewards
        self.production_data = production_data
        
    
    def setup(self, env: VerificationEnv = None):
        """
        TODO: Implement setup

        core idea: 
        fetch the data and sort the dataset by labels (to start with just binary lables)
        The data set has size M
        split the data set into groups of labels
        """
        if self.production_data:
            self.dataset = ForceDisplacementDataset(directory=env.data_path, window_size=env.window_size )
        else:
            self.dataset = ForceDataset(directory=env.data_path, window_size=env.window_size )
        
        self.func_dict = env.funcidx2token
        
        self.float = torch.float32

        # Build labels by iterating the dataset to avoid relying on dataset[:] slicing
        label_list = []
        for i in range(len(self.dataset)):
            _, lbl = self.dataset[i]
            if isinstance(lbl, torch.Tensor):
                label_list.append(int(lbl.item()))
            else:
                label_list.append(int(lbl))
        self.labels = torch.tensor(label_list, device=self.device, dtype=torch.int8)
        self.all_forces = torch.stack(self.dataset.labeled_forces['data'].tolist(), dim=0)  # (num_recordings, T)
        # Move the force tensor to the proxy device and cast to working float dtype
        self.all_forces = self.all_forces.to(device=self.device, dtype=self.float)





    def __call__(self, states):


        rewards = reward_function(self.apply_functions(states), self.labels)
        output = tfloat(rewards, device = self.device, float_type = self.float)

        return output 


    # def apply_functions(self, states: torch.TensorType):
        
    #     # state: ()
    #     function_results = torch.zeros(states.shape[0], states.shape[1], len(self.dataset), device = self.device, dtype = self.float)
        
    #     # num_recordings, n_split, recording_len // n_split
    #     segmented_data = self.dataset.segment_force_curves(n_splits=states.shape[1])
        
    def apply_functions(self, states: torch.Tensor):
        """
        Apply functions to inclusive [start, end] slices for all recordings.
        - states: (num_states, max_functions, 3) with (func_idx, start, end), end exclusive
        - self.dataset.labeled_forces: (num_recordings, T)
        - Each function in self.func_dict accepts variable-length slices (num_recordings, seg_len)
        Returns:
            function_results: (num_states, max_functions, num_recordings)
        """
        # Shapes
        num_states, max_functions, _ = states.shape

        # Build a single tensor of all recorded force traces from the dataset DataFrame.
        # ForceDataset stores tensors under the 'data' column, so stack them into a tensor:
        if len(self.dataset.labeled_forces) == 0 or self.all_forces.numel() == 0:
            raise ValueError("Dataset is empty")

        

        num_recordings, T = self.all_forces.shape

        # Move states to device and ensure integer indices
        states = states.to(device=self.device)
        func_ids = states[:, :, 0].to(dtype=torch.long)
        starts   = states[:, :, 1].to(dtype=torch.long)
        ends     = states[:, :, 2].to(dtype=torch.long)

        # Output: (num_states, max_functions, num_recordings)
        function_results = torch.zeros(
            num_states, max_functions, num_recordings,
            device=self.device, dtype=self.float
        )

        # Prefer iterating unique function ids in states to match whatever keys exist in func_dict
        unique_fids = torch.unique(func_ids)
        for f_id in unique_fids.tolist():
            if f_id not in self.func_dict:
                continue

            mask = (func_ids == f_id)  # (num_states, max_functions)
            if not mask.any():
                continue

            cur_func = self.func_dict[f_id]
            idxs = torch.nonzero(mask, as_tuple=False)  # [(state_idx, func_pos), ...]

            # Try torch.vmap; fall back to torch.func.vmap for older PyTorch
            try:
                vmap = torch.vmap
            except AttributeError:
                from torch.func import vmap

            for state_idx, func_pos in idxs:
                start = int(starts[state_idx, func_pos].item())
                end   = int(ends[state_idx, func_pos].item())  # exclusive

                # Exclusive slicing: use end 
                seg = self.all_forces[:, start:end]  # (num_recordings, seg_len)

                # Apply function across recordings
                per_rec = vmap(cur_func)(seg)  # -> (num_recordings,) or (num_recordings, k)

                # Squeeze trailing singleton if present
                if per_rec.dim() > 1 and per_rec.shape[-1] == 1:
                    per_rec = per_rec.squeeze(-1)

                # Store results
                function_results[state_idx, func_pos, :] = per_rec.to(dtype=self.float)

        return function_results
 


def reward_function(grouped_data,
                    labels, 
                    alpha=1.0, 
                    beta=1.0):
    """
    Apply score_grouping_and_separation to multiple groups of binary labelled points.
    
    Args:
        grouped_data: torch.Tensor of shape [num_groups, num_points_per_group, n_dims + 1]
                     Last dimension contains binary labels (0/1)
        alpha: weight for tightness score
        beta: weight for separation score
    
    Returns:
        aggregate_score or (aggregate_score, individual_scores) if return_individual=True
    """
    
    num_groups = grouped_data.shape[0]
    individual_scores = []
    valid_groups = []
    
    for group_idx in range(num_groups):
        group_data = torch.transpose(grouped_data[group_idx], 0,1)  
        
        try:
            score = score_grouping_and_separation(
                coordinates=group_data,
                labels=labels, 
                alpha=alpha, 
                beta=beta
            )
            individual_scores.append(score)
            valid_groups.append(group_idx)
            
        except ValueError as e:
            print(f"Warning: Group {group_idx} skipped - {str(e)}")
            continue
    
    if not individual_scores:
        raise ValueError("No valid groups found for scoring")
    
    individual_scores = torch.tensor(individual_scores)

    return individual_scores    

def score_grouping_and_separation(coordinates: torch.TensorType, 
                                labels, 
                                alpha=1.0, 
                                beta=1.0):
    """
    Score tight grouping of label 1 points and their distance from label 0 points.
    
    Args:
        points_with_labels: torch.Tensor of shape [N, n+1] where last column is binary label
        alpha: weight for tightness score (higher = more importance on tight grouping)
        beta: weight for separation score (higher = more importance on separation)
        return_components: if True, returns individual components
    
    Returns:
        Combined score (higher is better) or tuple of (tightness_score, separation_score, combined_score)
    """
    
    # Ensure tensors
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    labels = torch.as_tensor(labels)

    # Drop recordings that contain NaNs in any feature (these correspond to invalid / padded segments)
    invalid_rows = torch.isnan(coordinates).any(dim=1)
    if invalid_rows.any():
        # raise ValueError("Invalid (NaN) recordings available for this group")
        return torch.tensor(0.0)  # Return zero score if invalid recordings are present
    # coordinates = coordinates[valid_rows]
    # labels = labels[valid_rows]

    # Get indices for each class
    label_1_mask = labels == 1
    label_0_mask = labels == 0
    
    if not label_1_mask.any():
        raise ValueError("No points with label 1 found")
    if not label_0_mask.any():
        raise ValueError("No points with label 0 found")
    
    label_1_points = coordinates[label_1_mask, :]  # [N1, n]
    label_0_points = coordinates[label_0_mask, :]  # [N0, n]
    
    # 1. Tightness score: inverse of average pairwise distance within label 1
    if len(label_1_points) > 1:
        # Compute pairwise distances within label 1 group
        pairwise_dist_1 = torch.cdist(label_1_points, label_1_points, p=2)
        # Get upper triangle (excluding diagonal) to avoid double counting
        mask = torch.triu(torch.ones_like(pairwise_dist_1, dtype=torch.bool), diagonal=1)
        avg_internal_dist = pairwise_dist_1[mask].mean()
        tightness_score = 1.0 / (1.0 + avg_internal_dist)  # Higher score for tighter grouping
    else:
        tightness_score = torch.tensor(1.0)  # Perfect tightness for single point
    
    # 2. Separation score: minimum distance from any label 1 to any label 0
    distances_between_classes = torch.cdist(label_1_points, label_0_points, p=2)
    min_separation_distance = distances_between_classes.min()
    
    
    

    # Combine scores
    if tightness_score.isnan() or min_separation_distance.isnan():
        combined_score = torch.tensor(0.0)

    if tightness_score.isinf() or min_separation_distance.isinf():
        combined_score = torch.tensor(0.0)

    if min_separation_distance <= 1.0:
        combined_score = torch.tensor(0.0)
    
    else:

        combined_score = alpha * tightness_score + beta * min_separation_distance
    
    
    return combined_score
import torch
from gflownet.envs.verification_env import VerificationEnv
from gflownet.utils.verification_utils import ForceDataset
from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

class VerificationProxy(Proxy) :
    def __init__(self,
        reward_min: float = 0.0,
        do_clip_rewards: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reward_min = reward_min
        self.do_clip_rewards = do_clip_rewards
        
    
    def setup(self, env: VerificationEnv = None):
        """
        TODO: Implement setup

        core idea: 
        fetch the data and sort the dataset by labels (to start with just binary lables)
        The data set has size M
        split the data set into groups of labels
        """
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





    def __call__(self, states):


        rewards = reward_function(self.apply_functions(states), self.labels, beta=3)
        output = tfloat(rewards, device = self.device, float_type = self.float)

        return output 


    # def apply_functions(self, states: torch.TensorType):
        
    #     # state: ()
    #     function_results = torch.zeros(states.shape[0], states.shape[1], len(self.dataset), device = self.device, dtype = self.float)
        
    #     # num_recordings, n_split, recording_len // n_split
    #     segmented_data = self.dataset.segment_force_curves(n_splits=states.shape[1])
        
    def apply_functions(self, states: torch.TensorType):
        # states shape: (num_states, max_functions, 3) e.g., [100, 4, 3]
        #   That is 100 states of 4 functions each defined by (func_idx, start, end)
        # segmented_data shape: (num_recordings, max_functions, segment_length) e.g., [215, 4, start to end]
        # output shape: (num_states, max_functions, num_recordings) e.g., [100, 4, 215]
        function_results = torch.zeros(states.shape[0], states.shape[1], len(self.dataset),  device=self.device, dtype=self.float)
        
        
        #slice the dataset into segments based on start and end indices in states
        segmented_data = self.dataset.labeled_forces[:, states[:,:,1], states[:,:,2]]  # Shape: (num_recordings, max_functions, segment_length)


        # Apply functions to each segment based on states by vectorization
        for func_idx in range(1, len(self.func_dict) + 1):
            mask = states[:,:,0] == func_idx  # Shape: (num_states, max_functions)
            if mask.any():
                cur_func = self.func_dict[func_idx]
                # Select segments corresponding to the current function
                cur_data = segmented_data[:, mask]  # Shape: (num_recordings, num_selected_segments, segment_length)

                # cur_data: (num_recordings, num_selected_segments, seg_len)
                # For each selected segment j we want a vector of length num_recordings
                # with the function applied per recording. Compute per-segment results
                # and stack so res has shape (num_selected_segments, num_recordings[, out_dim]).
                num_recordings, num_selected_segments, _ = cur_data.shape
                res_list = []
                for j in range(num_selected_segments):
                    seg = cur_data[:, j, :]  # (num_recordings, seg_len)
                    per_rec = torch.vmap(cur_func)(seg)  # (num_recordings,) or (num_recordings, k)
                    res_list.append(per_rec)
                res = torch.stack(res_list, dim=0)  # (num_selected_segments, num_recordings, ...)

                # If functions return an extra singleton dim, squeeze it
                if res.dim() == 3 and res.shape[2] == 1:
                    res = res.squeeze(2)  # -> (num_selected_segments, num_recordings)

                res = res.to(dtype=self.float)

                # Assign results back to function_results.
                # idxs order matches the flattened boolean mask used by SegmentedData.__getitem__,
                # so enumerate(idx) corresponds to rows of `res`.
                idxs = torch.nonzero(mask, as_tuple=False)  # Get indices where mask is True
                for i, (state_idx, func_pos) in enumerate(idxs):
                    function_results[state_idx, func_pos, :] = res[i, :]


        
        # for func_idx in range(1, len(self.func_dict)) :
        #     mask = states[:,0] == func_idx
        #     cur_func = torch.vmap(self.func_dict[func_idx])
            

        #     for idx in range(mask.shape[0]) :
                
        #         state_mask = mask[idx,:]
        #         if state_mask.any():
                    
        #             cur_data = segmented_data[:,state_mask]
        #             res = cur_func(cur_data)
        #             # Handle different result shapes and data types
        #             if res.dim() == 1:
        #                 # Functions like min, max, argmin, argmax return 1D tensors
        #                 res = res.unsqueeze(-1)  # Shape: [215] -> [215, 1]
        #             elif res.dim() == 2 and res.shape[1] > 1:
        #                 # This shouldn't happen with your current setup, but just in case
        #                 pass
        #             # For trapezoid, res.dim() == 2 and res.shape[1] == 1, so no change needed
                    
        #             # Convert to the correct dtype
        #             res = res.to(dtype=self.float)
        #             # print(f"function: {self.func_dict[func_idx].__name__}",
        #             #     f"mask: {state_mask}",
        #             #     f"data shape: {cur_data.shape}",
        #             #       f"res shape: {res.shape}")
        #             function_results[idx, :, state_mask] = res
        #             #print(function_results)

        
        return function_results
    




        # # Apply each function to corresponding segments based on states
        # for segment_idx, data_segment in enumerate(segmented_data):
        #     for func_idx in range(1, len(self.func_dict)):
        #         # Find positions where this function should be applied
        #         mask = states == func_idx
        #         if mask.any():  # Only process if there are positions for this function
        #             # Apply the function to the corresponding data segment
        #             result = self.func_dict[func_idx](data_segment[func_idx])
        #             # Update function_results at masked positions
        #             # Assuming result needs to go into the segment_idx column of function_results
        #             function_results[:,:, segment_idx][mask] = result

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
    valid_rows = ~torch.isnan(coordinates).any(dim=1)
    if valid_rows.numel() == 0 or valid_rows.sum() == 0:
        raise ValueError("No valid (non-NaN) recordings available for this group")
    coordinates = coordinates[valid_rows]
    labels = labels[valid_rows]

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
    separation_score = min_separation_distance  # Higher score for better separation
    
    # 3. Combined score
    combined_score = alpha * tightness_score * beta * separation_score
    
    return combined_score
import torch
from torchmetrics.clustering import CalinskiHarabaszScore, DunnIndex
from torchmetrics.functional import f1_score as tm_f1_score
from gflownet.envs.verification_env import VerificationEnv
from gflownet.utils.verification_utils import ForceDataset, ForceDisplacementDataset
from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import cohen_kappa_score
import numpy as np

class VerificationProxy(Proxy) :
    def __init__(self,
        production_data = True,
        reward_min: float = 0.01,
        do_clip_rewards: bool = False,
        alpha = 1,
        beta = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reward_min = reward_min
        self.do_clip_rewards = do_clip_rewards
        self.production_data = production_data
        self.alpha = alpha
        self.beta = beta
        self.scoring_function = normalized_f1_score
        # self.min_f1_score = 0.95  # Minimum acceptable F1 score for verification

        # self.A = 100
        # self.B = 0
        # self.Q = 100 * self.min_f1_score
        # self.P = self.Q + (self.A-self.Q)/2
        # self.K = (self.A -self.Q) / (self.Q-self.B) ** (1/(self.P-self.Q))

    
    def setup(self, env: VerificationEnv = None):
        """
        core idea: 
        fetch the data and sort the dataset by labels (to start with just binary lables)
        The data set has size M
        split the data set into groups of labels
        """
        if self.production_data:
            self.dataset = ForceDisplacementDataset(directory=env.data_path, window_size=env.window_size )
        else:
            self.dataset = ForceDataset(directory=env.data_path, window_size=env.window_size, resolution=env.resolution)
        
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
        if self.production_data:
            key = 'force'
        else:
            key = 'data'
        

        self.all_forces = torch.stack(self.dataset.labeled_forces[key].tolist(), dim=0)  # (num_recordings, T)
        # Move the force tensor to the proxy device and cast to working float dtype
        self.all_forces = self.all_forces.to(device=self.device, dtype=self.float)

        # Calculate percentage of label 1 points
        num_label_1 = (self.labels == 1).sum().item()
        self.label_1_percentage = num_label_1 / self.labels.numel()


    def __call__(self, states):


        rewards = torch.clamp(self.get_score_by_group(self.apply_functions(states), self.labels, alpha=self.alpha, beta = self.beta), min=self.reward_min)
        output = tfloat(rewards, device = self.device, float_type = self.float)
        
        return output 

        
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
        
        starts   = states[:, :, 0].to(dtype=torch.long)
        ends     = states[:, :, 1].to(dtype=torch.long)
        func_ids = states[:, :, 2].to(dtype=torch.long)
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
 

    def get_score_by_group(self, grouped_data,
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
                score = self.scoring_function(
                    coordinates=group_data,
                    labels=labels, 
                    alpha=alpha, 
                    beta=beta
                )
                # # Soft plus to avoid zero scores
                #score = torch.log1p(torch.exp(score - self.min_f1_score))

                individual_scores.append(score)
                valid_groups.append(group_idx)
                
            except ValueError as e:
                print(f"Warning: Group {group_idx} skipped - {str(e)}")
                continue
        
        if not individual_scores:
            raise ValueError("No valid groups found for scoring")
        
        individual_scores = torch.tensor(individual_scores)

        return individual_scores    


def rank_normalize(data):
            """
            Rank-based normalization (quantile normalization).
            Maps each value to its percentile rank in [0, 1].
            Robust to any distribution shape.
            
            Args:
                data: torch.Tensor of shape [N, n]
            Returns:
                Normalized data of shape [N, n] with values in [0, 1]
            """
            N, n_dims = data.shape
            normalized = torch.zeros_like(data)
            
            for dim in range(n_dims):
                # Get values for this dimension
                values = data[:, dim]
                
                # Compute ranks (argsort twice gives ranks)
                sorted_indices = torch.argsort(values)
                ranks = torch.empty_like(sorted_indices, dtype=torch.float32)
                ranks[sorted_indices] = torch.arange(N, device=values.device, dtype=torch.float32)
                
                # Normalize ranks to [0, 1]
                if N > 1:
                    normalized[:, dim] = ranks / (N - 1)
                else:
                    normalized[:, dim] = 0.5
            
            return normalized


def IQR_normalize(data : torch.TensorType) -> torch.TensorType:
    """
    Robust normalization using median and IQR.
    Robust to outliers and non-parametric distributions.

    Args:
        data: torch.Tensor of shape [N, n]
    Returns:
        Normalized data of shape [N, n]
    """

    # Ensure data is tensor
    data = torch.as_tensor(data, dtype=torch.float32)

    # Ensure data is not mutated in-place
    data = data.clone()

    median = torch.median(data, dim=0, keepdim=True)[0]  # [1, n]

    # Calculate IQR (Interquartile Range)
    q75 = torch.quantile(data, 0.75, dim=0, keepdim=True)  # [1, n]
    q25 = torch.quantile(data, 0.25, dim=0, keepdim=True)  # [1, n]
    iqr = q75 - q25  # [1, n]

    # Avoid division by zero
    iqr = torch.where(iqr < 1e-8, torch.ones_like(iqr), iqr)

    # Normalize: (x - median) / IQR
    normalized = (data - median) / iqr

    return normalized

def min_normalized_distance_score(coordinates: torch.TensorType, 
                                  labels: torch.TensorType,
                                  alpha: float,
                                  beta: float) -> torch.TensorType:
    """
    Compute minimum normalized distance score between points with different labels.
    """
    normalized_coords = IQR_normalize(coordinates)
    distances_between_classes = torch.cdist(normalized_coords[labels == 1], normalized_coords[labels == 0], p=2)
    min_separation_distance = distances_between_classes.abs().min()
    return min_separation_distance


def norm_dunn_index_score(coordinates: torch.TensorType, 
                    labels: torch.TensorType, 
                    alpha: float, 
                    beta: float) -> torch.TensorType:

    normalized_coords = IQR_normalize(coordinates)
    
    distances_between_classes = torch.cdist(normalized_coords[labels == 1], normalized_coords[labels == 0], p=2)
    min_separation_distance = distances_between_classes.abs().min()

    # Compute intra-class distances
    distances_within_class_1 = torch.cdist(normalized_coords[labels == 1], normalized_coords[labels == 1], p=2)
    max_intra_class_distance = distances_within_class_1.abs().max()

    # Compute Dunn index
    dunn_index = min_separation_distance / (max_intra_class_distance + 1e-8)
    return dunn_index


def Calinski_Harabasz(coordinates: torch.TensorType, 
                      labels: torch.TensorType) -> torch.TensorType:
    """
    Compute Calinski-Harabasz index.
    """
    normalized_points = IQR_normalize(coordinates)

    # Between-cluster dispersion
    between_dispersion = torch.cdist(normalized_points[labels == 1], normalized_points[labels == 0], p=2).abs().sum()

    # Within-cluster dispersion
    within_dispersion = torch.cdist(normalized_points[labels == 1], normalized_points[labels == 1], p=2).abs().sum()

    ch_index = between_dispersion / (within_dispersion + 1e-8)
    return ch_index



def f1_score(coordinates: torch.TensorType, 
             labels, 
             k=10):
    """
    Compute F1 score using k-NN classification based on average distance to k nearest label
    
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary labels (0 or 1)
        k: number of nearest neighbors to consider 
    
    Returns:
        f1_score: float tensor with F1 score
    """



    label_1_mask = labels == 1

    # Compute pairwise distances from all points to label 1 points
    distances_to_label_1 = torch.cdist(coordinates, coordinates[label_1_mask], p=2)  # [N, N1]
    
    # Find k nearest label 1 neighbors for each point
    k_nearest_distances, _ = torch.topk(distances_to_label_1, k= k, 
                                    largest=False, dim=1)  # [N, k]
    
    # Average distance to k nearest label 1 points
    avg_distance_to_label_1 = k_nearest_distances.mean(dim=1)  # [N]
    
    # Compute threshold as the maximum average distance among label 1 points
    # This ensures all label 1 points are "inside" by definition
    threshold = avg_distance_to_label_1[label_1_mask].max()
    
    # Add 1 std deviation to threshold for some tolerance
    threshold += avg_distance_to_label_1[label_1_mask].std()
    
    # Predict labels: points within threshold are predicted as label 1
    predicted_labels = (avg_distance_to_label_1 <= threshold).long()

    f1 = tm_f1_score(predicted_labels, labels, task="binary")
    
    # f1 = sk_f1_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
    # f1 = torch.tensor(f1, device=coordinates.device, dtype=coordinates.dtype)
    return f1


def normalized_f1_score(coordinates: torch.TensorType,
                        labels: torch.TensorType,
                        alpha=1.0,
                        beta=1.0, 
                        k=10) -> torch.TensorType:
    """
    Compute normalized F1 score using k-NN classification based on average distance to k nearest label 1 points.
    """
    f1 = f1_score(IQR_normalize(coordinates), labels, k=k)
    return f1


def f1_reciprocal_score(coordinates: torch.TensorType, 
                        labels, 
                        k=5,
                        alpha=1.0, 
                        beta=1.0):       
    """
    Compute reciprocal of F1 score using k-NN classification based on average distance to k nearest label 1 points.
    """
    f1 = f1_score(coordinates, labels, k=k)
    return 1.0 / (1 - f1 + 1e-8) - 0.999



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




def score_knn_grouping_and_separation(coordinates: torch.TensorType,
                                labels,
                                alpha=1.0,
                                beta=1.0,
                                k=5):
    """
    Score based on false positives and false negatives using k-NN classification.
    Uses separation distance as a multiplier.
    
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary labels (0 or 1)
        alpha: weight for false positive penalty (higher = more penalty for FP)
        beta: weight for false negative penalty (higher = more penalty for FN)
        k: number of nearest neighbors to consider
        
    Returns:
        Combined score (higher is better, penalized by FP and FN, multiplied by separation)
    """
    # Ensure tensors
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    
    # Drop recordings that contain NaNs in any feature
    invalid_rows = torch.isnan(coordinates).any(dim=1)
    if invalid_rows.any():
        return torch.tensor(0.0)
    
    # Get indices for each class
    label_1_mask = labels == 1
    label_0_mask = ~label_1_mask
    
    if not label_1_mask.any():
        raise ValueError("No points with label 1 found")
    if not label_0_mask.any():
        raise ValueError("No points with label 0 found")
    
    label_1_points = coordinates[label_1_mask, :]  # [N1, n]
    label_0_points = coordinates[label_0_mask, :]  # [N0, n]
    
    n_label_1 = len(label_1_points)
    
    # Adjust k if needed
    k_actual = min(k, n_label_1)
    
    if k_actual < 1:
        return torch.tensor(0.0)
    
    try:
        # Compute F1 score using k-NN
        f1 = f1_score(coordinates, labels, k=k_actual)

        # # Normalize dimensions to estiamte "seperability"
        coordinates_normalized = IQR_normalize(coordinates)
        
        # # Extract normalized points for each label
        label_1_points_norm = coordinates_normalized[label_1_mask, :]
        label_0_points_norm = coordinates_normalized[label_0_mask, :]
        
        # # Calculate separation score on normalized coordinates
        distances_between_classes = torch.cdist(label_1_points_norm, label_0_points_norm, p=2)
        min_separation_distance = distances_between_classes.min()
        
        # Check for invalid values
        if f1.isnan() or min_separation_distance.isnan():
            combined_score = torch.tensor(0.0) 
        elif f1.isinf() or min_separation_distance.isinf():
            combined_score = torch.tensor(0.0)
        #elif min_separation_distance <= 0.0:
            #combined_score = torch.tensor(0.0)
        else:
            # Use separation as multiplier
            combined_score = f1 * min_separation_distance  
        # else:
        #         combined_score = torch.tensor(0.0)
        
        return combined_score
        
    except Exception as e:
        print(f"Warning: k-NN computation failed: {e}")
        raise e
        #return torch.tensor(0.0)

def identify_zero_dimension(coordinates: torch.TensorType):
    """
    Identify uninformative dimensions that could be removed or explained using a single dimension.
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
    Returns:
        zero_dim_indices: list of dimension indices with near-zero variance

    """
    u, s, v = torch.pca_lowrank(coordinates, q=coordinates.shape[1])
    zero_dim_indices = (s < 1).nonzero(as_tuple=True)[0].tolist()
    return zero_dim_indices

def score_dist_over_1_minus_f1(coordinates: torch.TensorType,
                                labels,
                                alpha=1.0,
                                beta=1.0):
    """
    Score based on minimum separation distance divided by (1 - F1 score).
    
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary labels (0 or 1)
        alpha: dummy parameter for compatibility
        beta: dummy parameter for compatibility
        
    Returns:
        Combined score defined as score = min_separation_distance / (1 - F1 score) (higher is better)
    """
    normalized_coords = IQR_normalize(coordinates.copy())

    f1 = f1_score(normalized_coords, labels, k=5)


    distances_between_classes = torch.cdist(normalized_coords[labels == 1], normalized_coords[labels == 0], p=2)

    # Find minimum separation distance
    min_separation_distance = distances_between_classes.abs().min()




    combined_score = min_separation_distance / (1.0 - f1 + 1e-8)

    # clamp to avoid extreme values
    combined_score = torch.clamp(combined_score, min=0.0, max=1e6)


    return combined_score

def clustered_f1_score(labels,
                         cluster_labels,
                         predicted_labels):
    """
    Compute weighted F1 score based on DBSCAN cluster assignments.
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary true labels (0 or 1)
        cluster_labels: array-like of shape [N] with cluster assignments from DBSCAN
        predicted_labels: torch.Tensor of shape [num_clusters] with predicted labels for each cluster
    Returns:
        total_f1: float tensor with weighted F1 score across clusters
        num_clusters: int with number of clusters considered
    """

    f1_total = torch.tensor(0.0, device=labels.device)
    N = labels.sum()

    clusters = torch.unique(cluster_labels)
    for cluster_id in clusters:
        if cluster_id == -1:
            continue  # Skip noise points

        # Get cluster label from predicted labels
        predicted_label = predicted_labels[cluster_id]

        cluster_mask = (cluster_labels == cluster_id)
        cluster_labels_in_data = labels[cluster_mask]
       

        # Calculate true positives, false positives, false negatives
        true_positives = ((predicted_label == 1) & (cluster_labels_in_data == 1)).sum().float()
        false_positives = ((predicted_label == 1) & (cluster_labels_in_data == 0)).sum().float()
        false_negatives = ((predicted_label == 0) & (cluster_labels_in_data == 1)).sum().float()


        # Calculate false positives and false negatives for this cluster
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # F1_i * N_i / N_total 
        N_i = cluster_mask.sum().item()
        f1_total += f1 * N_i / N

    return f1_total

def get_unique_cluster_predictions(cluster_labels, labels):
    """
    Assign predicted label to each cluster based on majority vote.
    
    Args:
        cluster_labels: array-like of shape [N] with cluster assignments from DBSCAN
        labels: torch.Tensor of shape [N] with binary true labels (0 or 1)
        
    Returns:
        predicted_labels: torch.Tensor of shape [N] with predicted labels based on cluster majority
    """
    
    unique_labels = torch.unique(cluster_labels)

    # Default predicted label to 0
    predicted_labels = torch.zeros_like(unique_labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_mask = (cluster_labels == cluster_id)
        cluster_labels_in_data = labels[cluster_mask]
        # Assign predicted label as majority label in cluster
        num_label_1 = cluster_labels_in_data.sum().item()
        num_label_0 = cluster_labels_in_data.shape[0] - num_label_1
        # Update predicted label for this cluster if majority is label 1
        if num_label_1 >= num_label_0:
            predicted_labels[cluster_id] = 1

    
    return predicted_labels

def DBSCAN_clustering_PCA(coordinates: torch.TensorType,
                                labels,
                                eps=0.1,
                                min_samples=5,
                                alpha=1.0,
                                beta=1.0):
    """
    Score based on DBSCAN clustering to identify tight groups and their separation.
    For each identified cluster calculate F1 score based on labels and combine based on number of points.
    
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary labels (0 or 1)
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        alpha: weight for tightness score
        beta: weight for separation score

    Returns:
        Combined score defined as score = F1 * alpha + separation_score * beta (higher is better)
    """
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    normalized_coords = IQR_normalize(coordinates)

    cluster_labels = torch.tensor(dbscan.fit_predict(normalized_coords.detach().cpu().numpy()), device=coordinates.device)

    predicted_labels = get_unique_cluster_predictions(cluster_labels, labels)

    # TODO: Incorporate tightness and separation calculations into combined score
    # # Compute tightness score (intra-cluster distance)
    # tightness_score = 0.0
    # for cluster_id in set(cluster_labels):
    #     if cluster_id == -1:
    #         continue  # Skip noise points
    #     cluster_points = coordinates[cluster_labels == cluster_id]
    #     if len(cluster_points) > 1:
    #         tightness_score += 1.0 / (1.0 + torch.mean(torch.pdist(cluster_points)))


    # Find FP and FN based on cluster assignments

    f1_total = clustered_f1_score(labels, cluster_labels, predicted_labels)


    zero_dims = identify_zero_dimension(coordinates)
    
    score = f1_total / (1 + len(zero_dims))

    return score



def HDBSCAN_clustering_PCA(coordinates: torch.TensorType,
                                labels,
                                min_cluster_size=5,
                                alpha=1.0,
                                beta=1.0):
    """
    Score based on HDBSCAN clustering to identify tight groups and their separation.
    For each identified cluster calculate F1 score based on labels and combine based on number of points.
    
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary labels (0 or 1)
        min_cluster_size: HDBSCAN min_cluster_size parameter
        alpha: weight for tightness score
        beta: weight for separation score

    Returns:
        score: weighted F1 score adjusted for uninformative dimensions (higher is better)
    """
    # Apply HDBSCAN clustering
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
    normalized_coords = IQR_normalize(coordinates)

    cluster_labels = torch.tensor(hdbscan.fit_predict(normalized_coords.detach().cpu().numpy()), device=coordinates.device)

    predicted_labels = get_unique_cluster_predictions(cluster_labels, labels)

    # TODO: Incorporate tightness and separation calculations into combined score
    # # Compute tightness score (intra-cluster distance)
    # tightness_score = 0.0
    # for cluster_id in set(cluster_labels):
    #     if cluster_id == -1:
    #         continue  # Skip noise points
    #     cluster_points = coordinates[cluster_labels == cluster_id]
    #     if len(cluster_points) > 1:
    #         tightness_score += 1.0 / (1.0 + torch.mean(torch.pdist(cluster_points)))


    # Find FP and FN based on cluster assignments

    f1_total = clustered_f1_score(labels, cluster_labels, predicted_labels)


    zero_dims = identify_zero_dimension(coordinates)

    score = f1_total / (1 + len(zero_dims))

    return score

# def feature_entropy_score(coordinates: torch.TensorType,
#                           labels: torch.TensorType,
#                           alpha=1.0,
#                           beta=1.0):

#     """
#     Score based on feature entropy to identify uninformative dimensions.
#     Args:
#         coordinates: torch.Tensor of shape [N, n] with point coordinates
#         labels: torch.Tensor of shape [N] with binary labels (0 or 1)
#         alpha: dummy parameter for compatibility
#         beta: dummy parameter for compatibility

#     Returns:
#         score: float, the computed feature entropy score
#     """
#     label_1



#     return score

def score_dist_over_1_minus_DBSCAN_f1(coordinates: torch.TensorType,
                                labels,
                                alpha=1.0,
                                beta=1.0):
    """
    Score based on minimum separation distance divided by (1 - F1 score).
    
    Args:
        coordinates: torch.Tensor of shape [N, n] with point coordinates
        labels: torch.Tensor of shape [N] with binary labels (0 or 1)
        alpha: dummy parameter for compatibility
        beta: dummy parameter for compatibility
        
    Returns:
        Combined score defined as score = min_separation_distance / (1 - F1 score) (higher is better)
    """
    normalized_coords = IQR_normalize(coordinates)

    f1 = DBSCAN_clustering_PCA(normalized_coords, labels, eps=0.1, min_samples=5)


    distances_between_classes = torch.cdist(normalized_coords[labels == 1], normalized_coords[labels == 0], p=2)

    # Find minimum separation distance
    min_separation_distance = distances_between_classes.abs().min()




    combined_score = min_separation_distance / (1.0 - f1 + 1e-8)

    # clamp to avoid extreme values
    combined_score = torch.clamp(combined_score, min=0.0, max=1e6)


    return combined_score
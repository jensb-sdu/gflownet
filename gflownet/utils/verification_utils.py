import torch
import numpy as np
import scipy
import ast
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from matplotlib import pyplot as plt
from itertools import combinations
from torch.distributions import Categorical

#import umap

class ForceDataset(Dataset):
    def __init__(self, directory = None, transform=None, window_size = 1024, sim_data = False, filter = False):
        self.directory = directory
        self.transform = transform
        self.window_size = window_size
        self.labeled_forces = pd.DataFrame()  # Initialize an empty DataFrame

        if directory is not None:
            self.load_forces(directory)





    def __len__(self):
        return len(self.labeled_forces)

    def __getitem__(self, idx):

        forces = self.labeled_forces['data'][idx]

        label = self.labeled_forces['label'][idx] #.unsqueeze_(0)


        return forces, label
    
    def load_forces(self, directory):
        self.labeled_forces = self.get_full_dataframe_from_directory(directory)


    def get_full_dataframe_from_directory(self, directory_path):
        """
        Parallelized version of get_fulldatafrane_from_directory.
        Processes all files in the directory and combines them into a single DataFrame.
        """
        # List all files in the directory

        files = [file for file in Path(directory_path).iterdir()]
        # Use ProcessPoolExecutor for parallel processing     
        with ProcessPoolExecutor() as executor:
            # Map the process_file function to the list of files
            dataframes = list(executor.map(self.process_file, files))

        # Ensure all elements in dataframes are DataFrames
        for i, df in enumerate(dataframes):
            if not isinstance(df, pd.DataFrame):
                print(f"Error: Output of process_file for file {files[i]} is not a DataFrame. Got: {type(df)}")
                continue

        # Combine all DataFrames into a single DataFrame
        #full_dataframe = pd.concat(dataframes, ignore_index=False)
        return pd.concat(dataframes, ignore_index=True)
    
    def process_file(self, file_path):
        """
        Process a single CSV file and return a DataFrame.
        This function reads the file, extracts forces, positions, and labels,
        and returns a DataFrame with the extracted data.
        """
        data = []
        label = None

        
        # Read the CSV file into a list of lines
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Check if the last line contains a valid label
            try:
                label = int(lines[-1].strip())
            except ValueError:
                print(f"Error: Invalid label in file {file_path}")
                return pd.DataFrame()

            # Convert each line to a list using ast.literal_eval
            for line in lines[:-1]:  # Exclude the last line
                try:
                    # Split the line into forces and positions
                    forces_str = line.strip().split('\t')[0]
                    forces = ast.literal_eval(forces_str)[0]
                    data.append(forces)
                except (SyntaxError, ValueError, TypeError) as e:
                    print(f"Error: Unable to process line {line} in file {file_path}: {e}")


        z_force = [force[2] for force in data]

        z_force = preprocess_force(z_force, self.window_size)
        

        file_df = pd.DataFrame({'data':[torch.tensor(z_force.copy(), dtype=torch.float32)], 'label': label})
        return file_df
    

    def segment_force_curves(self, n_splits, split_dim=0):

        split_data = []
        
        for idx, force_curve in enumerate(self.labeled_forces['data']):
            # Split tensor into n parts
            splits = torch.chunk(force_curve, n_splits, dim=split_dim)
            
            split_data.append(torch.stack(splits))
                
        return torch.stack(split_data)


class LargeCategorical(Categorical):
        

    def sample(self, sample_shape=...):
        
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = np.random.Generator.multinomial(n = sample_shape.numel(), pvals=probs_2d)
        return samples_2d.reshape(self._extended_shape(sample_shape))
    
    def _chunked_categorical_sample(self, extended_shape: torch.Size) -> torch.Tensor:
        """
        Sample from categorical distribution with large number of categories
        using a chunked approach.
        """
        if not isinstance(extended_shape, torch.Size):
            extended_shape = torch.Size(extended_shape)
        # Get probabilities
        if self.probs is not None:
            probs = self.probs
        else:
            probs = F.softmax(self.logits, dim=-1)
        
        # Reshape for batch processing
        batch_shape = probs.shape[:-1]
        num_categories = probs.shape[-1]
        probs_2d = probs.reshape(-1, num_categories)
        
        # Calculate number of samples needed
        num_samples = extended_shape.numel() // probs_2d.shape[0]
        
        # Use inverse transform sampling for large category counts
        samples_list = []
        
        for i in range(probs_2d.shape[0]):
            batch_probs = probs_2d[i]
            batch_samples = self._inverse_transform_sample(num_samples)
            samples_list.append(batch_samples)
        
        samples_2d = torch.stack(samples_list, dim=0)
        return samples_2d.reshape(extended_shape)

    def _inverse_transform_sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample using inverse transform method (CDF-based sampling).
        This works well for any number of categories.
        """
        device = self.probs.device
        dtype = self.probs.dtype
        
        # Compute cumulative distribution
        cumulative_probs = torch.cumsum(self.probs, dim=0)
        
        # Generate uniform random numbers
        uniform_samples = torch.rand(num_samples, device=device, dtype=dtype)[None,:]
        
        # Use searchsorted to find the category indices
        # Add small epsilon to handle edge cases
        samples = torch.searchsorted(cumulative_probs, uniform_samples, right=False)
        
        # Clamp to valid range (shouldn't be necessary but good for safety)
        samples = torch.clamp(samples, 0, len(self.probs) - 1)
        
        return samples
    


    







def windowed_signal(data, minimum_idx, window_size):

    """Extract a windowed signal around the minimum index.
    Args:
        data (list or np.ndarray): Input data to extract the window from.
        minimum_idx (int): Index of the minimum value around which to extract the window.
        window_size (int): Size of the window to extract.
    Returns:   
        np.ndarray: Windowed data around the minimum index.
    """
    start = 0
    end = 0

    half_len = window_size // 2  # Half the window size
    data_len = half_len * 2  # Ensure the length is even
    if minimum_idx - half_len >= 0 and minimum_idx + half_len <= len(data):
        # Case: Window fits within the series
        start = minimum_idx - half_len
        end = minimum_idx + half_len
    elif minimum_idx - half_len < 0:
        # Case: Window extends before the start of the series
        end = data_len
    elif minimum_idx + half_len > len(data):
        # Case: Window extends beyond the end of the series
        end = len(data) - 1
        start = end - data_len
    
    windowed_data = data[start:end]
    
    return windowed_data, start, end











def low_pass_filter(data, cutoff_freq, sampling_rate = 500.0):
    """Apply a low-pass filter to the data."""
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data



def preprocess_force(force, window_size):



    force = low_pass_filter(force, cutoff_freq=25.0, sampling_rate=500)
    peaks = scipy.signal.find_peaks(-force, prominence = 4)[0]
    prominences = scipy.signal.peak_prominences(-force, peaks)[0]
    window_mid = 0
    if len(prominences) :
    
        window_mid = peaks[0]
    else:
        window_mid = force.argmin()



    force, start, end = windowed_signal(force, window_mid, window_size)



    return force


def plot_umap_dimensions(points, labels, color_scheme = None):



    pass


def plot_all_dimension_combinations(points, masks, labels=None, point_colors=None):
    """
    Plot all unique combinations of dimensions for N-dimensional points
    
    Parameters:
    points: array-like, shape (n_samples, n_dimensions)
    labels: list of strings, dimension labels (optional)
    point_colors: array-like, colors for each point (optional)
    """
    
    # Convert to numpy array if not already
    points = np.array(points)
    n_samples, n_dimensions = points.shape
    
    # Generate dimension labels if not provided
    if labels is None:
        labels = [f'Dim_{i}' for i in range(n_dimensions)]
    
    # Get all unique combinations of 2 dimensions
    dim_combinations = list(combinations(range(n_dimensions), 2))
    n_combinations = len(dim_combinations)
    
    # Calculate subplot grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_combinations)))
    n_rows = int(np.ceil(n_combinations / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 8*n_rows))
    
    # Handle case where we have only one subplot
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each combination
    for idx, (dim1, dim2) in enumerate(dim_combinations):
        ax = axes[idx]
        
        # Extract the two dimensions
        x1 = points[masks[0], dim1]
        y1 = points[masks[0], dim2]
        x2 = points[masks[1], dim1]
        y2 = points[masks[1], dim2]
        

        scatter1 = ax.scatter(x1, y1, alpha=0.7, c = 'red')
        scatter2 = ax.scatter(x2, y2, alpha=0.7, c= 'green')
        
        # Set labels and title
        # ax.set_xlabel(labels[dim1])
        # ax.set_ylabel(labels[dim2])
        ax.set_title(f'{dim1}: {labels[dim1]} vs {dim2}: {labels[dim2]}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_combinations, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(pad=2)
    return fig, axes
 





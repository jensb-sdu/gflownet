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
from gflownet.envs.verification_env import FUNCTIONS
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
 

def read_gfn_samples(sample_path):
    df = pd.read_csv(sample_path)
    if 'readable' not in df.columns or 'energies' not in df.columns:
        raise ValueError("CSV must contain 'readable' and 'energies' columns")

    readable_list = df['readable'].astype(str).tolist()
    energies = df['energies'].astype(float).to_numpy()
    return readable_list, energies



def plot_gfn_samples_umap(sample_path, n_points=1000):
    """
    Plot GFlowNet samples in 2D UMAP space colored by energy.

    Args:
        sample_path: path to CSV containing columns 'readable' and 'energies'
        n_points: number of points to sample for plotting
    """
    try:
        import umap
    except Exception:
        raise ImportError("umap-learn is required for plot_gfn_samples")

    # Read CSV and extract columns
    readable_list, energies = read_gfn_samples(sample_path)
    # Simple tokenization + count-vectorizer (vocabulary from tokens in readable strings)
    token_lists = [s.split() for s in readable_list]
    vocab = {}
    for tokens in token_lists:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    # Build count vectors
    vectors = np.zeros((len(token_lists), len(vocab)), dtype=float)
    for i, tokens in enumerate(token_lists):
        for t in tokens:
            vectors[i, vocab[t]] += 1.0

    # Compute UMAP embeddings
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings = reducer.fit_transform(vectors)

    # Sample a subset for plotting if necessary
    indices = np.arange(len(embeddings))
    if len(embeddings) > n_points:
        indices = np.random.choice(len(embeddings), n_points, replace=False)
    emb_plot = embeddings[indices]
    energies_plot = energies[indices]

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_plot[:, 0], emb_plot[:, 1], c=energies_plot, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Energy')
    plt.title('GFlowNet Samples in UMAP Space Colored by Energy')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig("gfn_samples_umap.png")
    plt.close()




def plot_gfn_functions_3D(sample_path, n_points=1000):
    """
    Plot GFlowNet sampled functions in 3D space. Each token is shown as a point
    colored by its energy. Uses function indices from FUNCTIONS (verification_env).
    """
    import re

    readable_list, energies = read_gfn_samples(sample_path)

    # Parse readable strings into list of (name, start, end)
    parsed_funcs = []
    func_names = set()
    for s in readable_list:
        matches = re.findall(r'([A-Za-z0-9_+-]+)\[(\d+),\s*(\d+)\]', s)
        token_list = []
        for name, a, b in matches:
            a = int(a)
            b = int(b)
            token_list.append((name, a, b))
            func_names.add(name)
        parsed_funcs.append(token_list)

    if len(parsed_funcs) == 0:
        raise ValueError("No parsable function tokens found in 'readable' column.")

    # Map function name -> env index using FUNCTIONS (1-based)
    func_to_idx = {}
    for i, f in enumerate(FUNCTIONS, start=1):
        if callable(f) and hasattr(f, "__name__"):
            fname = f.__name__
        else:
            fname = str(f)
        func_to_idx[fname] = i

    # Fallback for unseen names
    next_idx = max(func_to_idx.values()) + 1 if func_to_idx else 1
    for name in sorted(func_names):
        if name not in func_to_idx:
            func_to_idx[name] = next_idx
            next_idx += 1

    # Subsample indices if needed
    indices = np.arange(len(parsed_funcs))
    if len(indices) > n_points:
        indices = np.random.choice(indices, n_points, replace=False)

    energies_arr = np.array(energies)
    vmin = np.nanmin(energies_arr)
    vmax = np.nanmax(energies_arr)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 3D scatter plotting (no connecting lines)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for idx in indices:
        func = parsed_funcs[int(idx)]
        if not func:
            continue
        # base x indices from env mapping
        xs_base = [func_to_idx.get(name, -1) for name, _, _ in func]
        # filter invalid mappings
        valid = [(x, s, e) for x, (name, s, e) in zip(xs_base, func) if x >= 0]
        if not valid:
            continue
        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        zs = [v[2] for v in valid]

        # add small deterministic jitter to x to separate stacked tokens visually
        n_tokens = len(xs)
        if n_tokens > 1:
            jitter = np.linspace(-0.2, 0.2, n_tokens)
        else:
            jitter = [0.0]
        xs_jitter = np.array(xs, dtype=float) + jitter

        energy = float(energies_arr[int(idx)])
        color = cmap(norm(energy))
        ax.scatter(xs_jitter, ys, zs, c=[color], s=30, depthshade=True)

    ax.set_xlabel('Function (env index, 1-based)')
    ax.set_ylabel('Start Position')
    ax.set_zlabel('End Position')
    ax.set_title('GFlowNet Sampled Functions in 3D Space (points colored by energy)')

    # colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(energies_arr)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Energy')

    plt.savefig("gfn_functions_3D.png")
    plt.close()  





if __name__ == "__main__":
    sample_path = "/home/dmd_user/Desktop/ECAA/gfn_verification/gflownet/samples/gfn_samples._8_funcs_nan_zero_no_duplicates.csv"
    print("Plotting GFlowNet samples...")
    plot_gfn_samples_umap(sample_path, n_points=1000)
    plot_gfn_functions_3D(sample_path, n_points=1000)
    print("Plots saved.")
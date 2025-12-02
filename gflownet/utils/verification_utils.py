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

import csv
import os
#import umap



from datetime import datetime


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
        if directory is not None:
            dir = Path(directory)
        else:
            dir = self.directory
        try :
            self.labeled_forces = self.get_full_dataframe_from_directory(dir)
        except Exception as e:
            print(f"Error loading forces from directory {dir}: {e}")
            self.labeled_forces = pd.DataFrame()
            raise e

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

class ForceDisplacementDataset(ForceDataset):
    def __init__(self, directory = None, transform=None, AlignmentX = 50, MinX = -10, MaxX = 30, window_size = 500):

        self.AlignmentX = AlignmentX  # Force threshold for alignment
        self.MinX = MinX
        self.MaxX = MaxX
        super().__init__(directory, transform, window_size)


    def __getitem__(self, idx):
        
        forces = self.labeled_forces['force'][idx]
        label = self.labeled_forces['label'][idx] #.unsqueeze_(0)
        return forces, label

    def legend_without_duplicate_labels(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique),fontsize=10)



    def get_full_dataframe_from_directory(self, directory_path):
        """
        Parallelized version of get_fulldatafrane_from_directory.
        Processes all files in the directory and combines them into a single DataFrame.
        Since the folder structure is different for force-displacement data, find all subfolders first.
        Use the folder names as labels for the data.
        """
        # Find all subfolders in the directory
        subfolders = [f.path for f in os.scandir(directory_path) if f.is_dir()]

        # Find subfolder named good or reference to use as label 1
        label_map = {}
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder).lower()
            if 'good' in folder_name or 'reference' in folder_name:
                label_map[subfolder] = 1
            else:
                label_map[subfolder] = 0
        
        # each sub folder contains folders from different tests
        files = []
        for subfolder in subfolders:
            for case in os.scandir(subfolder):
                if case.is_dir():
                    for f in os.scandir(case.path):
                        if f.is_file() and f.name.endswith('.csv'):
                            files.append((f.path, label_map[subfolder]))  # Store file path with its label
        
        # # Use ProcessPoolExecutor for parallel processing     
        # with ProcessPoolExecutor() as executor:
        #     # Map the process_file function to the list of files
        #     dataframes = list(executor.map(self.process_file, files))


        dataframes = []
        for file in files:
            df = self.process_file(file)

            dataframes.append(df)

        # Ensure all elements in dataframes are DataFrames
        for i, df in enumerate(dataframes):
            if not isinstance(df, pd.DataFrame):
                print(f"Error: Output of process_file for file {files[i]} is not a DataFrame. Got: {type(df)}")
                continue

        # Combine all DataFrames into a single DataFrame
        #full_dataframe = pd.concat(dataframes, ignore_index=False)
        return pd.concat(dataframes, ignore_index=True)
        

    # Functions to read csv file data no matter the maXYmos firmware version  
    def read_csv_first_six_lines(self, filename):
        first_six_lines = []

        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')

            for _ in range(6):
                first_six_lines.append(next(reader))

        keys = first_six_lines[4][0].split(';')
        values = first_six_lines[5][0].split(';')

        data_dict = {key: value for key, value in zip(keys, values)}

        return first_six_lines, data_dict

    def read_lines_until_empty(self, filename):

        if filename.endswith(".csv"): 
            first_six_lines, data_dict = self.read_csv_first_six_lines(filename)

            keys_to_extract = ["Measuring points"]
            start_indices = [data_dict[key] for key in keys_to_extract if key in data_dict]

            with open(filename, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')

                for start_index in start_indices:
                    start_index = int(start_index)
                    lines = []
                    file.seek(0)  
                    for i, row in enumerate(reader, start=1):
                        if i >= start_index:
                            if not any(row):  # Check if the row is empty
                                break
                            split_row = row[0].split(';') if row else []
                            lines.append(split_row)
            return np.array(lines, dtype='str')
        return np.array([])



    def find_min_and_index(self, data):
        """
        Returns the minimum value and its index from a list or numpy array.
        
        Parameters:
            data (list or np.ndarray): Input data.
            
        Returns:
            tuple: (min_value, index)
        """
        data_array = np.array(data)
        min_index = np.argmin(data_array)
        min_value = data_array[min_index]
        return min_value, min_index


    def process_file(self, file):
        file_path, label = file
        data = self.read_lines_until_empty(file_path)
                    
        # Conversion from string to float
        load = np.array([float(data[j,2].replace(",",".")) for j in range(len(data))])
        disp = np.array([float(data[j,1].replace(",",".")) for j in range(len(data))])
        

        
        # Align on the x axis - set x=0 at first threshold crossing
        idx_above = np.where(load >= self.AlignmentX)[0]
        if len(idx_above) > 0:
            threshold_idx = idx_above[0]  # First time load crosses threshold
            disp = disp - disp[threshold_idx]  # Set this point as x=0





        # Remove initial points goinfing backward (right to left)
        start_idx = 0
        for j in range(1, len(disp)):
            # Look for the point where displacement starts increasing consistently
            if j < len(disp) - 5:  # Need at least 5 points ahead
                # Check if next 5 points show increasing trend
                if all(disp[j+k] > disp[j] for k in range(1, min(5, len(disp)-j))):
                    start_idx = j
                    break
        # Use data only from start_idx onward
        disp = disp[start_idx:]
        load = load[start_idx:] 


        # Window signal
        # Cut where disp less or equal to MinX
        idx_leq = np.where(disp <= self.MinX)[0][-1]
        idx_geq = np.where(disp >= self.MaxX)[0][0]
        disp = disp[idx_leq:idx_geq]
        load = load[idx_leq:idx_geq]



        if self.window_size is not None:
            # Interpolate to fixed length
            disp_new = np.linspace(disp[0], disp[-1], self.window_size) 
            load = np.interp(disp_new, disp, load)
        else:
            disp_new = disp




        file_df = pd.DataFrame({'force':[torch.tensor(load.copy(), dtype=torch.float32)], 'displacement': [torch.tensor(disp_new.copy(), dtype=torch.float32)], 'label': int(label)})

        return file_df


    def load_and_process_force_data(self ):
        
        # Folder and file names
        #folder_Data = str(path) + "/test/Data_aspect-ratio/" # Folder containing good files must be names "Good" and be located in script folder
        
      
        # Ensure path exists
        folder_Data = os.path.abspath(self.directory)
        # Raise error if path does not exist
        if not os.path.exists(folder_Data):
            raise FileNotFoundError(f"The specified folder does not exist: {folder_Data}")


        Results_folder = self.directory / "Plots" # Folder for output of result files
        # Ensure results folder exists
        Results_folder = os.path.abspath(Results_folder)
        if not os.path.exists(Results_folder):
            os.makedirs(Results_folder)
            

        print(f"Getting data from folder: {folder_Data}")

       

        #############################################
        ############ Save/load structure ############
        #############################################

        now = str(datetime.now())
        now = now.replace(" ", "_")
        now = now.replace(":", "-")


        # Find paths of all subfolders in folder_Data# 
        subfolders = [f.path for f in os.scandir(folder_Data) if f.is_dir()]

        extension = '.csv'
        colorvector = ["Green","Black","Blue","Red","purple","teal","Orange","Lightblue","magenta"]

        #############################################
        ########### Load data from files ############
        #############################################


        # Dictionary to store raw data before and after alignment
        raw_data_dict = {}

        # loop through all folders
        for subfolder in subfolders:

            # Extract folder name from path (get the last folder name)
            folder_name = os.path.basename(subfolder)
            
            # Skip if this is the root data folder itself
            if folder_name == 'Data_aspect-ratio':
                continue
            
            # Get only CSV files
            fileNames = [fn for fn in os.listdir(subfolder) if fn.endswith(extension)]
            nof = len(fileNames)
            
            if nof > 0:
                data_sorted = [[] for _ in range(2)]  # Only need displacement and load
                data_before_alignment = [[] for _ in range(2)]  # Store unaligned data
                
                # Load all data from the data files in the current folder
                for fname in fileNames:
                    self.process_file(Path(subfolder) / fname)
                
                # Store both aligned and unaligned data
                raw_data_dict[folder_name] = {
                    'before': data_before_alignment,
                    'after': data_sorted
                }
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(nof):
                    disp_i = data_sorted[0][i]
                    load_i = data_sorted[1][i]
                    
                    # Remove initial points where displacement goes backward (right to left)
                    # Find where displacement starts consistently increasing
                    start_idx = 0
                    for j in range(1, len(disp_i)):
                        # Look for the point where displacement starts increasing consistently
                        if j < len(disp_i) - 5:  # Need at least 5 points ahead
                            # Check if next 5 points show increasing trend
                            if all(disp_i[j+k] > disp_i[j] for k in range(1, min(5, len(disp_i)-j))):
                                start_idx = j
                                break
                    
                    # Use data only from start_idx onward
                    disp_i = disp_i[start_idx:]
                    load_i = load_i[start_idx:]

                    #add to plot
                    ax.plot(disp_i, load_i, color=colorvector[f % len(colorvector)], alpha=0.5)
                
                ax.set_title(f'Force-Displacement Curves - {folder_name}')
                ax.set_xlabel('Displacement (deg)')
                ax.set_ylabel('Force (N)')
                # show plot
                plt.savefig(f"{Results_folder}Force-Displacement_Curves_{folder_name}.png", dpi=300)
                plt.close()

                print(f"Processed folder: {folder_name} with {nof} files.")
                # Compute avg length after alignment
                lengths = [len(data_sorted[0][i]) for i in range(nof)]
                avg_length = int(np.mean(lengths))

                print(f"Average length of aligned curves in folder '{folder_name}': {avg_length} points.")




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




def plot_gfn_functions_3D(sample_path, funcs, n_points=1000):
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
    for i, f in enumerate(funcs, start=1):
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

def find_best_function_sample(sample_path):
    """
    Find the function sample with the highest energy from GFlowNet samples.
    """
    readable_list, energies = read_gfn_samples(sample_path)
    max_energy_idx = np.argmax(energies)
    best_function = readable_list[max_energy_idx]
    best_energy = energies[max_energy_idx]
    return best_function, best_energy

def display_best_function_over_curve(sample_path, proxy, env, number_of_curves = 10):
    best_function, best_energy = find_best_function_sample(sample_path)

    
    state_tensor = env.readable2state(best_function)

    state_list = state_tensor.tolist()[0]  # assuming batch size 1
    # Visualize on a subset of sample force curves
    sample_forces = proxy.all_forces[:number_of_curves]
    sample_labels = proxy.labels[:number_of_curves]

    plt.figure(figsize=(10,5))
    ax = plt.gca()

    # Plot force curves with function overlays
    # collect numeric arrays to compute axis extents for annotation placement
    numeric_forces = []
    for i in range(len(sample_forces)):
        sample_force = sample_forces[i]
        sample_label = sample_labels[i]
        if torch.is_tensor(sample_force):
            arr = sample_force.detach().cpu().numpy()
        else:
            arr = np.asarray(sample_force)
        numeric_forces.append(arr)
        # Plot individual force curve color coded by label
        ax.plot(arr, color = 'green' if sample_label == 1 else 'red', alpha=0.3)

    # determine y placement for the dimension annotation
    ymax = 0.0
    ymin = -80.0
    y_offset = (ymax - ymin) / len(state_list)

    cmap = plt.get_cmap('viridis')
    # fallback to FUNCTIONS length (proxy not passed here)
    n_funcs = len(env.functions)
    norm = plt.Normalize(vmin=0, vmax=max(1, n_funcs - 1))

    for idx, func in enumerate(state_list):
        func_idx, start, end = func
        color = cmap(norm(idx))
        # Overlay function region color coded by function index
        #ax.axvspan(start, end, alpha=0.3, label=f'{func_idx}: {func_name} [{start}, {end}]', color=color)
        # add vertical markers at boundaries
        ax.axvline(start, color=color, linestyle='--', alpha=0.6)
        ax.axvline(end, color=color, linestyle='--', alpha=0.6)
        y_text = ymax - (idx * y_offset)
        # add dimension annotation of the format  func_name     centered across the span
        #                                       |-----------| 
        cur_func = proxy.func_dict.get(func_idx, None)
        dim_text = f"{idx}: {cur_func.__name__ if callable(cur_func) else str(cur_func)}"
        ax.text((start + end) / 2.0, y_text, dim_text, ha='center', va='bottom',
                fontsize=9, color=color)
        ax.plot([start, end], [y_text - 1, y_text - 1], linestyle = "--", color=color, alpha=0.5, linewidth=2)

    ax.set_title('Best GFlowNet Function Applied to Sample Force Curve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force')
    ax.grid(True)
    plt.savefig("best_function_on_force_curve.png")
    plt.close()
    print("Best function visualization saved.")


def plot_best_function_coordinates_UMAP(best_function, proxy, env):
    #parse best function into state tensor
    state_tensor = env.readable2state(best_function)
    coordinates = proxy.apply_functions(state_tensor)

    # Convert coordinates to numpy for UMAP
    coords_np = coordinates.detach().cpu().numpy()  # e.g. shape (1, n_funcs, n_recordings) or similar
    try:
        import umap
    except Exception:
        raise ImportError("umap-learn is required for UMAP dimensionality reduction")
    reducer = umap.UMAP(n_components=2, random_state=42)

    # Remove possible batch dim
    if coords_np.ndim == 3 and coords_np.shape[0] == 1:
        coords_np = coords_np[0]  # now (n_funcs, n_recordings) or (n_recordings, n_funcs)
    elif coords_np.ndim == 3:
        # If multiple batches present, flatten batch and funcs into samples
        b, f, r = coords_np.shape
        coords_np = coords_np.reshape(b * f, r)

    # Prepare labels as numpy 1D array
    labels_arr = proxy.labels
    if isinstance(labels_arr, torch.Tensor):
        labels_arr = labels_arr.detach().cpu().numpy()
    labels_arr = np.asarray(labels_arr).reshape(-1)

    # Determine whether rows correspond to recordings or functions and align to labels
    # If number of rows matches labels -> good. Else if number of columns matches labels -> transpose.
    if coords_np.shape[0] == labels_arr.shape[0]:
        samples = coords_np  # rows are samples
    elif coords_np.shape[1] == labels_arr.shape[0]:
        samples = coords_np.T
    else:
        # Last resort: try flattening per-recording if possible (e.g. each recording expanded over funcs)
        # If total elements equal labels * k, reshape to (labels, -1)
        total = coords_np.size
        if labels_arr.shape[0] > 0 and total % labels_arr.shape[0] == 0:
            samples = coords_np.reshape(labels_arr.shape[0], -1)
        else:
            raise ValueError(f"Cannot align coordinates shape {coords_np.shape} with labels shape {labels_arr.shape}")

    # Run UMAP
    reduced_coords = reducer.fit_transform(samples)

    # Build boolean masks for indexing reduced_coords (length must match reduced_coords.shape[0])
    bad_mask = (labels_arr == 0)
    good_mask = (labels_arr == 1)
    if reduced_coords.shape[0] != bad_mask.shape[0]:
        # If masks still do not match, try to broadcast by repeating per-chunk if possible
        raise IndexError(f"Label length {bad_mask.shape[0]} does not match reduced samples {reduced_coords.shape[0]}")

    reduced_bad = reduced_coords[bad_mask]
    reduced_good = reduced_coords[good_mask]

    # Plot
    plt.figure(figsize=(10, 8))
    if reduced_bad.size:
        plt.scatter(reduced_bad[:, 0], reduced_bad[:, 1], c='red', label='Bad', alpha=0.7)
    if reduced_good.size:
        plt.scatter(reduced_good[:, 0], reduced_good[:, 1], c='green', label='Good', alpha=0.7)
    plt.title('Function Coordinates in UMAP Space')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("function_coordinates_umap.png")
    plt.close()
    print("Function coordinates UMAP plot saved.")





if __name__ == "__main__":
    sample_path = "/home/dmd_user/Desktop/ECAA/gflownet/samples/prod_data_y_test.csv"
    print("Plotting GFlowNet samples...")
    plot_gfn_samples_umap(sample_path, n_points=1000)
    plot_gfn_functions_3D(sample_path, n_points=1000)
    print("Plots saved.")
    print("Finding best function sample...")
    best_function, best_energy = find_best_function_sample(sample_path)
    print(f"Best function: {best_function} with energy: {best_energy}")

    


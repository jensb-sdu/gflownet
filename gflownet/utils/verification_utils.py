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





import torch
import scipy
import ast
from torch.utils.data import Dataset
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path



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
        row = self.labeled_forces.iloc[idx]
        # row['data'] may be stored as a dict inside the cell or as a pandas Series/object.
        data_cell = row['data']

        # Normalize to a dict-like structure and extract z_force
        forces = None
        if isinstance(data_cell, dict):
            forces = data_cell.get('z_force')
        elif isinstance(data_cell, (pd.Series, pd.DataFrame)):
            # try to access 'z_force' in the Series/DataFrame
            try:
                forces = data_cell['z_force']
            except Exception:
                # if the cell contains a single-element object/list with dict inside
                # try a couple of fallbacks
                try:
                    maybe = data_cell.iloc[0]
                except Exception:
                    maybe = data_cell
                if isinstance(maybe, dict):
                    forces = maybe.get('z_force')
                else:
                    raise KeyError("z_force")
        else:
            raise KeyError("z_force")

        if forces is None:
            raise KeyError("z_force")

        label = torch.tensor(row['label'], dtype =torch.int8)
        if self.transform:
            forces = self.transform(forces)
        else:
            forces = torch.tensor(forces, dtype=torch.float32)

        return forces, label
    
    def load_forces(self, directory):
        self.labeled_forces = get_full_dataframe_from_directory(directory, self.window_size)

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



def get_full_dataframe_from_directory(directory_path, window_size):
    """
    Parallelized version of get_fulldatafrane_from_directory.
    Processes all files in the directory and combines them into a single DataFrame.
    """
    # List all files in the directory

    files = [(file, window_size) for file in Path(directory_path).iterdir()]
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Map the process_file function to the list of files
        dataframes = list(executor.map(process_file, files))

    # Ensure all elements in dataframes are DataFrames
    for i, df in enumerate(dataframes):
        if not isinstance(df, pd.DataFrame):
            print(f"Error: Output of process_file for file {files[i]} is not a DataFrame. Got: {type(df)}")
            continue

    # Combine all DataFrames into a single DataFrame
    #full_dataframe = pd.concat(dataframes, ignore_index=False)
    return pd.concat(dataframes, ignore_index=True)


def process_file(file):
    """
    Process a single CSV file and return a DataFrame.
    This function reads the file, extracts forces, positions, and labels,
    and returns a DataFrame with the extracted data.
    """
    data = []
    label = None

    file_path = file[0]
    window_size = file[1]

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

    z_force = preprocess_force(z_force, window_size)
    

    file_df = pd.DataFrame({'data':[{'z_force':z_force,'timestamp':[i for i in range(len(forces))]}], 'label': label})
    return file_df




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


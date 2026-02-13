from gflownet.utils.verification_utils import ForceDisplacementDataset
from gflownet.envs.verification_env import VerificationEnv, FUNCTIONS
from gflownet.proxy.verification_proxy import VerificationProxy

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch

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



def plot_gfn_samples_umap(sample_path, save_path: Path, n_points=1000):
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

    # Sort from lowest to highest energy
    sorted_indices = np.argsort(energies)
    readable_list = [readable_list[i] for i in sorted_indices]
    energies = energies[sorted_indices]

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

    # Normalize energies for better visualization and cap outliers
    energies_min = np.percentile(energies_plot, 1)
    energies_max = np.percentile(energies_plot, 99)
    energies_range = energies_max - energies_min
    energies_range = energies_range if energies_range > 0 else 1
    energies_plot = (energies_plot - energies_min) / energies_range
    # Filter outliers for marking
    normalized_energies = (energies_plot >= 0) & (energies_plot <= 1)
    outliers = ~normalized_energies


    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_plot[normalized_energies, 0], emb_plot[normalized_energies, 1], c=energies_plot[normalized_energies], cmap='viridis', alpha=0.7)
    if np.any(outliers):
        plt.scatter(emb_plot[outliers, 0], emb_plot[outliers, 1], c='gray', marker='x', label='Outliers', alpha=0.5)
        plt.legend()
    plt.colorbar(scatter, label='Normalized Energy')
    plt.title('GFlowNet Samples in UMAP Space Colored by Energy')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / "gfn_samples_umap.png")
    plt.close()




def plot_gfn_functions_3D(sample_path, funcs, save_path: Path, n_points=1000):
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

    plt.savefig(save_path / "gfn_functions_3D.png")
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

def sort_by_energy(sample_path):
    readable_list, energies = read_gfn_samples(sample_path)
    sorting = np.argsort(energies)
    return [readable_list[i] for i in sorting], energies[sorting]

def display_function_over_curve(readable, energy, env, save_path: Path, number_of_curves = 10, fig_idx = None):

    state = env.readable2state(readable).tolist()


    # Visualize on a subset of sample force curves
    sample_forces = proxy.all_forces[:number_of_curves]
    sample_labels = proxy.labels[:number_of_curves]

    plt.figure(figsize=(10,5))
    ax = plt.gca()

    # Plot force curves with function overlays
    # collect numeric arrays to compute axis extents for annotation placement
    numeric_forces = []
    ymax = 0
    ymin = 0

    for i in range(len(sample_forces)):
        sample_force = sample_forces[i]
        sample_label = sample_labels[i]
        if torch.is_tensor(sample_force):
            arr = sample_force.detach().cpu().numpy()
        else:
            arr = np.asarray(sample_force)
        ymax = arr.max() if arr.max() > ymax else ymax
        ymin = arr.min() if arr.min() < ymin else ymin
            
        numeric_forces.append(arr)
        # Plot individual force curve color coded by label
        ax.plot(arr, color = 'green' if sample_label == 1 else 'red', alpha=0.3)

    # determine y placement for the dimension annotation
    
    y_offset = (ymax - ymin) / len(state)

    cmap = plt.get_cmap('viridis')
    # fallback to FUNCTIONS length (proxy not passed here)
    norm = plt.Normalize(vmin=0, vmax=len(state))

    for idx, func in enumerate(state):
        start, end, func_idx = func
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

    ax.set_title('GFlowNet Function Applied to Sample Force Curve\nScore: {:.6f}'.format(energy))
    ax.set_xlabel('Time')
    ax.set_ylabel('Force')
    ax.grid(True)
    if fig_idx is not None:
        plt.savefig(save_path / f"function_on_force_curve_{energy}_{fig_idx}.png")
    else:
        plt.savefig(save_path / f"function_on_force_curve_{energy}.png")
    plt.close()
    print("Best function visualization saved.")


def plot_function_coordinates_UMAP(function, energy, proxy, env, save_path : Path, fig_idx = None):
    #parse best function into state tensor
    state_tensor = env.readable2state(function)
    # Unsqueeze so that batch dim exists
    state_tensor = state_tensor.unsqueeze(0)
    coordinates = proxy.apply_functions(state_tensor)

    # Convert coordinates to numpy for UMAP
    coords_np = coordinates.detach().cpu().numpy()  # e.g. shape (1, n_funcs, n_recordings) or similar
    try:
        import umap
    except Exception:
        raise ImportError("umap-learn is required for UMAP dimensionality reduction")
    reducer = umap.UMAP(n_components=3, random_state=42)

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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    if reduced_good.size:
        ax.scatter(reduced_good[:, 0], reduced_good[:, 1], reduced_good[:, 2], c='green', label='Good', alpha=0.7)
    if reduced_bad.size:
        ax.scatter(reduced_bad[:, 0], reduced_bad[:, 1], reduced_bad[:, 2], c='red', label='Bad', alpha=0.7)
    plt.title(f'Function Coordinates in UMAP Space\nscore: {energy:.6f}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if fig_idx is not None:
        plt.savefig(save_path / f"function_coordinates_umap_{energy}_{fig_idx}.png")
    else:
        plt.savefig(save_path / f"function_coordinates_umap_{energy}.png")
    plt.close()
    print("Function coordinates UMAP plot saved.")

def plot_all_dimension_combinations(points, masks, labels=None):
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
        
        scatter2 = ax.scatter(x2, y2, alpha=0.7, c= 'green')
        # Plot red after green so its easier to see False positives
        scatter1 = ax.scatter(x1, y1, alpha=0.7, c = 'red')
        
        
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


def plot_gfn_dimension_combinations(function, energy, proxy, env, save_path : Path, n_points=1000, fig_idx = None):
    #parse best function into state tensor
    state_tensor = env.readable2state(function)
    # Unsqueeze so that batch dim exists
    state_tensor = state_tensor.unsqueeze(0)
    coordinates = proxy.apply_functions(state_tensor)

    # Get space separated dimension labels from readable state
    dimension_labels = function.split() 


    # Convert coordinates to numpy for UMAP
    coords_np = coordinates.detach().cpu().numpy()  # e.g. shape (1, n_funcs, n_recordings) or similar

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

    # Sample a subset for plotting if necessary
    if samples.shape[0] > n_points:
        indices = np.random.choice(samples.shape[0], n_points, replace=False)
        samples_plot = samples[indices]
        labels_plot = labels_arr[indices]
    else:
        samples_plot = samples
        labels_plot = labels_arr

    # Build boolean masks for indexing samples_plot
    bad_mask = (labels_plot == 0)
    good_mask = (labels_plot == 1)
    if samples_plot.shape[0] != bad_mask.shape[0]:
        # If masks still do not match, try to broadcast by repeating per-chunk if possible
        raise IndexError(f"Label length {bad_mask.shape[0]} does not match samples {samples_plot.shape[0]}")
    masks = [bad_mask, good_mask]
    fig, axes = plot_all_dimension_combinations(samples_plot, masks, labels=dimension_labels)
    plt.suptitle(f'Function Dimension Combinations\nscore: {energy:.6f}', fontsize=16)

    if fig_idx is not None:
        plt.savefig(save_path / f"function_dimension_combinations_{energy}_{fig_idx}.png")
    else:
        plt.savefig(save_path / f"function_dimension_combinations_{energy}.png")
    plt.close()
    print("Function dimension combinations plot saved.")


def plot_function_segments(function, energy, proxy, env, save_path : Path, fig_idx = None):
    #parse best function into state tensor
    state_tensor = env.readable2state(function)
    state_length = state_tensor.shape[0]
    colormap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=state_length)
    # Plot each segment in its own subplot
    num_segments = state_length 
    # calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    


    for idx, (start, end, func) in enumerate(state_tensor.tolist()):
        ax = axes[idx]
        func_obj = proxy.func_dict.get(func, None)
        if func_obj is None:
            continue
        color = colormap(norm(idx))
        
        # Plot full curve in the background for context
        for i in range(proxy.all_forces.shape[0]):
            full_curve = proxy.all_forces[i].detach().cpu().numpy()
            ax.plot(np.arange(len(full_curve)), full_curve, color='gray', alpha=0.1)

        # Extract and plot the segment of the force curve
        y = proxy.all_forces[:, start:end].detach().cpu().numpy().T
        x = np.linspace(start, end, y.shape[0])
        
        ax.plot(x, y, color=color, alpha=0.3)



        ax.set_title(f'{idx}: {func_obj.__name__ if callable(func_obj) else str(func_obj)}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Function Output')
        ax.grid(True)

    # Hide unused subplots
    for idx in range(num_segments, len(axes)):
        axes[idx].set_visible(False)    

    plt.suptitle(f'Function Segments Applied to Sample Force Curves\nscore: {energy:.6f}', fontsize=16)

    plt.tight_layout()

        
    plt.grid(True)
    if fig_idx is not None:
        plt.savefig(save_path / f"function_segments_{energy}_{fig_idx}.png")
    else:
        plt.savefig(save_path / f"function_segments.png")
    plt.close()


def debug(save_path: Path):
    colorvector = ["Red","Green","Black","Blue","purple","teal","Orange","Lightblue","magenta"]
    path = Path("/home/dmd_user/Desktop/ECAA/gflownet/assembly_case/MP 2 Development dataset 1.3/Data/NPF14/NPF14/Test/White MM & White CH")
    dataset = ForceDisplacementDataset(path, window_size=256)

    print("Dataset loaded\nPrinting the 10 first entries:")
    print(dataset[:10])

    # Check if all data entries have the same length
    lengths = [len(force) for force in dataset.labeled_forces['force']]


        
    if all(length == lengths[0] for length in lengths):
        print(f"All data entries have the same length: {lengths[0]}")
    else:
        print("Data entries have varying lengths.")
        unique_lengths = set(lengths)
        for length in unique_lengths:
            count = lengths.count(length)
            print(f"Length {length}: {count} entries")

    forces = dataset.labeled_forces['force']
    displacements = dataset.labeled_forces['displacement']
    labels = dataset.labeled_forces['label']


    #plot all force-displacement curves colored by label
    import matplotlib.pyplot as plt

    for i in range(len(forces)):
        label = labels[i]
        plt.plot(displacements[i], forces[i], label=f"Sample {i} - Label {label}", color=colorvector[label % len(colorvector)])  

    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.title("Force-Displacement Curves")
    plt.savefig(save_path / "force_displacement_curves.png")
    plt.close()
    print("Force-displacement curves plotted and saved as 'force_displacement_curves.png")                              

    # generate plot for good and bad samples
    for target_label in set(labels):
        plt.figure()
        for i in range(len(forces)):
            label = labels[i]
            if label == target_label:
                plt.plot(displacements[i], forces[i], label=f"Sample {i} - Label {label}", color=colorvector[label % len(colorvector)])  

        plt.xlabel("Displacement")
        plt.ylabel("Force")
        plt.title(f"Force-Displacement Curves for Label {target_label}")
        plt.savefig(save_path /  f"force_displacement_curves_label_{target_label}.png")
        plt.close()
        print(f"Force-displacement curves for label {target_label} plotted and saved as 'force_displacement_curves_label_{target_label}.png")



if __name__ == "__main__":
    case = "interval_func_norm_dunn_lr001"

    sample_path = f"/home/dmd_user/Desktop/ECAA/gflownet/samples/{case}.csv"
    save_path = Path(f"images/{case}")
    # ensure folder exists
    if not save_path.is_dir():
        save_path.mkdir()

    

    print("Plotting GFlowNet samples...")
    plot_gfn_samples_umap(sample_path, save_path, n_points=1000)
    plot_gfn_functions_3D(sample_path, funcs= FUNCTIONS, save_path=save_path, n_points=1000)
    print("Plots saved.")
    print("Finding best function sample...")
    best_function, best_energy = find_best_function_sample(sample_path)
    print(f"Best function: {best_function} with energy: {best_energy}")

    print("Generating environment and proxy for visualization...")
    #apply best function to verification env and visualize

    # Get environment params from config file
    env_config_path = Path("/home/dmd_user/Desktop/ECAA/gflownet/config/env/verification.yaml")
    with open(env_config_path, "r") as f:
        import yaml
        env_config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_config["device"] = device
    env = VerificationEnv(data_path=env_config["data_path"], 
                          device=env_config["device"], 
                          window_size=env_config["window_size"], 
                          resolution=env_config["resolution"],
                          min_function_width=env_config["min_function_width"], 
                          max_length=env_config["max_length"])
    
    proxy_config_path = Path("/home/dmd_user/Desktop/ECAA/gflownet/config/proxy/verification.yaml")
    with open(proxy_config_path, "r") as f:
        import yaml
        proxy_config = yaml.safe_load(f)


    proxy = VerificationProxy(reward_min=float(proxy_config["reward_min"]), 
                              do_clip_rewards=False, 
                              device=proxy_config["device"], 
                              production_data=bool(proxy_config["production_data"]))


    proxy.setup(env)

    
    
    
    reward = proxy(env.readable2state(best_function).unsqueeze(0))
    print(f"Reward of best function on training dataset: {reward}")
    #print("Displaying best function over sample force curves...")
    #display_best_function_over_curve(sample_path, proxy, env, save_path, number_of_curves = 10)

    print("Plotting best function coordinates in UMAP space...")
    #plot_best_function_coordinates_UMAP(best_function, proxy, env, save_path,)


    sorted_list, energies = sort_by_energy(sample_path)


    # print("First 10")
    # print([f"{sorted_list[i]}, {energies[i]}\n" for i in range(10)])
    
    print("Top 10")
    for i in range(1,10) :
        print("===================================")
        print(f"Processing function {i} of 10...")
        print(f"{energies[-i]}: {sorted_list[-i]}")
        plot_function_coordinates_UMAP(sorted_list[-i], energies[-i], proxy, env, save_path, fig_idx = i)
        display_function_over_curve(sorted_list[-i], energies[-i], env, save_path, number_of_curves = 10, fig_idx = i)
        plot_gfn_dimension_combinations(sorted_list[-i], energies[-i], proxy, env, save_path, n_points=1000, fig_idx = i)
        plot_function_segments(sorted_list[-i], energies[-i], proxy, env, save_path, fig_idx = i)
    # # generate new environment for validation dataset
    # validation_env = VerificationEnv(data_path="/home/dmd_user/Desktop/ECAA/gflownet/assembly_case/MP 2 Development dataset 1.3/Data/NPF14/NPF14/Test/White MM & White CH", device = "cuda" if torch.cuda.is_available() else "cpu", window_size=256, min_function_width=32, max_length=8)



    # proxy.setup(validation_env)

    # print("applying best function to validation dataset...")
    # reward = proxy.__call__(env.readable2state(best_function))
    # print(f"Reward of best function on validation dataset: {reward}")




    # debug(save_path)
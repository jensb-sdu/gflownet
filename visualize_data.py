
import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from gflownet.utils.verification_utils import ForceDisplacementDataset, ForceDataset

from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib
from pathlib import Path as _Path

def visualize_force_displacement_curves(dataset, save_path="force_displacement_curves.png"):
    """
    Visualize all force-displacement curves in the dataset.
    """
    plt.figure()
    curves = 0
    for i in range(len(dataset)):
        forces, label = dataset[i]
        if label == 0:
            plt.plot(forces, color='red')
            curves += 1

        if curves > 10:
            break
    curves = 0
    for i in range(len(dataset)):
        forces, label = dataset[i]
        if label == 1:
            plt.plot(forces, color='green')
            curves += 1
        if curves > 10:
            break


        # else:
        #     plt.plot(forces, color='red')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def _save_curve_worker(args):
    idx, forces, label, save_dir = args
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = _Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(forces, color='green' if int(label) == 1 else 'red')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.grid()
    plt.savefig(str(save_dir / f"force_displacement_curve_{idx}_label_{label}.png"))
    plt.close()

def save_individual_curves(dataset, save_dir, num_workers=None):
    """
    Save individual force-displacement curves as images in parallel.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i in range(len(dataset)):
        forces, label = dataset[i]
        forces = np.asarray(forces)  # ensure picklable (numpy array)
        tasks.append((i, forces, int(label), str(save_dir)))

    # Use default number of workers if none provided
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(_save_curve_worker, tasks)

def plot_clusters_from_DBSCAN(X, labels, save_path="dbscan_clusters.png"):
    """
    Plot clusters identified by DBSCAN. in each sub figure, plot all curves belonging to the same cluster.
    0. Noise points are labeled as -1 by DBSCAN.
    1. Each cluster is represented with a different color.
    2. Save the plot to the specified path.
    3. X is a 2D numpy array where each row is a data point.
    4. labels is a 1D numpy array of cluster labels corresponding to each data point in X.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_labels = set(labels)
    colormap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=len(unique_labels))

    # Plot each cluster


    for k in unique_labels:
        class_member_mask = (labels == k)
        class_members = X[class_member_mask]
        ax.plot(class_members, color=colormap(norm(k)), label=f'Cluster {k}' if k != -1 else 'Noise')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("DBSCAN Clustering")
    ax.legend()
    ax.grid()
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    data_path = pathlib.Path("/home/dmd_user/Desktop/ECAA/gflownet/assembly_case/MP 2 Development dataset 1.3/Data/NPF14/NPF14/Test/White MM & White CH")
    test_data_path = pathlib.Path("/home/dmd_user/Desktop/ECAA/gflownet/csv_data/csv_real_robot_sdu/csv_real_data")
    window_size = 2048
    dataset = ForceDisplacementDataset(data_path, AlignmentX=50, window_size=window_size)

    dbscan = DBSCAN(eps=0.5, min_samples=2)
    forces = []
    labels = []


    for i in range(len(dataset)):
        f, l = dataset[i]
        forces.append(np.asarray(f))
        labels.append(int(l))


    cluster_labels = dbscan.fit_predict(forces)

    # plot_clusters_from_DBSCAN(forces, cluster_labels, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/NPF14/dbscan_clusters.png")

    test_dataset = ForceDataset(test_data_path, window_size=window_size, resolution=128)

    visualize_force_displacement_curves(test_dataset, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/test_data/force_curves_test_dataset.png")
    save_individual_curves(test_dataset, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/test_data/individual_curves_test_dataset")


    # print("Visualizing force-displacement curves...")
    # visualize_force_displacement_curves(dataset, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/NPF14/force_displacement_curves.png")
    # save_individual_curves(dataset, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/NPF14/individual_curves")

    # print("Visualizing OOD force-displacement curves...")

    # OOD_data_path = pathlib.Path("/home/dmd_user/Desktop/ECAA/gflownet/assembly_case/MP 2 Development dataset 1.3/Data/NPF18/NPF18/Test/White MM & White CH")
    # OOD_dataset = ForceDisplacementDataset(OOD_data_path, AlignmentX=50, window_size=window_size)
    # visualize_force_displacement_curves(OOD_dataset, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/NPF18/force_displacement_curves.png")
    # save_individual_curves(OOD_dataset, "/home/dmd_user/Desktop/ECAA/gflownet/dataset_visualization/NPF18/individual_curves")

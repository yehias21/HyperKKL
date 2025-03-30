import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import linregress

def calculate_lyapunov(data, max_timestep=None, min_separation=10):
    """
    Calculate the largest Lyapunov exponent using Rosenstein's method.

    Parameters:
    - data: np.array of shape (e, n, d, t)
    - max_timestep: Maximum number of timesteps to consider in divergence (optional)
    - min_separation: Minimum temporal separation to avoid autocorrelation

    Returns:
    - lyapunov_exponents: np.array of calculated Lyapunov exponents
    """
    e_size, n_size, d_size, t_size = data.shape
    lyapunov_exponents = []

    for e_idx in range(e_size):
        print(f"Processing exogenous input {e_idx+1}/{e_size}")
        for n_idx in range(n_size):
            trajectory = data[e_idx, n_idx]  # Shape: (d, t)

            # Transpose trajectory to shape (t, d)
            trajectory = trajectory.T  # Now shape is (t, d)

            # Build KD-tree for nearest neighbor search
            tree = KDTree(trajectory)

            # Initialize arrays
            if max_timestep is None:
                max_timestep = t_size // 2
            divergences = []

            # For each point in trajectory, find nearest neighbor
            for i in range(t_size - max_timestep):
                # Exclude temporal neighbors within min_separation
                idxs = list(range(i - min_separation)) + list(range(i + min_separation, t_size))
                if not idxs:
                    continue

                # Find nearest neighbor
                distances, indices = tree.query(trajectory[i], k=2)
                nn_idx = indices[1] if indices[0] == i else indices[0]

                # Ensure temporal separation
                if abs(nn_idx - i) < min_separation:
                    continue

                # Calculate divergence over time
                divergence = []
                for k in range(max_timestep):
                    if i + k >= t_size or nn_idx + k >= t_size:
                        break
                    dist = np.linalg.norm(trajectory[i + k] - trajectory[nn_idx + k])
                    divergence.append(dist)

                if len(divergence) > 0:
                    divergences.append(np.log(divergence))

            # Calculate average divergence
            divergences = np.array(divergences)
            if divergences.size == 0:
                print(f"Insufficient data for trajectory {n_idx} under exogenous input {e_idx}")
                continue

            avg_divergence = np.mean(divergences, axis=0)

            # Linear fit to estimate Lyapunov exponent
            time = np.arange(len(avg_divergence))
            slope, intercept, r_value, p_value, std_err = linregress(time, avg_divergence)
            lyapunov_exponents.append(slope)

            # Plotting (optional)
            plt.figure()
            plt.plot(time, avg_divergence, label='Average divergence')
            plt.plot(time, slope * time + intercept, 'r--', label=f'Fit line (slope={slope:.4f})')
            plt.xlabel('Time steps')
            plt.ylabel('Average log divergence')
            plt.title(f'Lyapunov Exponent Estimation (e={e_idx}, n={n_idx})')
            plt.legend()
            plt.show()

    return np.array(lyapunov_exponents)

# Example usage
# Assuming `data` is your np.array of shape (e, n, d, t)
# data = np.random.rand(e_size, n_size, d_size, t_size)  # Replace with your actual data
# lyapunov_exponents = calculate_lyapunov(data)
# print("Calculated Lyapunov exponents:", lyapunov_exponents)

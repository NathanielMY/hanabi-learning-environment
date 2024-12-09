import pickle
import os
import matplotlib.pyplot as plt

def load_logged_data(log_dir, filename_prefix='log'):
    """Loads logged data from the experiment directory.
    
    Args:
        log_dir: str, path to the directory containing log files.
        filename_prefix: str, prefix used for log files (default is 'log').
    
    Returns:
        A dictionary of all logged statistics.
    """
    logged_data = {}
    for filename in sorted(os.listdir(log_dir)):
        if filename.startswith(filename_prefix):
            file_path = os.path.join(log_dir, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                logged_data.update(data)
    return logged_data

def plot_average_return(logged_data):
    """Plots average return per iteration.
    
    Args:
        logged_data: dict, dictionary containing logged experiment statistics.
    """
    iterations = []
    average_returns = []

    # Extract average returns from logged data
    for key, stats in logged_data.items():
        if 'average_return' in stats:
            iteration = int(key.replace('iter', ''))
            iterations.append(iteration)
            average_returns.append(stats['average_return'])

    # Sort iterations and corresponding returns (just in case)
    sorted_data = sorted(zip(iterations, average_returns))
    iterations, average_returns = zip(*sorted_data)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, average_returns, marker='o', linestyle='-', label='Average Return')
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Agent Performance: Average Return Per Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
log_dir = './experiment5/logs/'  # Replace with the path to your logs
logged_data = load_logged_data(log_dir)
plot_average_return(logged_data)

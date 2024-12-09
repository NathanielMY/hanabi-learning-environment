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

def extract_average_return(logged_data):
    """Extracts average return from logged IterationStatistics objects.

    Args:
        logged_data: dict, dictionary containing logged experiment statistics.

    Returns:
        A tuple (iterations, average_returns) for plotting.
    """
    iterations = []
    average_returns = []
    print(logged_data)

    for key, stats in logged_data.items():
        print(f"Key: {key}")
        if hasattr(stats, 'data_lists'):
            print("Logged data_lists:", stats.data_lists)
        else:
            print(f"No data_lists attribute in {key}")
        #print(stats)  # This will display the object’s string representation
        #print(dir(stats))
        # Check if the key corresponds to an iteration (e.g., 'iter0')
        if key.startswith('iter') and isinstance(stats, dict):
            iteration = int(key.replace('iter', ''))
            if 'average_return' in stats:
                iterations.append(iteration)
                average_returns.append(stats['average_return'])

    return iterations, average_returns

def plot_average_return(iterations, average_returns):
    """Plots average return per iteration.

    Args:
        iterations: list of ints, training iteration numbers.
        average_returns: list of floats, average returns for each iteration.
    """
    # Sort data to ensure proper plotting order
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
iterations, average_returns = extract_average_return(logged_data)
plot_average_return(iterations, average_returns)


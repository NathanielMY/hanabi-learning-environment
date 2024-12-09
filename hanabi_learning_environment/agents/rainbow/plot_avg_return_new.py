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
    """Extracts average training return from logged IterationStatistics objects.

    Args:
        logged_data: dict, dictionary containing logged experiment statistics.

    Returns:
        A tuple (iterations, average_returns) for plotting.
    """
    iterations = []
    average_returns = []

    for key, stats in logged_data.items():
        if hasattr(stats, 'data_lists') and 'average_training_return' in stats.data_lists:
            iteration = int(key.replace('iter', ''))
            iterations.append(iteration)
            # Extract the last value from average_training_return list
            average_returns.append(stats.data_lists['average_training_return'][-1])
        else:
            print(f"Skipped key {key}: stats not in expected format or missing 'average_training_return'")

    return iterations, average_returns


def plot_average_return(iterations, average_returns,  output_file='average_return.png'):
    """Plots average return per iteration.

    Args:
        iterations: list of ints, training iteration numbers.
        average_returns: list of floats, average returns for each iteration.
    """
    if not iterations or not average_returns:
        print("No valid data found for plotting.")
        return

    # Sort data to ensure proper plotting order
    sorted_data = sorted(zip(iterations, average_returns))
    iterations, average_returns = zip(*sorted_data)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, average_returns, marker='o', linestyle='-', label='Average Training Return')
    plt.xlabel('Iteration')
    plt.ylabel('Average Training Return')
    plt.title('Agent Performance: Average Training Return Per Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


# Example Usage
log_dir = './experiment5/logs/'  # Replace with the path to your logs
logged_data = load_logged_data(log_dir)
iterations, average_returns = extract_average_return(logged_data)
plot_average_return(iterations, average_returns)


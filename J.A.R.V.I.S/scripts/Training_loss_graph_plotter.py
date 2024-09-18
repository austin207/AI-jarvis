import matplotlib.pyplot as plt
import os

# Directory and file paths
logs_dir = r'C:\Users\mrult\ai_project\results\logs'
save_dir = r'C:\Users\mrult\ai_project\results\figures'
counter_file = os.path.join(save_dir, 'plot_counter.txt')
logs_file = os.path.join(logs_dir, 'training_log.txt')

# Ensure the directories exist
os.makedirs(save_dir, exist_ok=True)

def read_counter():
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as file:
            return int(file.read().strip())
    return 1

def write_counter(value):
    with open(counter_file, 'w') as file:
        file.write(str(value))

def generate_plots_from_logs():
    # Read all lines from the logs file
    with open(logs_file, 'r') as file:
        lines = file.readlines()

    # Process lines in chunks of 10 (one session)
    num_epochs_per_session = 5
    num_sessions = len(lines) // num_epochs_per_session
    plot_counter = read_counter()
    
    for i in range(num_sessions):
        # Extract the lines for this session
        session_lines = lines[i * num_epochs_per_session:(i + 1) * num_epochs_per_session]
        epochs = []
        loss = []

        for line in session_lines:
            epoch, value = line.strip().split()
            epochs.append(int(epoch))
            loss.append(float(value))

        # Generate filename
        filename = f'Training_Loss_Curve_{plot_counter}.png'
        filepath = os.path.join(save_dir, filename)

        # Plot and save
        plt.plot(epochs, loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve {plot_counter}')
        plt.savefig(filepath)
        plt.close()  # Close the plot to avoid overlapping

        # Increment and save the counter
        plot_counter += 1
        write_counter(plot_counter)

# Generate and save plots
generate_plots_from_logs()

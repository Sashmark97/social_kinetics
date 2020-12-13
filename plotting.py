import matplotlib.pyplot as plt
import numpy as np

def show_hist(input_array):
    hist_data = []
    for i in range(len(input_array)):
        for _ in range(input_array[i]):
            hist_data.append(i)

    plt.hist(hist_data, bins=100)
    return hist_data

def show_distribution(input_array, theoretical_vertice):
    
    hist_data = []
    for i in range(len(input_array)):
        for _ in range(input_array[i]):
            hist_data.append(i)

    plt.hist(hist_data, bins=100)
    return hist_data

def show_subplots(input_list, step, iterations_per_step):
    number_of_hists = len(input_list) // step
    nrows = number_of_hists // 4
    ncols = 4

    f,a = plt.subplots(nrows, ncols, figsize=(20, 10))
    a = a.ravel()
    for idx, ax in enumerate(a):
        if idx == 0: 
            additional = 1
        else:
            additional = 0
        hist_data = []

        for i in range(0, len(input_list[idx * step + additional])):
            for _ in range(input_list[idx * step + additional][i]):
                hist_data.append(i)

        ax.hist(hist_data, bins=100)
        ax.set_title(f'{idx * step * iterations_per_step + additional * iterations_per_step} Epoch')
        ax.set_xlabel('Money')
        ax.set_ylabel('People')
    plt.tight_layout()
    
def plot_loss_history(saved_losses, epochs_save):
    if saved_losses[0] == 0:
        saved_losses = saved_losses[1:]
        epochs_save = epochs_save[1:]
    plt.figure(figsize=(15, 5))
    plt.plot(epochs_save, saved_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
    
def plot_theory_vs_experiment(population, saved_c_vector):
    richest_person = np.max(np.where(saved_c_vector))
    distribution_experiment = saved_c_vector[:richest_person + 5] / population.N
    plt.figure(figsize=(15, 5))
    plt.plot(population.theoretical_vertice[:richest_person + 5], label='Theory')
    plt.scatter([i for i in range(richest_person + 5)], distribution_experiment, c='tab:orange', label='Experiment', alpha=0.8)
    plt.legend()
    plt.xlim([0, richest_person + 5])
    plt.grid()
    plt.show()
    
def plot_mixing_times(populations, epoch, log_scale=False):
    plt.figure(figsize=(7.5, 7.5))
    plt.title('Mixing time plot')
    plt.xlabel('Log(Number of people in population)')
    plt.ylabel('Convergence epochs')
    if log_scale:
        populations = np.log(populations)
    else:
        plt.xscale("linear")
    plt.plot(populations, epoch)
    
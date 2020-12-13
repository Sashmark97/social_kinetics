from social import Person, Population
from tqdm import tqdm
import time

def train_with_patience(population, patience, logging_interval, bet_size,
                        tax_percentage=None, tax_interval=None, initial_vector_save=False):
    
    if (tax_percentage is None and tax_interval is not None) or (tax_percentage is not None and tax_interval is None):
        raise Exception('Not found other taxes parameter')
        
    saved_vectors, saved_losses, epochs_save = [], [], []
    current_patience = patience
    current_loss = 1e12
    epoch_counter = 0
    start = time.time()
    while current_patience > 0:
        population.run_iteration(bet_size=bet_size)
        if initial_vector_save and epoch_counter == 0:
            population.update_c_vector()
            saved_vectors.append(population.c)
            saved_losses.append(population.compare_with_theory())
            epochs_save.append(epoch_counter)
        epoch_counter += 1
        if epoch_counter % logging_interval == 0:
            if epoch_counter - logging_interval == 0:
                print(f'{epoch_counter} epochs take {time.time() - start} seconds')
            population.update_c_vector()
            loss = population.compare_with_theory()
            print(f'Epoch: {epoch_counter:4}. Current loss: {loss:8.7}')
            saved_vectors.append(population.c)
            saved_losses.append(loss)
            epochs_save.append(epoch_counter)
            if current_loss > loss:
                current_loss = loss
                current_patience = patience
                print('Found new best loss!')
            else:
                current_patience -= 1
    return saved_vectors, saved_losses, epochs_save

def train_epochs(population, epochs, logging_interval, bet_size, whole_population=True,
                 tax_percentage=None, tax_interval=None, initial_vector_save=False):
    
    if (tax_percentage is None and tax_interval is not None) or (tax_percentage is not None and tax_interval is None):
        raise Exception('Not found other taxes parameter')

    saved_vectors, saved_losses, epochs_save = [], [], []
    current_loss = 1e12
    epoch_counter = 0
    start = time.time()
    with tqdm(total=epochs / logging_interval, position=0, leave=True) as bar:
        for iteration in range(epochs):
            if whole_population:
                population.run_iteration(bet_size=bet_size)
            else:
                population.run_one_game(bet_size=bet_size)

            if initial_vector_save and epoch_counter == 0:
                population.update_c_vector()
                saved_vectors.append(population.c)
                saved_losses.append(population.compare_with_theory())
                epochs_save.append(epoch_counter)
            epoch_counter += 1
            if epoch_counter % logging_interval == 0:
                if epoch_counter - logging_interval == 0:
                    print(f'{epoch_counter} epochs take {time.time() - start} seconds')
                population.update_c_vector()
                loss = population.compare_with_theory()
                bar.set_description(f"Epoch: {epoch_counter:4}. Current loss: {population.compare_with_theory():8.7}")
                bar.update()
                saved_vectors.append(population.c)
                saved_losses.append(loss)
                epochs_save.append(epoch_counter)
                if current_loss > loss:
                    current_loss = loss
                    print('Found new best loss!')
    return saved_vectors, saved_losses, epochs_save

def train_for_mixing_time(population_list, money_stratification, patience, logging_interval, bet_size,
                          tax_percentage=None, tax_interval=None, initial_vector_save=False,
                          loss_type='L2', quantize_theory=True):
    
    epochs_to_converge = []
    for num_of_people in population_list:
        population = Population(N=num_of_people, money_stratification=money_stratification,
                                loss=loss_type, quantize_theory=quantize_theory, multiprocessing_enable=False)
        print(f'=========|Starting training with {num_of_people} people|=========')
        vectors, losses, epochs = train_with_patience(population, patience, logging_interval, bet_size,
                                                      tax_percentage, tax_interval)
        epochs_to_converge.append((len(epochs) - patience) * logging_interval)
        print(f'=========|Finished after {epochs_to_converge[-1]} epochs|=========')
    return population_list, epochs_to_converge
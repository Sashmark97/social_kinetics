import random
import numpy as np

def quantize_vector(theory, people):
    step = 1 / people
    quantized = np.zeros(len(theory))
    for money in range(len(theory)):
        if quantized[money] == theory[money] % step >= step / 2:
            quantized[money] = theory[money] // step * step + step
        else:
            quantized[money] = theory[money] // step * step
        
    return quantized

def multiprocess_gamble(participants):
    player, opponent, bet_size = participants
    bet_pool = player.make_bet(bet_size) + opponent.make_bet(bet_size)
    if random.random() >= 0.5:
        player.recieve_money(bet_pool)
    else:
        opponent.recieve_money(bet_pool)
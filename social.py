import numpy as np
import random
from math import sqrt, floor
from multiprocessing.dummy import Pool as ThreadPool
from utils import quantize_vector, multiprocess_gamble

class Person():
    def __init__(self, money, id):
        self.money = money
        self.id = id

    def __repr__(self):
        return str(self.money)

    def __str__(self):
        return str(self.money)

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

    def get_money(self):
        return self.money

    def make_bet(self, bet_size):
        if self.money > 0 and self.money > bet_size:
            self.money -= bet_size
            return bet_size
        elif self.money > 0 and self.money <= bet_size:
            diff = self.money
            self.money = 0
            return diff
        else:
            return 0
    def pay_taxes(self, tax_percentage):
        pay = floor(self.money * tax_percentage)
        if self.money > 0 and self.money > pay:
            self.money -= pay
            return pay
        elif self.money > 0 and self.money <= pay:
            diff = self.money
            self.money = 0
            return diff
        else:
            return 0
        
    def recieve_money(self, money_recieved):
        self.money += money_recieved
        
class Population():
    def __init__(self, N, money_stratification, loss='L1', multiprocessing_enable=False, quantize_theory=True):
        self.multiprocessing_enable = multiprocessing_enable
        self.loss = loss
        self.quantize_theory = quantize_theory
        if N == 0:
            raise ValueError('Number of people should be non-zero')
        if N % 2 != 0:
            raise ValueError('Number of people should be an even number')

        # These values won't change ever
        self.N = N
        self.money_stratification = money_stratification
        self.money_ = 0

        # These values may change
        # In case where stratification is a number we calculate total money and initial wealth vector
        if isinstance(self.money_stratification, int):
            self.c = np.zeros(self.N * self.money_stratification + 1, dtype=np.int8)
            self.c[money_stratification] = N
        self.person_list = []

        # Initialize population
        # If stratification is a dict, we create wealth strates according to dict
        if isinstance(self.money_stratification, dict):
            sum_money = 0
            for money in self.money_stratification.keys():
                sum_money += self.money_stratification[money]
            self.c = np.zeros(self.N * sum_money + 1, dtype=np.int8)
            id = 0
            for money_ in self.money_stratification.keys():
                while self.money_stratification[money_] > 0:
                    self.person_list.append(Person(money_, str(id)))
                    self.money_stratification[money_] -= 1
                    self.c[money_] += 1
                    self.money_ += money_
                    id +=1
                    
        # If stratification is a number, we create population where each person have equal wealth
        elif isinstance(self.money_stratification, int):
            for id in range(self.N):
                self.person_list.append(Person(self.money_stratification, str(id)))
                self.money_ += self.money_stratification

        # Theoretical distribution of wealth
        self.theoretical_vertice = np.exp(- np.arange(self.money_ * self.N + 1) / (self.money_ / self.N)) / (self.money_ / self.N)
        if self.quantize_theory:
            self.theoretical_vertice = quantize_vector(self.theoretical_vertice, self.N)
    
    def update_c_vector(self):
        self.c = np.zeros(self.N * self.money_ + 1, dtype=np.int8)
        for person in self.person_list:
            self.c[person.get_money()] += 1

    def compare_with_theory(self):
        loss = 0.0
        eps = 1e-6
        for value in np.where(self.c > 0)[0]:
            if self.loss == 'L1':
                loss += abs(self.theoretical_vertice[value] - self.c[value] / self.N)
            elif self.loss == 'L2':
                loss += sqrt((self.theoretical_vertice[value] - self.c[value] / self.N) ** 2)
        return loss / (len(np.where(self.c > 0)[0]) + eps)
    """
    Previous version
    return np.linalg.norm(self.theoretical_vertice - self.c / self.N)
    """
    
    def collect_and_distribute_taxes(self, tax_percentage):
        tax_pool = 0
        for person in self.person_list:
            tax_pool += person.pay_taxes(tax_percentage)
        each_recieves = tax_pool // self.N
        leftovers = tax_pool - each_recieves * self.N
        for person in self.person_list:
            person.recieve_money(each_recieves)
        idx = 0
        while leftovers > 0:
            self.person_list[idx].recieve_money(1)
            leftovers -= 1
            
    def reset_population(self):
        self.person_list.clear()
        self.c = np.zeros(self.N * self.money_ + 1, dtype=np.int8)
        self.c[money_] = N
        for id in range(self.N):
            self.person_list.append(Person(money_, str(id)))

    def run_iteration(self, bet_size):
        players = random.sample(self.person_list, self.N // 2)
        opponents = [x for x in self.person_list if x not in players]
        random.shuffle(opponents)

        if self.multiprocessing_enable:
            bets = [bet_size] * len(players)
            pairs = list(zip(players, opponents, bets))
            pool = ThreadPool(10)
            results = pool.map(multiprocess_gamble, pairs)
            pool.close()
            pool.join()
        else:
            for player, opponent in zip(players, opponents):
                bet_pool = player.make_bet(bet_size) + opponent.make_bet(bet_size)
                if random.random() >= 0.5:
                    player.recieve_money(bet_pool)
                else:
                    opponent.recieve_money(bet_pool)

    def run_one_game(self, bet_size):
        player, opponent = random.sample(self.person_list, 2)
        bet_pool = player.make_bet(bet_size) + opponent.make_bet(bet_size)
        if random.random() >= 0.5:
            player.recieve_money(bet_pool)
        else:
            opponent.recieve_money(bet_pool) 
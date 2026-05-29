from parameters import *
import numpy as np
import random
from scipy.stats import poisson

class environment:
    def __init__(self):
        self.jammer_state = 0
        self.data_state = 0
        self.energy_state = 0

    def get_state(self):
        count=0
        state=0
        for jammer in range(0,2):
            for data in range(0,d_queue_size+1):
                for energy in range(0,e_queue_size+1):
                    if self.jammer_state == jammer and self.data_state == data and self.energy_state == energy:
                        state = count
                    count += 1
        return state

    def get_possible_action(self):
        list_actions = [0]
        if self.jammer_state == 0 and self.data_state > 0 and self.energy_state >= e_t:
            list_actions.append(1)
        if self.jammer_state == 1:
            list_actions.append(2)
            if self.data_state > 0:
                list_actions.append(3)
                if self.energy_state >= e_t:
                    list_actions.append(4)
                    list_actions.append(5)
                    list_actions.append(6)
        return list_actions

    def calculate_reward(self,action) :
        reward = 0
        loss = 0
        if action == 0 :
            reward = 0 
        elif action == 1 :
            reward = self.active_transmit(d_t)
        elif action == 2:
            reward = random.choices(e_hj_arr, nu_p, k = 1)[0]
        elif action == 3:
            d_bj = random.choices(d_bj_arr, nu_p, k = 1)[0]
            if self.data_state >= b_dagger:
                max_rate = b_dagger
            else:
                max_rate = self.data_state
            if self.data_state > d_bj:
                reward = d_bj
            else:
                reward = self.data_state
            if max_rate > reward:
                loss = max_rate - reward
        elif action == 4:
            max_ra = random.choices(dt_ra_arr, nu_p, k = 1)[0]
            reward = self.active_transmit(dt_ra_arr[0])

            if dt_ra_arr[0] > max_ra:
                loss = reward
                reward = 0
        elif action == 5:
            max_ra = random.choices(dt_ra_arr, nu_p, k = 1)[0]
            reward = self.active_transmit(dt_ra_arr[1])

            if dt_ra_arr[1] > max_ra:
                loss = reward
                reward = 0
        elif action == 6:
            reward = 0

        return reward, loss

    def active_transmit(self, maximum_transmit_packet):
        num_transmitted = 0
        if 0 < self.data_state < maximum_transmit_packet:
            if self.energy_state >= e_t * self.data_state:
                num_transmitted = self.data_state
            elif self.energy_state >= e_t:
                num_transmitted = self.energy_state // e_t
        else:
            if self.energy_state >= e_t * maximum_transmit_packet:
                num_transmitted = maximum_transmit_packet
            elif self.energy_state >= e_t:
                num_transmitted = self.energy_state // e_t
        return num_transmitted
            
    def perform_action(self, action):
        reward, loss = self.calculate_reward(action)
        if action == 1:
            self.data_state -= reward
            self.energy_state -= reward * e_t
        elif action == 2:
            self.energy_state += reward
            if self.energy_state > e_queue_size:
                self.energy_state = e_queue_size
            reward = 0
        elif action == 3:
            if self.data_state >= b_dagger:
                max_rate = b_dagger
            else:
                max_rate = self.data_state
            self.data_state -= max_rate
        elif action == 4 or action == 5:
            if reward > 0:
                self.data_state -= reward
                self.energy_state -= reward * e_t
            else:
                self.data_state -= loss
                self.energy_state -= loss * e_t
        data_arrive_1 = poisson.rvs(mu=arrival_rate, size = 1)
        data_arrive = data_arrive_1[0]
        self.data_state += data_arrive
        if self.data_state > d_queue_size:
            self.data_state = d_queue_size
        
        if self.jammer_state == 0:
            if np.random.random() <= 1 - nu:
                self.jammer_state = 1
        else:
            if np.random.random() <= nu:
                self.jammer_state = 0
        
        next_state = self.get_state()
        return reward, next_state

    def get_state_deep(self):
        state = [self.jammer_state, self.data_state, self.energy_state]
        return np.array(state)

    def perform_action_deep(self, action):
        reward, _ = self.perform_action(action)
        return reward, self.get_state_deep()

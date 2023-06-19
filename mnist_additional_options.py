#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:35:16 2023

@author: paularoth
"""

def ran_generator(length, shots=1):
    rand_list = random.sample(range(0, length), shots)
    return rand_list

# validation set #
num = ran_generator(len(test_anom_array), 10)
val_anom = [test_anom_array[i] for i in num]
num = ran_generator(len(test_normal_array), 10)
val_norm = [test_normal_array[j] for j in num]
val_set = [*val_norm, *val_anom]
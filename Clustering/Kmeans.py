import numpy as np

def initialize_centers():
    pass

def calc_membership():
    pass

def cost_function():
    pass

def update_centers():
    pass

def kmeans( X, c=2, ite=10 ):
    initialize_centers()
    for i in range(ite):
        calc_membership()
        cost_function()
        update_centers()

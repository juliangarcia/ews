# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from ews import *


def plot_abundance(matrix_aux, intensity_vector):
    #if number of rows in matrix_out is longer than
    intensity_vector = intensity_vector[0: matrix_aux.shape[0]]
    number_of_strategies = matrix_aux.shape[1]
    plt.xscale('log')
    for i in xrange(number_of_strategies):
        plt.plot(intensity_vector, matrix_aux[:, i], label="strategy {}".format(i+1))
    result = count_intersections(matrix_aux)
    most_abundant = count_most_abundant_changes(matrix_aux)
    #result_truncated = str(wsd_gen.count_intersections(game_matrix, population_size, left_limit, right_limit, num=wsd_gen.NUM, truncate=True))
    title = "Number of intersections:  {}, Changes in the most abundant strategy = {} ".format(result, most_abundant)
    plt.title(title)
    plt.legend(loc='best')

#!/usr/bin/python
# -*- coding: utf-8 -*-

#Dependencies
# - numpy 1.7.1
# - python 2.7.5

from math import e as e
from itertools import groupby
import numpy as np
from math import fsum
from operator import itemgetter
from itertools import combinations
from collections import Counter
import pickle
import sys

#epsilon for comparisons
EPSILON = 1e-3
#left-hand-side limit of the intensity of selection
LEFT_LIMIT = -3.0
#right-hand-side limit of the intensity of selection
RIGHT_LIMIT = 0.65
#number of point values in the intensity of selection interval
NUM = 100


def __auxiliary_function_for_exponential_mapping(a, b, c, d,
                                                 intensity_of_selection,
                                                 k_value, population_size):
    """
    This is an auxiliary function, used to compute fixation probabilities. It
    should not be called alone It implements the sum term in equation 20
    of  Traulsen,Shoresh, Nowak  2008
    """
    part_0 = k_value * (k_value + 1.0) * (intensity_of_selection / 2.0) * \
        (1.0 / (population_size - 1.0)) * (-a + b + c - d)
    part_1 = k_value * intensity_of_selection * \
        (1.0 / (population_size - 1.0)) * \
        (a - b * population_size + d * population_size - d)
    return e ** (part_0 + part_1)


def fixation_probability(game_matrix_2_x_2, intensity_of_selection,
                         population_size):
    """
    Computes the fixation probability of a mutant B in a pop of A's,
    where the game between A and B is given by a matrix 2x2 game.
    This corresponds to  equation 20 of Traulsen,Shoresh, Nowak  2008.

    Parameters:
    -----------
    game_matrix_2_x_2: ndarray
    intensity_of_selection: double
    population_size: int

    Returns:
    -------
    double: fixation probability

    """
    lista = [__auxiliary_function_for_exponential_mapping(
        game_matrix_2_x_2[1][1],
        game_matrix_2_x_2[1][0], game_matrix_2_x_2[0][1],
        game_matrix_2_x_2[0][0], intensity_of_selection, k, population_size)
        for k in xrange(0, population_size)]
    if any(np.isinf(lista)):
        return 0.0
    try:
        suma = fsum(lista)
    except OverflowError:
        return 0.0
    if np.isinf(suma):
        return 0.0
    return 1.0 / suma


def monomorphous_transition_matrix(game_matrix, population_size,
                                   intensity_of_selection):
    """
    Computes the associated markov chain (transition matrix),
    when mutations are assumed to be small. The approximation is accurate
    when there are no stable mixtures between any pair of strategies.

    Parameters
    ----------
    game_matrix: numpy matrix
    population_size: int
    intensity_of_selection: float
    kernel: ndarray, optional stochastic matrix

    Returns
    -------
    ans: ndarray, stochastic matrix

    """
    size = len(game_matrix)
    ans = np.zeros((size, size))
    for i in xrange(0, size):
        for j in xrange(0, size):
            if i != j:
                sub_game = np.array([[game_matrix[i][i], game_matrix[i][j]],
                                    [game_matrix[j][i], game_matrix[j][j]]])
                fix = fixation_probability(sub_game, intensity_of_selection,
                                           population_size)
                ans[i, j] = 1.0 / (size - 1.0) * fix

    for i in range(0, size):
        ans[i, i] = 1.0 - fsum(ans[i, :])
    return ans


def stationary_distribution(transition_matrix_markov_chain):
    '''
    Computes the stationary_distribution of a markov chain.
    The matrix is given by rows.

    Parameters
    ----------
    transition_matrix_markov_chain: ndarray (must be a numpy array)

    Returns
    -------
    out: ndarray

    Examples
    -------
    >>>stationary_distribution(np.array([[0.1,0.9],[0.9,0.1]]))
    Out[1]: array([ 0.5,  0.5])
    >>>stationary_distribution(np.array([[0.1,0.0],[0.9,0.1]]))
    Out[1]: array([ 1.,  0.])
    >>>stationary_distribution(np.array([[0.6,0.4],[0.2,0.8]]))
    Out[1]: array([ 0.33333333,  0.66666667])
    '''
    transition_matrix_markov_chain = transition_matrix_markov_chain.T
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix_markov_chain)
    # builds a dictionary with position, eigenvalue
    # and retrieves from this, the index of the largest eigenvalue
    index = max(
        zip(range(0, len(eigenvalues)), eigenvalues), key=itemgetter(1))[0]
    # returns the normalized vector corresponding to the
    # index of the largest eigenvalue
    # and gets rid of potential complex values
    result = np.real(eigenvectors[:, index])
    result /= np.sum(result, dtype=float)
    return result


def random_game(number_of_strategies, distribution_type):
    """
    Get me a random game, 0 uniformly; or 1, normally distributed
    """
    sample_size = number_of_strategies ** 2.0
    if distribution_type == 0:
        return (np.random.random_sample(sample_size)).reshape((
            number_of_strategies, number_of_strategies))
    if distribution_type == 1:
        return (np.random.normal(size=sample_size)).reshape((
            number_of_strategies, number_of_strategies))
    return None


def get_curves_truncating(intensity_vector, game_matrix, population_size,
                          truncate=True):
    """
    Computes abundances across a range of intensity of selection values,
    automatically adjusting the maximum intensity of selection to avoid
    numerical errors.

    Parameters
    ----------
    intensity_vector:
    game_matrix
    population_size

    Returns
    --------
    A matrix with as many columns as strategies, and as many rows as
    intensities of selection.

    """
    number_of_strategies = game_matrix.shape[0]
    matrix_aux = np.empty(number_of_strategies)
    for ios in intensity_vector:
        stationary = stationary_distribution(
            monomorphous_transition_matrix(game_matrix, population_size, ios))
        if (truncate and not is_stationary(stationary)):
            break
        matrix_aux = np.vstack((matrix_aux, stationary))
    matrix_aux = np.delete(matrix_aux, (0), axis=0)
    return matrix_aux


def is_stationary(stationary_vector):
    """
    Checks if a vector has all components between 0.0 and 1.0 (within epsilon).
    """
    for i in stationary_vector:
        if i < 0.0 - EPSILON or i > 1.0 - EPSILON:
            return False
    return True


def zero_crossings_where(x):
    """
    Returns a list of positions where x changes sign
    """
    return np.where(np.diff(np.sign(x)))[0]


def count_intersections(matrix_aux):
    """
    This function counts the number of intersections, given a matrix
    with as many columns as strategies, and as many rows as value points
    for selection intensity.
    """
    # turn matrix into a list of n primitive (abundance) curves
    number_of_strategies = matrix_aux.shape[1]
    primitive_curves = []
    for i in xrange(0, number_of_strategies):
        primitive_curves.append(matrix_aux[:, i])
    pairwise_intersections_dict = dict()
    all_possible_combinations = tuple(
        combinations(range(number_of_strategies), 2))
    for i in all_possible_combinations:
        specific_crossings = zero_crossings_where(
            primitive_curves[i[0]] - primitive_curves[i[1]])
        pairwise_intersections_dict[i] = specific_crossings
    ans = 0
    for i in pairwise_intersections_dict.keys():
        ans = ans + len(pairwise_intersections_dict[i])
    return ans


def count_most_abundant_changes(matrix_aux):
    """
    Counts the number of times the most abundant strategy changes, given a matrix
    with as many columns as strategies, and as many rows as value points
    for selection intensity.
    """
    #we build a list with the index of the most abundant strategy
    #for every intensity value
    most_abundant_list = [np.argmax(matrix_aux[i])
                          for i in xrange(0, matrix_aux.shape[0])]
    #summarize the list by eliminating repeated indexes that are adjacent
    most_abundant_changes = [k for k, g in groupby(most_abundant_list)]
    #the number of most abundant changes is the size of the
    #summary list minus 1.
    return len(most_abundant_changes) - 1


def estimate(seed, reps=100, population_size=30, number_of_strategies=3,
             save_files=True, distribution_type=0, num=NUM,
             left_limit=LEFT_LIMIT, right_limit=RIGHT_LIMIT):
    """
    This method estimates the number of crossings by
    sampling reps times, games from a given distribution.

    The counts are stored in a dictionary that has the number of
    changes as keys, and the number of occurrences of such
    changes as values.

    These dictionaries are pickled to the hard disk.
    """
    # first seed the random number generator
    np.random.seed(seed)
    estimations_list = []
    most_abundant_changes_list = []
    for _ in xrange(0, reps):
        game = random_game(number_of_strategies, distribution_type)
        intensity_vector = np.logspace(left_limit, right_limit, num,
                                       endpoint=True)
        matrix_aux = get_curves_truncating(intensity_vector, game,
                                           population_size,
                                           truncate=True)
        result = count_intersections(matrix_aux)
        most_abundant = count_most_abundant_changes(matrix_aux)
        estimations_list.append(result)
        most_abundant_changes_list.append(most_abundant)
    count = Counter(estimations_list)
    count_most_abundant = Counter(most_abundant_changes_list)
    if save_files:
        intersections_file_name = 'n_{}_dist_{}_seed_{}_count.pickle'.format(
            str(number_of_strategies), str(distribution_type), str(seed))
        most_abundant_file_name = 'ma_n_{}_dist_{}_seed_{}_count.pickle'.format(
            str(number_of_strategies), str(distribution_type), str(seed))
        pickle.dump(count, open(intersections_file_name, "wb"))
        pickle.dump(count_most_abundant, open(most_abundant_file_name, "wb"))
        print "Files {} and {} have been created".format(
            intersections_file_name, most_abundant_file_name)
    return (count, count_most_abundant)


if __name__ == '__main__':
    # we first check that at least three arguments are passed
    assert len(sys.argv) == 5, "You must specify the SEED, the NUMBER_OF_STRATEGIES " \
                               "and the NUMBER_OF_REPETITIONS, and the DISTRIBUTION_TYPE "\
                               "0 -- UNIFORM, 1 -- NORMAL"
    # parse command line arguments
    seed = int(sys.argv[1])
    number_of_strategies_ = int(sys.argv[2])
    reps = int(sys.argv[3])
    distribution_type = int(sys.argv[4])
    # check that  distribution is either 1 or 0
    assert distribution_type == 0 or distribution_type == 1,  "Distribution type should be 0 for UNIFORM, or 1 for NORMAL"
    population_size = 30
    estimate(seed, reps, population_size, number_of_strategies_, distribution_type=distribution_type)

# -*- coding: utf-8 -*-

#Dependencies
# - numpy 1.7.1
# - python 2.7.5

import unittest
from ews import *
import numpy as np
import time

#auxiliary function for seeding
current_milli_time = lambda: int(round(time.time() * 1000))


class Test(unittest.TestCase):

    pd_game = np.array([[3.0, 0.0], [4.0, 1.0]])

    def testFixationProbability(self):
        fix = fixation_probability(self.pd_game, 1.0, 10)
        msg = 'Failed non-trivial fixation probability test'
        self.assertTrue(np.allclose(0.7364040545619178, fix), msg)

    def testNeutralFixationProbability(self):
        fix = fixation_probability(self.pd_game, 0.0, 10)
        msg = 'Failed neutral fixation probability test'
        self.assertTrue(np.allclose(1.0/10.0, fix), msg)

    def testTransitionMatrix(self):
        transition = monomorphous_transition_matrix(self.pd_game,
                                                    population_size=10,
                                                    intensity_of_selection=20.0)
        #for strong selection in a PD game should be
        target = np.array([[0.0, 1.0], [0.0, 1.0]])
        msg = 'Failed transition matrix test for strong selection'
        self.assertTrue(np.allclose(target, transition), msg)

    def testIsStationary(self):
        stationary_vector = [0.4, 0.7]
        non_stationary_vector = [1.0, 1.0]
        self.assertTrue(is_stationary(stationary_vector))
        self.assertFalse(is_stationary(non_stationary_vector))

    def testGetCurvesTruncating(self):
        intensity_vector = [0.0, 20.0]
        target = np.array([[0.5, 0.5], [0.0, 1.0]])
        #First without truncation
        matrix = get_curves_truncating(intensity_vector, self.pd_game,
                                       population_size=30,
                                       truncate=False)
        msg = 'Failed auxiliary test matrix'
        self.assertTrue(np.allclose(target, matrix), msg)
        #Now truncating - just first row of target
        matrix = get_curves_truncating(intensity_vector, self.pd_game,
                                       population_size=30,
                                       truncate=True)
        self.assertTrue(np.allclose(target[0], matrix), msg)

    def testCountIntersections(self):
        #a quadratic function and a linear function
        #intersect twice in the interval -2, 2
        x = np.linspace(-2.0, 2.0, 100)
        auxiliary_matrix = np.empty((len(x), 2))
        for i, x_value in enumerate(x):
            auxiliary_matrix[i, 0] = x_value
            auxiliary_matrix[i, 1] = x_value**2.0
        self.assertEquals(2, count_intersections(auxiliary_matrix))

    def testCountMostAbundantChanges(self):
        #a quadratic function and a linear function
        #intersect twice in the interval -2, 2
        #twice are the changes in most  abundant
        x = np.linspace(-2.0, 2.0, 100)
        auxiliary_matrix = np.empty((len(x), 2))
        for i, x_value in enumerate(x):
            auxiliary_matrix[i, 0] = x_value
            auxiliary_matrix[i, 1] = x_value**2.0
        self.assertEquals(2, count_most_abundant_changes(auxiliary_matrix))

    def testEstimate(self):
        #no intesections in two player games
        reps = 50
        (count, count_most_abundant) = estimate(current_milli_time(),
                                                reps=reps,
                                                population_size=10,
                                                number_of_strategies=2,
                                                save_files=False,
                                                distribution_type=0)
        #all repetitions report 0 intersections
        #for games with two players
        self.assertEquals(count[0], reps)
        self.assertEquals(count_most_abundant[0], reps)


if __name__ == "__main__":
    unittest.main()

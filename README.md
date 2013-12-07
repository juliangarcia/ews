## Introduction

This software performs the simulations for the paper: **Extrapolating weak selection in evolutionary games**, by *Bin Wu*, *Julian Garcia*, *Christoph Hauert* and *Arne Traulsen*. [Journal link.](dx.doi.org/10.1371/journal.pcbi.1003381)

A good place to start is the included IPython notebook [viewable on-line here](http://nbviewer.ipython.org/urls/raw.github.com/juliangarcia/ews/master/notebook.ipynb)

## Included files

- *ews.py*: estimates the number of rank changes and changes of the most abundant strategy in the stationary distribution associated to an imitation process in random games of a given size and distribution. It samples, counts and writes the results to the hard disk. Please see the paper for technical details and read further for computational details. 

- *visualization.py*: Contains a function to plot abundance curves (using matplotlib).

- *tests.py*: contains unit tests

- *notebook.ipynb*: is an interactive IPython notebook that describes the details of the estimation procedure.

- *LICENSE*: MIT License

- *README.MD*: This file 


## Dependencies

**ews.py** was tested using *numpy* version 1.7.1 and python 2.7.5.

**notebook.ipynb** was created with *IPython* version 1.0.

**visualization.py** uses *matplotlib* version 1.3

## Running the software
The program should be invoked as:

```bash 
python ews.py SEED NUMBER_OF_STRATEGIES NUMBER_OF_REPETITIONS DISTRIBUTION_TYPE
```

*DISTRIBUTION_TYPE* is 0 for uniformly distributed games, or 1 for sampling from a Gaussian distribution. 

For example

```bash 
python ews.py 8902 5 1000 0
```

will generate 1000 uniformly distributed random games of size 5, and count the number of rank changes and changes in the most abundant strategy for every game. 

These counts are stored in a dictionary that has the number of changes as keys, and the number of occurrences of such changes as values. In this particular case, the two dictionaries are stored in the hard disk in files *n_5_dist_0_seed_8902_count.pickle* for changes in raking, and *ma_n_5_dist_0_seed_8902_count.pickle* for changes in the most abundant strategy. 


## Tests

For unit tests please run 

```bash
python tests.py 
```

## License

See LICENSE for details. 

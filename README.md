This repository contains python code to analyze dynamical properties of ensembles of nowcasts from radar precipitation data. 

The repository is organized in the following way:
 - src/dynnow/ensemble contains the definition of the main class: the 'Ensemble' class. An example of the way to use it is in doc/example.ipynb
 - src/dynnow/surrogates.py contains the functions to generate the surrogate ensembles
 - src/dynnow/analysis.py contains some functions to automatically generate the surrogates and compute the scores of the ensembles
 - src/dynnow/utils.py contains some functions to produce the nowcasts with the models of interest
 - src/dynnow/plots.py contains the functions to reproduce the plots in the paper

All computations are lazily done first and then handled by dask when '.compute()' is called.

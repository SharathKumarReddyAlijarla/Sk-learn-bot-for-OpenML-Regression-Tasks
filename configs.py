# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 02:00:37 2023

@author: askr1
"""

from ConfigSpace import *
from ConfigSpace import ConfigurationSpace
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import argparse
import sklearn





def config(reg_name):
   
    if reg_name == 'decision_tree':
        cs = ConfigurationSpace()

        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "friedman_mse", "mae"]
         )
        max_features = Constant("max_features", 1.0)
        max_depth_factor = UniformFloatHyperparameter(
            "max_depth_factor", 0.0, 2.0, default_value=0.5
        )
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        cs.add_hyperparameters(
             [
                 criterion,
                 max_features,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 min_impurity_decrease,
             ]
         )

        clf= sklearn.tree.DecisionTreeRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)
        
    if reg_name == 'random_forest':
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "friedman_mse", "mae"]
        )

        # In contrast to the random forest classifier we want to use more max_features
        # and therefore have this not on a sqrt scale
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0
        )

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )
        bootstrap = CategoricalHyperparameter(
            "bootstrap", [True, False], default_value=True
        )

        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                #max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                #max_leaf_nodes,
                min_impurity_decrease,
                bootstrap,
            ]
        )  
        clf= sklearn.ensemble.RandomForestRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)

    if reg_name == 'k_nearest_neighbors': 
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform"
        )
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        
        cs.add_hyperparameters([n_neighbors, weights, p])
        clf=sklearn.neighbors.KNeighborsRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)
    return(clf)
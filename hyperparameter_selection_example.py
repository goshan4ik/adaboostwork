import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from hyperparameter_selection import HyperparameterGrid

MIN_N_ESTIMATORS = 10
MAX_N_ESTIMATORS = 510
STEP = 10

cancer = datasets.load_breast_cancer()


def draw_plot(hyperparam: [int], score: [float]):
    plt.plot(hyperparam, score)
    plt.ylabel('score')
    plt.xlabel('hyperparameter value')
    plt.show()


grid = HyperparameterGrid(MIN_N_ESTIMATORS, MAX_N_ESTIMATORS, STEP)
best_score, best_param_value = grid.research(DecisionTreeClassifier(max_depth=5))

print(best_score, best_param_value)

draw_plot(grid.all_params, grid.all_scores)

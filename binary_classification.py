from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperparameter_selection import HyperparameterGrid

MIN_N_ESTIMATORS = 10
MAX_N_ESTIMATORS = 510
STEP = 10

cancer = datasets.load_breast_cancer()

weak_classifiers = {
    'Random Forest classifier': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'Decision Tree Classifier': DecisionTreeClassifier(max_depth=5),
}

for name, classifier in weak_classifiers.items():
    print('The score for %s is %s' % (name, cross_val_score(classifier, cancer.data, cancer.target).mean()))

boosted_classifiers = {
    'Boosted Random Forest classifier': AdaBoostClassifier(RandomForestClassifier(max_depth=5, n_estimators=10,
                                                                                  max_features=1)),
    'Boosted Decision Tree Classifier': AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),
}

for name, classifier in boosted_classifiers.items():
    grid = HyperparameterGrid(MIN_N_ESTIMATORS, MAX_N_ESTIMATORS, STEP)
    grid.research(classifier, cancer)
    print('The score for %s is %s when hyperparameter is %s' % (name, grid.best_score, grid.best_score_parameter))

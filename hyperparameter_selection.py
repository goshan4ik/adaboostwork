from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


class HyperparameterGrid:

    __best_score = 0
    __best_param_value = 0
    __params = []
    __scores = []

    def __init__(self, left_bound, right_bound, step) -> None:
        self.__left_bound = left_bound
        self.__right_bound = right_bound
        self.__step = step

    @property
    def best_score(self):
        return self.__best_score

    @property
    def best_score_parameter(self):
        return self.__best_param_value

    @property
    def all_scores(self):
        return self.__best_score

    @property
    def all_params(self):
        return self.__best_param_value

    def research(self, classifier, dataset):
        param_value = self.__left_bound
        best_score = 0
        self.__best_param_value = self.__left_bound
        while param_value <= self.__right_bound:
            score = cross_val_score(AdaBoostClassifier(classifier, algorithm="SAMME.R", n_estimators=param_value),
                                    dataset.data, dataset.target).mean()
            if score > best_score:
                best_score = score
                self.__best_param_value = param_value
            param_value += self.__step
            self.__scores.append(score)
            self.__params.append(param_value)
        return self.__params, self.__scores

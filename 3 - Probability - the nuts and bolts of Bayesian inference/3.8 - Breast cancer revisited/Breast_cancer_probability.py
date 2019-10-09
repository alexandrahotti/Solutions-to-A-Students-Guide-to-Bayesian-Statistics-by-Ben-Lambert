
class Breast_cancer_probability:

    def __init__(self):
        self.__positive_given_cancer = 0.9
        self.__positive_given_not_cancer = 0.08
        self.__probability_cancer_rare = 0.01
        self.__probability_not_cancer_rare = 0.99
        self.__probability_cancer_common = 0.1
        self.__probability_not_cancer_common = 0.8

    def get_positive_given_cancer(self):
        return self.__positive_given_cancer

    def get_positive_given_not_cancer(self):
        return self.__positive_given_not_cancer

    def get_probability_cancer_rare(self):
        return self.__probability_cancer_rare

    def get_probability_not_cancer_rare(self):
        return self.__probability_not_cancer_rare


    def get_probability_cancer_common(self):
        return self.__probability_cancer_common

    def get_probability_not_cancer_common(self):
        return self.__probability_not_cancer_common


    def set_positive_given_cancer(self, value):
        self.__positive_given_cancer = value

    def set_positive_given_not_cancer(self, value):
        self.__positive_given_not_cancer = value

    def set_probability_cancer(self, value):
        self.__probability_cancer = value

    def set_probability_not_cancer(self, value):
        self.__probability_not_cancer = value

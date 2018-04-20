import random
from classifier.ensemble import ExtraTreesClassifier
from classifier.k_fold_validator import cross_validate

def main() :

    '''
    This is for testing the ExtraTreeClassifier using 100 data of two feature
    - x1 : random integer
    - x2 : random integer
    and label :
    - label : k1, k2

    Then predict 50 data using the model
    - test : dataset of testing for predict
    '''

    ensemble = ExtraTreesClassifier(5, 5)

    n = 100
    data = {"x1": [], "x2": []}
    label = []
    for i in range(50):
        label.append("k1")
        data["x1"].append(random.randint(50, 100))
        data["x2"].append(random.randint(1,10))
    for i in range(50):
        label.append("k2")
        data["x1"].append(random.randint(1, 40))
        data["x2"].append(random.randint(20,40))

    print(data)
    print("\ny :\n")
    print(label)

    ensemble.train(data, label)

    test = {"x1": [], "x2": []}
    for i in range(50):
        test["x1"].append(random.randint(1, 20))
        test["x2"].append(random.randint(1,40))

    print("\nprediction: \n")
    print(ensemble.predict(test))

    '''
    For cross validation
    Input : 100 data of two feature
    - x1 : random integer
    - x2 : random integer
    and label :
    - label : k1, k2

    '''
    
    train_score, prediction_score = cross_validate(ensemble, data, label, 10)
    print(train_score, prediction_score)

if __name__ == '__main__':
    main()
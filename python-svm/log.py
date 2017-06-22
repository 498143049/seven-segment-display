import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def getData(url):
    file = open(url,"r")
    list_arr = file.readlines()
    lists = []
    for index, x in enumerate(list_arr):
        x = x.strip()
        x = x.strip('[]')
        x = x.split(", ")
        lists.append(x)
    a = np.array(lists)
    a = a.astype(float)
    file.close()
    return a;

if __name__ == "__main__":
    x_n=getData('outn.txt');
    y_n=np.zeros(len(x_n));
    x_p = getData('outp.txt');
    y_p = np.ones(len(x_p));
    x = np.concatenate((x_n,x_p))
    y = np.concatenate((y_n,y_p))
    expected = y
    # y = y_n + y_p

    classifier = LogisticRegression('l1',C=0.1)
    classifier.fit(x, y)
    # print(classifier)
    x_t = getData('outtest.txt');
    # predicted = classifier.predict(x)
    # print(clf.fit(x, y))
    predicted = classifier.predict(x_t)
    predictedp = classifier.predict_proba(x_t)
    print(predicted)
    print(predictedp)
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))
print("ok!")
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import  OneClassSVM
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
    print(x.shape,y.shape)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    clf=SVC(C=1, gamma=2.811, kernel='rbf', probability=1,random_state = 0)
    # scores = cross_val_score(clf, x, y, cv=5)
    # print(scores)
    # y = y_n + y_p
    # clf = SVC(C=3,gamma=3,kernel='rbf',class_weight='balanced' ,probability=1)
    clf.fit(x, y)
    # scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
    # print(clf)
    # x_t = getData('outt.txt');
    # # print(clf.fit(x, y))
    predicted = clf.predict(x)
    predictedx = clf.predict_proba(x)
    # predictedX = clf.predict(x_t)
    # print(predictedX)
    for letter in predictedx:
        print(letter[0]/letter[1])
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    # clf.svm_save_model("s");
print("ok!")
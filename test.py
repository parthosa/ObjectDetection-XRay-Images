from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score

def sq_norm(mat):
    normM = mat - mat.min()
    normM = np.sqrt(normM)
    normM = normM / normM.max()
    mat_sq_norm = normM
    return mat_sq_norm

x = loadmat('../matlab-outputs/knn/with-others/trainFeaturesData.mat')
y = loadmat('../matlab-outputs/knn/with-others/k-3/testFeaturesData.mat')
# y = np.load('../pred.npy')
# z = np.load('./pred (1).npy')

train_features = x['trainingFeaturesNorm']
# train_features_norm_mat = x['trainingFeaturesNorm']
# train_features = np.load('../test_features.npy')

test_features = y['testFeaturesNorm']
# print y
# print('Matlab')
# print(train_features_mat)
# print('Python')
# print(train_features)

# print('Matlab - Norm')
# print(train_features_norm_mat)
# print('Python - Norm')
# print(sq_norm(train_features))

# # x = np.load('/model/pred.npy')
# # print sq_norm(y)[0]
# # print sq_norm(z)[0]
# # print x['trainingFeaturesNorm'][0]

# # x = np.array([[1,4,2],[4,9,6]])
# # print sq_norm(x)

# train_features = sq_norm(np.load('../train_features.npy'))
train_labels = np.load('../train_labels.npy')
# test_features = sq_norm(np.load('./test_features.npy'))
test_labels = np.load('../test_labels.npy')
target_names = ['blade','gun','others','shuriken']
print(train_features[0])
try:
    svm = LinearSVC(random_state=0)
    svm.fit(train_features, train_labels)

    pred = svm.predict(test_features)
    np.save('../output/pred_svm',pred)
    # pred = np.load('/model2/pred.npy')
    print('SVM')
    print(accuracy_score(test_labels, pred))
    print(classification_report(test_labels, pred,target_names=target_names))
    for i in range(1,9,2):
        print("k=%d" %(i))
        knn = KNeighborsClassifier(n_neighbors=i)
        # print('train',train_features.shape,train_labels.shape)
        # print('test',test_features.shape,test_labels.shape)
        # train_features = train_features.reshape(train_features.shape[0],-1)
        knn.fit(train_features, train_labels)

        # test_features = test_features.reshape(test_features.shape[0],-1)
        pred = knn.predict(test_features)
        np.save('../output/pred_%d' %(i),pred)
        # pred = np.load('/model2/pred.npy')
        print(accuracy_score(test_labels, pred))
        print(classification_report(test_labels, pred,target_names=target_names))
except Exception as e: print(e)

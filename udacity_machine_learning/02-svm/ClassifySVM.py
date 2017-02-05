def classify(features_train, labels_train):   
    ### import the sklearn module for SVM
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    ### your code goes here!
    
    #引入函式庫
    from sklearn.svm import SVC
    #選擇kernel 初始化分類器
    clf = SVC(kernel="linear")

	#給定訓練資料  特徵 與 標籤
    clf.fit(features_train, labels_train)
	
	#回傳分類器
    return clf
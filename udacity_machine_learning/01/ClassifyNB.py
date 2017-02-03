def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
	#給定訓練資料  特徵 與 標籤
    clf.fit(features_train, labels_train)
   # pred = clf.predict(features_test)
    return clf
    ### your code goes here!
    

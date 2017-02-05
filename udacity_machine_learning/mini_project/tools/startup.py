#!/usr/bin/python


#原始程式是在python2.7執行的  本範例採用python3.6 針對部分程式碼做修正(ex. print 加上())

#檢查套件安裝
print
print ("checking for nltk")
try:
    import nltk
except ImportError:
    print ("you should install nltk before continuing")

print ("checking for numpy")
try:
    import numpy
except ImportError:
    print ("you should install numpy before continuing")

print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")

print ("checking for sklearn")
try:
    import sklearn
except:
    print ("you should install sklearn before continuing")

#下載資料
print
print ("downloading the Enron dataset (this may take a while)")
print ("to check on progress, you can cd up one level, then execute <ls -lthr>")
print ("Enron dataset should be last item on the list, along with its current size")
print ("download will complete at about 423 MB")

#在python 2.7 與 python3.6 的 urllib的用法有差異做修正
#import urllib 
import urllib.request
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
#urllib.urlretrieve(url, filename="../enron_mail_20150507.tgz") 
urllib.request.urlretrieve(url, filename="../enron_mail_20150507.tgz") 
print ("download complete!")


#解壓縮資料
print
print ("unzipping Enron dataset (this may take a while)")
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tgz", "r:gz")
tfile.extractall(".")

print ("you're ready to go!")

#引用random函式庫  以便進行亂數的呼叫
import random

#答案
answer=random.randint(1,10)

#猜測的變數
guess=0

#猜的次數
count =0

print("The answer is",answer)

# 當答案與猜測的值不同時 重新以亂數猜測
while answer != guess :
	print("Error,please try again!")
	guess=random.randint(1,10)
	count+=1

#輸出結果
print("You got the answer, you totall guess",count," times")


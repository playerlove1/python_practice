#求最大公因數
#使用輾轉相除法求解
print("輸入兩個數字:")

#接收輸入的參數
num1=int(input("數字1:"))
num2=int(input("數字2:"))

#輾轉相除過程
	#商不為0時進行迴圈
while num2 !=0:
	#餘數
	r=num1%num2
	#將原先的除數視為新被除數
	num1=num2
	#將原先的商視為新除數  當商=0即為整除離開迴圈)
	num2=r
print("最大公因數(GCD):",num1)
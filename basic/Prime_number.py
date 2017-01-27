#這是python求質數的範例
#求質數所用的方式:小於該整數一半以下的數 都無法整除該整數則為質數

#接收輸入的參數
input_number=int(input('輸入數字:'))

#該整數除以2的商
half_input_number=input_number//2

#迴圈(由 i=2 跑到一半+1)
for i in range(2,half_input_number+1):
	#如果該數被一半以下的數整除(即取餘數為0)
	if(input_number%i)==0:
		print(input_number,"不是質數,可被",i,"整除")
		break

#此為for迴圈的else敘述  當迴圈沒有被任何break中斷時會執行的敘述
else:
	print(input_number,"是質數")


try:
	input_num=int(input("請輸入整數:"))
	print("{0}為{1}".format(input_num,"奇數" if input_num%2 else "偶數"))
except ValueError as e:
	print(e.args)
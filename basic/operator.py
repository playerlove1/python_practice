

t=True
f=False

a=1
b=2

#python logic operator邏輯運算符
#python與其他程式語言不太一樣的地方
#其他程式語言可能是透過"&&"代表and "||"代表or "!"代表not
#python值些使用 and or not 作為關鍵字使用

#and
if a==1 and b==2:
	print("and 運算範例")

#or
if t or f:
	print("or運算範例")

#not
if not f:
	print("not")

#python is operator   is運算符
#用來判斷兩個物件(object)是否相等	
if t is f:
	print("t is f")
else:
	print("t is not f")

#可與not連用
if t is not f:
	print("t is not f")
else:
	print("t is f")
#但

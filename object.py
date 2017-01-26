#Python的所有東西都是Object,每個object都有 id(),type(),print()的函式

#list(串列)
simple_list=[1,"two",True]
print(simple_list)


# 即在串列的最後接上"新元素"字串
simple_list.append("新元素")
print(simple_list)
# 即移除串列中最後一個元素
simple_list.pop()
print(simple_list)
#即為將"two"元素由串列中移除
simple_list.remove("two")
print(simple_list) 
#即為在list index為0的位置插入 "zero"的元素
simple_list.insert(0,"zero") 
print(simple_list)

#透過len函數取得串列長度
len(simple_list)

#也可在[]中指定index值 直接進行修改
simple_list[len(simple_list)-1]=len(simple_list)-1
print(simple_list)

#刪除index 1以後的元素(包含index=1)  若輸入為 del simple_list[:] 即為清空該串列
del simple_list[1:]
print(simple_list)

# "[:]"對串列使用等於建立一個新串列，每個索引值都會參考到舊串列中每個所印位置的元素
#也就是所謂的淺層複製 (Shallow copy)
copy_list=simple_list[:]
print(copy_list)
#將list使用"*"運算 將會對list內容進行 淺層複製 (Shallow copy)
double_list=simple_list*2
print(double_list)

#將list使用"+"運算 將會建立一個新的物件 長度等於兩個串列的和  並合併兩個串列
add_list=copy_list+double_list
print(add_list)


#set (集合)
admins_set={"root","fong","s"}

users_set={"user1","user2","s"}

#交集
print("交集:",admins_set&users_set)
#聯集
print("聯集:",admins_set|users_set)
#差集
print("差集:",admins_set-users_set)
#XOR :排除共同元素
print("XOR:",admins_set^users_set)
#父集>子集     子集<父集  (> 與 < 回傳布林值  主要判斷兩個集合是否有父集與子集的關係)


#dictionary 即為key value 的模式  類似Java的map

passwords_dict={"user1":123,"user2":456} #同義於 passwords=dict(user1=123,user2=456)

#可以直接使用passwords["user1"]取得該變數的值  但若名稱不存在 則會拋出KeyError
#因此可以透過get取得  若不存在則回傳None(deault) 可在後面加入字串修改default回傳的結果
print(passwords_dict.get("user3","變數不存在"))

# 使用update() 增加元素
passwords_dict.update({"user3":789})
print(passwords_dict)
# 使用pop() 刪除元素
passwords_dict.pop("user3")
print(passwords_dict)
#可以使用passwords_dict.items()取得tuple
#可以使用passwords_dict.keys()取得key
#可以使用passwords_dict.value()取得value


#Tuple 像list一樣  但list是可變動 tuple是不可變動
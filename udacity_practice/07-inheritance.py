#介紹繼承
class Parent():
	def __init__(self, last_name, eye_color):
		print("Parent Constructor Called")
		self.last_name =last_name
		self.eye_color =eye_color
	def show_info(self):
		print("Last Name -- "+self.last_name)
		print("Eye Color --"+self.eye_color)
#繼承父類別
class Child(Parent):
	def __init__(self, last_name, eye_color, number_of_toys):
		print("Child Constructor Called")
		Parent.__init__(self, last_name, eye_color)
		self.number_of_toys = number_of_toys
	#Overriding 重寫方法
	def show_info(self):
		print("Last Name -- "+self.last_name)
		print("Eye Color --"+self.eye_color)
		print("Number of toys --"+self.number_of_toys)
		
		
yufong_wu = Parent("Wu","black")
yufong_wu.show_info()

child_wu = Child("Wu","black", 5)

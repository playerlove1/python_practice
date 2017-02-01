#用方形組成圓
import turtle

def     draw_square(some_turtle):
        for i in range(1,5):
                some_turtle.forward(100)
                some_turtle.right(90)

def     draw_art():
        
        #背景視窗
        window = turtle.Screen()
        window.bgcolor("red")
        #畫線的物件(turtle)
        #more information to change turtle
        #shape: https://docs.python.org/2/library/turtle.html#turtle.shape
        #color: https://docs.python.org/2/library/turtle.html#turtle.color
        #speed: https://docs.python.org/2/library/turtle.html#turtle.speed

        #建立方形
        brad=turtle.Turtle()
        brad.shape("turtle")
        brad.color("blue")
        brad.speed(2)
        #每次讓方形轉10度 共36次 
        for i in range(1,37):
                draw_square(brad)
                brad.right(10)
        #畫圓形
        #angie = turtle.Turtle()
        #angie.shape("arrow")
        #angie.color("yellow")
        #angie.circle(100)
        
        window.exitonclick()
	
draw_art()


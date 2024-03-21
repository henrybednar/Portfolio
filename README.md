# Creates turtle
import turtle

# Letter of numbers drawn
num_letters = 0

# Empty String for Input
user_input = ''

# Creates Menu and prompts for choice
menu_input = input('Choose one!\n1) Draw with the turtle!\n2) Draw with the mouse!')

# Runs code for Type draw
if menu_input == '1':
    user_input = input('Write a message for the turtle to draw! ')
    user_input = user_input.upper()

# Runs code for hand draw
if menu_input == '2':
    # Set up the turtle screen
    screen = turtle.Screen()
    screenTk = screen.getcanvas().winfo_toplevel()
    screenTk.attributes("-fullscreen", True)
    
    # Create Turtle
    pen = turtle.Turtle()
    pen.speed(100)  # Set drawing speed

    # Function to move the pen to a specified position
    def goto(x, y):
        pen.penup()
        pen.goto(x, y)
        pen.pendown()

    # Function to handle pen dragging
    def drag(x, y):
        pen.ondrag(None)  
        pen.goto(x, y)
        pen.ondrag(drag)  

   # Moves turtle to where you click
    pen.ondrag(drag)
    screen.onscreenclick(goto)

    # Main event loop
    turtle.done()

# Original cooridinates for turtle
x = -700
y = 300

# Creates turtle graphics in fullscreen
screen = turtle.Screen()
screenTk = screen.getcanvas().winfo_toplevel()
screenTk.attributes("-fullscreen", True)

# Sets pen size and speed
turtle.pensize(7)
turtle.speed(10)

def draw_letter_a(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'A'
    turtle.left(65)
    turtle.forward(size)
    turtle.right(130)
    turtle.forward(size)
    turtle.backward(size / 3)
    turtle.right(115)
    turtle.forward(size / 2)
    turtle.right(180)
    
def draw_letter_b(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'B'
    turtle.left(90)
    turtle.forward(size)
    turtle.right(90)
    turtle.forward(size/2)
    turtle.circle(-size/4, 180)
    turtle.forward(size/2)
    turtle.backward(size/2)
    turtle.right(180)
    turtle.circle(-size/4, 180)
    turtle.forward(size/2)
    turtle.right(180)
    
def draw_letter_c(size):
    turtle.penup()
    turtle.goto(x, y)
    

    # Draw the letter 'C'
    turtle.left(90)
    turtle.forward(size/2)
    turtle.pendown()
    turtle.circle(-size/2, 90)
    turtle.circle(-size/3, 45)
    turtle.circle(-size/3, -45)
    turtle.circle(-size/2, -90)
    turtle.circle(-size/2, -90)
    turtle.circle(-size/3, -45)
    turtle.right(225)
    
    #turtle.forward(size/8)
    #turtle.circle(size/3, 45)
    #turtle.circle(size/3, -45)
    #turtle.circle(size/2, -180)
    #turtle.circle(size/3, -45)
    #turtle.right(135)
    
def draw_letter_d(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'D'
    turtle.left(90)
    turtle.forward(size)
    turtle.right(90)
    turtle.forward(size/4)
    turtle.circle(-size/2, 180)
    turtle.forward(size/4)
    turtle.right(180)
    
def draw_letter_e(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'E'
    turtle.forward(size/1.5)
    turtle.backward(size/1.5)
    turtle.left(90)
    turtle.forward(size/2)
    turtle.right(90)
    turtle.forward(size/1.5)
    turtle.backward(size/1.5)
    turtle.left(90)
    turtle.forward(size/2)
    turtle.right(90)
    turtle.forward(size/1.5)
    
def draw_letter_f(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'F'
    turtle.left(90)
    turtle.forward(size/2)
    turtle.right(90)
    turtle.forward(size/1.5)
    turtle.backward(size/1.5)
    turtle.left(90)
    turtle.forward(size/2)
    turtle.right(90)
    turtle.forward(size/1.5)
    
def draw_letter_g(size):
    turtle.penup()
    turtle.goto(x, y)

    # Draw the letter 'G'
    turtle.right(90)
    turtle.backward(size/2)
    turtle.pendown()
    turtle.circle(size/2, 180)
    turtle.left(90)
    turtle.forward(size/2)
    turtle.backward(size/2)
    turtle.right(90)
    turtle.circle(size/2, -300)
    turtle.right(150)
    
def draw_letter_h(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'H'
    turtle.left(90)
    turtle.forward(size)
    turtle.backward(size/2)
    turtle.right(90)
    turtle.forward(size/1.5)
    turtle.left(90)
    turtle.forward(size/2)
    turtle.backward(size)
    turtle.right(90)
    
def draw_letter_i(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'I'
    turtle.forward(size/1.5)
    turtle.backward(size/3)
    turtle.left(90)
    turtle.forward(size)
    turtle.right(90)
    turtle.backward(size/3)
    turtle.forward(size/1.5)
    
def draw_letter_j(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'J'
    turtle.circle(size/3,-90)
    turtle.circle(size/3,180)
    turtle.forward(size/1.5)
    turtle.right(90)
    turtle.forward(size/3)
    turtle.backward(size/3)
    turtle.backward(size/3)
    turtle.forward(size/3)
    
def draw_letter_k(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'K'
    turtle.left(90)
    turtle.forward(size)
    turtle.backward(size/2)
    turtle.right(90)
    turtle.forward(3)
    turtle.left(40)
    turtle.forward(size/1.27)
    turtle.backward(size/1.27)
    turtle.right(80)
    turtle.forward(size/1.27)
    turtle.backward(size/1.27)
    turtle.left(40)
    
def draw_letter_l(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'L'
    turtle.forward(size/1.5)
    turtle.backward(size/1.5)
    turtle.left(90)
    turtle.forward(size)
    turtle.right(90)
    
def draw_letter_m(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'M'
    turtle.left(90)
    turtle.forward(size)
    turtle.right(135)
    turtle.forward(size/1.75)
    turtle.left(90)
    turtle.forward(size/1.75)
    turtle.right(135)
    turtle.forward(size)
    turtle.left(90)
    
def draw_letter_n(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    
    # Draw the letter 'N'
    turtle.left(90)
    turtle.forward(size)
    turtle.right(140)
    turtle.forward(size*1.27)
    turtle.left(140)
    turtle.forward(size)
    turtle.right(90)
    
def draw_letter_o(size):
    turtle.penup()
    turtle.goto(x, y)
    
    
    # Draw the letter 'O'
    turtle.left(90)
    turtle.forward(size)
    turtle.pendown()
    turtle.circle(-size, 360)
    turtle.right(90)
    
def draw_letter_p(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    
    # Draw the letter 'P'
    turtle.left(90)
    turtle.forward(size)
    turtle.right(90)
    turtle.forward(size/2)
    turtle.circle(-size/4, 180)
    turtle.forward(size/2)
    turtle.right(180)
    
def draw_letter_q(size):
    turtle.penup()
    turtle.goto(x, y)
    
    
    #Draw the letter 'Q'
    turtle.left(90)
    turtle.forward(size)
    turtle.pendown()
    turtle.circle(-size)
    turtle.circle(-size, 250)
    turtle.right(90)
    turtle.forward(size/2)
    turtle.backward(size/1.25)
    turtle.right(110)
    
def draw_letter_r(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'R'
    turtle.left(90)
    turtle.forward(size)
    turtle.right(90)
    turtle.forward(size/2)
    turtle.circle(-size/4, 180)
    turtle.forward(size/2)
    turtle.right(180)
    turtle.forward(size/6)
    turtle.right(40)
    turtle.forward(size/1.28)
    turtle.left(40)
    
def draw_letter_s(size):
    turtle.penup()
    turtle.goto(x, y)

    # Draw the letter 'S'
    turtle.left(90)
    turtle.forward(size/4)
    turtle.right(180)
    turtle.pendown()
    turtle.circle(size/4, 90)
    turtle.forward(size/8)
    turtle.circle(size/4, 155)
    turtle.forward(size/3)
    turtle.circle(-size/4, 155)
    turtle.forward(size/8)
    turtle.circle(-size/7, 90)
    turtle.left(90)
    
def draw_letter_t(size):
    turtle.penup()
    turtle.goto(x, y)

    # Draw the letter 'T'
    turtle.left(90)
    turtle.forward(size)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(size)
    turtle.backward(size/2)
    turtle.right(90)
    turtle.forward(size)
    turtle.left(90)
    
def draw_letter_u(size):
    turtle.penup()
    turtle.goto(x, y)

    # Draw the letter 'U'
    turtle.right(90)
    turtle.backward(size/2.4)
    turtle.pendown()
    turtle.backward(size/1.6)
    turtle.forward(size/1.6)
    turtle.circle(size/2.5, 180)
    turtle.forward(size/1.6)
    turtle.right(90)
    
def draw_letter_v(size):
    turtle.penup()
    turtle.goto(x, y)

    # Draw the letter 'V'
    turtle.right(90)
    turtle.backward(size/1.1)
    turtle.pendown()
    turtle.left(25)
    turtle.forward(size)
    turtle.left(130)
    turtle.forward(size)
    turtle.right(65)
    
def draw_letter_w(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'W'
    turtle.left(90)
    turtle.forward(size)
    turtle.backward(size)
    turtle.right(45)
    turtle.forward(size/1.75)
    turtle.right(90)
    turtle.forward(size/1.75)
    turtle.left(135)
    turtle.forward(size)
    turtle.right(90)
    
def draw_letter_x(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'X'
    turtle.left(45)
    turtle.forward(size)
    turtle.backward(size/2)
    turtle.left(90)
    turtle.forward(size/2)
    turtle.backward(size)
    turtle.right(135)
    
def draw_letter_y(size):
    turtle.penup()
    turtle.goto(x, y)
    

    # Draw the letter 'Y'
    turtle.left(90)
    turtle.forward(size)
    turtle.pendown()
    turtle.right(135)
    turtle.forward(size/1.55)
    turtle.left(90)
    turtle.forward(size/1.55)
    turtle.backward(size/1.55)
    turtle.right(135)
    turtle.forward(size/1.75)
    turtle.right(270)
    
def draw_letter_z(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    # Draw the letter 'Z'
    turtle.forward(size/1.25)
    turtle.backward(size/1.25)
    turtle.left(49)
    turtle.forward(size*1.2)
    turtle.left(131)
    turtle.forward(size/1.25)
    turtle.right(180)
    
def draw_space(size):
    turtle.penup()
    turtle.goto(x, y)

    # Draw the space
    turtle.forward(size)
    
def draw_period(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    

    # Draw the period
    turtle.circle(size, 360)
    
def draw_exclamation_point(size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    

    # Draw the exclamation point
    turtle.circle(size, 360)
    turtle.left(90)
    turtle.penup()
    turtle.forward(size*10)
    turtle.pendown()
    turtle.forward(size*25)
    turtle.right(90)
    
def draw_question_mark(size):
    turtle.penup()
    turtle.goto(x, y)
    
    # Draw the question mark
    turtle.left(90)
    turtle.forward(size*58)
    turtle.pendown()
    turtle.right(50)
    turtle.circle(-size*16.875, 230)
    turtle.left(100)
    turtle.forward(size*16.875)
    turtle.penup()
    turtle.forward(size*10)
    turtle.pendown()
    turtle.circle(size*1.6, 360)
    turtle.left(90)
    

# For how many letters the input is, how much space added
while num_letters != len(user_input):
    
    for char in user_input:
        if char == 'A':
            draw_letter_a(75)
        if char == 'B':
            draw_letter_b(67.5)
        if char == 'C':
            draw_letter_c(67.5)
        if char == 'D':
            draw_letter_d(67.5)
        if char == 'E':
            draw_letter_e(67.5)
        if char == 'F':
            draw_letter_f(67.5)
        if char == 'G':
            draw_letter_g(67.5)
        if char == 'H':
            draw_letter_h(67.5)
        if char == 'I':
            draw_letter_i(67.5)
        if char == 'J':
            draw_letter_j(67.5)
        if char == 'K':
            draw_letter_k(67.5)
        if char == 'L':
            draw_letter_l(67.5)
        if char == 'M':
            draw_letter_m(67.5)
        if char == 'N':
            draw_letter_n(67.5)
        if char == 'O':
            draw_letter_o(33.75)
        if char == 'P':
            draw_letter_p(67.5)
        if char == 'Q':
            draw_letter_q(33.75)
        if char == 'R':
            draw_letter_r(67.5)
        if char == 'S':
            draw_letter_s(60)
        if char == 'T':
            draw_letter_t(67.5)
        if char == 'U':
            draw_letter_u(67.5)
        if char == 'V':
            draw_letter_v(75)
        if char == 'W':
            draw_letter_w(67.5)
        if char == 'X':
            draw_letter_x(96)
        if char == 'Y':
            draw_letter_y(67.5)
        if char == 'Z':
            draw_letter_z(75)
        if char == ' ':
            draw_space(67.5)
        if char == '.':
            draw_period(2)
        if char == '!':
            draw_exclamation_point(2)
        if char == '?':
            draw_question_mark(1.25)
            
                    
        # Moves letters to next line
        if num_letters != 18 or 37 or 55 or 73 or 91 or 109:    
            x += 75  # Adjust the x position for the next letter
        if num_letters == 18:
            x = -700
            y = 200
        if num_letters == 37:
            x = -700
            y = 100
        if num_letters == 56:
            x = -700
            y = 0
        if num_letters == 75:
            x = -700
            y = -100
        if num_letters == 94:
            x = -700
            y = -200
        if num_letters == 113:
            x = -700
            y = -300
        if num_letters == 132:
            x = -700
            y = -400
        
        num_letters += 1    

import pygame
from tkinter import *
import time
import Scanner as scn

pygame.init()
screen = pygame.display.set_mode((450,450))
w = 450//9
h = 450//9
white = (255,255,255)
black = (0,0,0)
grey = (128,128,128)
red = (255,0,0)
screen.fill(white)

grid = scn.flowChart("7")

dp = []
for i in range(9):
	for j in range(9):
		if(grid[i][j] == 0):
			dp.append((i,j))

class Spot:
	def __init__(self, x, y, n):
		self.i = x
		self.j = y
		self.data = n

	def show(self, color):
		fnt = pygame.font.SysFont("comicsans", 40)
		pygame.draw.rect(screen, color, (self.j * w, self.i * h, w, h), 1)
		if(self.data != 0):
			text = fnt.render(str(self.data), 1, black)
			screen.blit(text, (self.j * w + w//2-10, self.i * h + h//2-10))
		pygame.display.update()

	def rePrint(self, color):
		fnt = pygame.font.SysFont("comicsans", 40)
		pygame.draw.rect(screen, color, (self.j * w, self.i * h, w, h), 1)
		text = fnt.render(str(self.data), 1, color)
		screen.blit(text, (self.j * w + w//2-10, self.i * h + h//2-10))
		pygame.display.update()
	
def drawLines():
	pygame.draw.line(screen, black, (450//3, 0),(450//3, 450), 3)
	pygame.draw.line(screen, black, ((2*450)//3, 0),((2*450)//3, 450), 3)
	pygame.draw.line(screen, black, (0, 450//3),(450, 450//3), 3)
	pygame.draw.line(screen, black, (0, (2*450)//3),(450, (2*450)//3), 3)
	pygame.display.update()

for i in range(9):
	for j in range(9):
		grid[i][j] = Spot(i, j, grid[i][j])

for i in range(9):
	for j in range(9):
		grid[i][j].show(grey)

drawLines()

def possible(x,y,n):
	global grid
	for i in range(9):
		if(grid[i][y].data == n):
			return False
		if(grid[x][i].data == n):
			return False
	anchor_x = 3*(x//3)
	anchor_y = 3*(y//3)
	for i in range(0, 3):
		for j in range(0, 3):
			if(grid[anchor_x+i][anchor_y+j].data == n):
				return False
	return True

def Solve():
	global grid
	for i in range(9):
		for j in range(9):
			if grid[i][j].data == 0:
				for n in range(1,10):
					if possible(i,j,n):
						grid[i][j].data = n						
						Solve()
						grid[i][j].data = 0
				return

	for i in range(9):
		for j in range(9):
			time.sleep(0.05)
			if((i, j) in dp):
				grid[i][j].rePrint(red)
	drawLines()
	return

def onClick():
	Solve()
	window.quit()
	window.destroy()

window = Tk()
solve = Button(window, text = "Solve", command = onClick)
solve.grid(row=1,column=2)
mainloop()

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
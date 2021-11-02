import tkinter as tk
from queue import Empty
from PIL import Image, ImageTk
import time
import pandas as pd


class Symbol():
    def __init__(self,display, x ,y, heading,radius,status, ownship,multiple_planning_agents):
        if display.count_uavs == 0:
            time.sleep(2)
            display.count_uavs = 1
            
        self.display = display
        if multiple_planning_agents:
            self.radius = radius / 2
        else:
            self.radius = radius
        _radius = self.display.meter2pix(self.radius)
        x_ = self.display.meter2pix_coords(x)
        y_ = self.display.meter2pix_coords(y)
        # print(x,y,x_,y_)
        # print('***********')
        self.ownship = ownship
        self.img_green = Image.open("./logo/eVTOL_icon_green.png")
        self.img_red = Image.open("./logo/eVTOL_icon_red.png")
        self.img_black= Image.open("./logo/eVTOL_icon_black.png")
        heading = 360 - heading
        self.img_green = self.img_green.resize((25, 25))
        self.img_green = self.img_green.rotate(heading)
        self.img_red = self.img_red.resize((25, 25))
        self.img_red = self.img_red.rotate(heading)
        self.img_black = self.img_black.resize((25, 25))
        self.img_black = self.img_black.rotate(heading)
        # print(heading)
        
        
        self.imgtk = ImageTk.PhotoImage(self.img_green)
        if ownship:
            # self.icon = self.display.canvas.create_oval(x_ - _radius, y_ - _radius, x_ + _radius, y_ + _radius)
            self.icon = self.display.canvas.create_image(x_, y_, image=self.imgtk)
            self.color='green'
        else:
            # self.icon = self.display.canvas.create_oval(x_ - _radius/10, y_ - _radius/10, x_ + _radius/10, y_ + _radius/10)
            self.icon = self.display.canvas.create_image(x_, y_, image=self.imgtk)
            self.color='black'
        if status == 'boom':
            self.color = 'red'
        self.change_color()

    def delete(self):
        self.display.canvas.delete(self.icon)

    def move(self,x,y,heading, status):
        x_ = self.display.meter2pix_coords(x)
        y_ = self.display.meter2pix_coords(y)
        _radius = self.display.meter2pix(self.radius)
        if self.ownship:
            # self.display.canvas.coords(self.icon, x_ - _radius, y_ - _radius, x_ + _radius, y_ + _radius)
            # self.img = self.img.rotate(heading)
            # self.imgtk = ImageTk.PhotoImage(self.img)
            # self.display.canvas.itemconfigure(self.icon, image=self.imgtk)
            self.display.canvas.coords(self.icon, x_, y_)
        else:
            # self.display.canvas.coords(self.icon, x_ - _radius/10, y_ - _radius/10, x_ + _radius/10, y_ + _radius/10)
            # self.img = self.img.rotate(heading)
            # self.imgtk = ImageTk.PhotoImage(self.img)
            # self.display.canvas.itemconfigure(self.icon, image=self.imgtk)
            self.display.canvas.coords(self.icon, x_, y_)
            
        if status == 'boom' and self.color is not 'red':
            self.color = 'red'
            self.change_color()
        elif status == 'ok':
            if self.ownship and self.color is not 'green':
                self.color = 'green'
                self.change_color()
            elif not self.ownship and self.color is not 'black':
                self.color = 'black'
                self.change_color()

    def change_color(self):
        # self.display.canvas.itemconfig(self.icon, outline=self.color)
        if self.color == 'green':
            self.imgtk = ImageTk.PhotoImage(self.img_green)
        elif self.color == 'red':
            self.imgtk = ImageTk.PhotoImage(self.img_red)
        elif self.color == 'black':
            self.imgtk = ImageTk.PhotoImage(self.img_black)
            
        self.display.canvas.itemconfigure(self.icon, image=self.imgtk)
        


class Display():
    def __init__(self, update_queue,length_arena,multiple_planning_agents=True,display_update=200):
        self.count_uavs = 0
        self.root = tk.Tk()
        self.root.title("UTM Simulator")
        self.root.resizable(True, False)
        self.root.aspect(1,1,1,1)
        self.canvas = tk.Canvas(self.root, width=700, height=700,borderwidth=0, highlightthickness=0)
        self.canvas.pack()
        self.border_ratio = 0.1
        self.length=length_arena
        self.multiple_planning_agents=multiple_planning_agents
        self.update_queue = update_queue
        self.display_update=display_update  # in ms how often to show the new position
        self.symbols = {}
        self.canvas.bind("<Configure>",self.create_border)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.plot_vertiport_flag = True
        
    def plot_vertiport(self):
        img_vertiport = Image.open("./logo/vertiport.png")
        img_vertiport = img_vertiport.resize((30, 30))
        
        vertiport_db = pd.read_csv("./config/vertiport_db.csv")
        # self.imgtks = []
        self.imgtk = ImageTk.PhotoImage(img_vertiport)
        for x,y in zip(vertiport_db['x'], vertiport_db['y']):
            # print(int(x),int(y))
            # self.imgtks.append(ImageTk.PhotoImage(img_vertiport))
            x_ = self.meter2pix_coords(x)
            y_ = self.meter2pix_coords(y)
            # print(x, y, x_, y_)
            self.canvas.create_image(x_, y_, image=self.imgtk)
        
    def create_border(self,event):
        x0=self.meter2pix_coords(0)
        y0=self.meter2pix_coords(0)
        x1=self.meter2pix_coords(self.length)
        y1=self.meter2pix_coords(self.length)
        self.canvas.create_line(x0,y0,x0,y1)
        self.canvas.create_line(x0, y0, x1, y0)
        self.canvas.create_line(x1, y1, x0, y1)
        self.canvas.create_line(x1, y1, x1, y0)

    def meter2pix(self,x):
        width_dis=self.canvas.winfo_width()
        return (1-self.border_ratio)*width_dis*x/self.length

    def meter2pix_coords(self,x):
        width_dis = self.canvas.winfo_width()
        # print("width dis::", width_dis)
        return self.border_ratio*width_dis/2 + (1-self.border_ratio)*width_dis * x / self.length

    def update(self):
        try:
            update_agents = self.update_queue.get(timeout=1.0)
        except Empty:
            print('nothing in the queue')
            self.root.quit()
            return
        
        agents_to_delete = []
        for agent_id, symbol in self.symbols.items():
            # if an agent is not present in the list, delete its representation from the canvas
            # otherwise we just update its position
            if agent_id not in update_agents.keys():
                symbol.delete()
                agents_to_delete.append(agent_id)
            else:
                x = update_agents[agent_id]['x']
                y = update_agents[agent_id]['y']
                heading = update_agents[agent_id]['heading']
                status = update_agents[agent_id]['status']
                symbol.move(x,y,heading,status)
                # print(update_agents[agent_id])
        for agent_to_delete in agents_to_delete:
            self.symbols.pop(agent_to_delete)

        # If an agent did not exist before, create it
        for update_agent_id, update_agent in update_agents.items():
            if update_agent_id not in self.symbols:
                x = update_agent['x']
                y = update_agent['y']
                radius = update_agent['radius']
                status = update_agent['status']
                ownship = update_agent['ownship']
                heading = update_agent['heading']
                symbol = Symbol(self,x,y,heading, radius,status,ownship,self.multiple_planning_agents)
                self.symbols[update_agent_id] = symbol
        
        if self.plot_vertiport_flag and self.canvas.winfo_width() > 1:
            self.plot_vertiport_flag = False
            self.plot_vertiport()
        self.canvas.after(self.display_update, self.update)

    def run(self):
        self.update()
        self.root.mainloop()
        
    def on_closing(self):
        self.root.destroy()




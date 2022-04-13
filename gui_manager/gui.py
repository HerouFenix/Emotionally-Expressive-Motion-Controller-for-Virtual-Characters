import os.path
import tkinter as tk
from turtle import left

class GUIManager():
    def __init__(self):
        self.window = tk.Tk()
        self.window.minsize(400,475)

        # Emotion Classification frame
        tk.Label(self.window, text = '== EMOTION PREDICTION ==', 
        font =('Verdana 13 bold')).pack(pady=(10,0))

        self.e_frame = tk.Frame(self.window)
        self.e_frame.pack(fill=tk.X)

        self.e_frame.columnconfigure(0, weight=1)
        self.e_frame.columnconfigure(1, weight=1)

        tk.Label(self.e_frame, text = 'Pleasure:', 
        font =('Verdana 11 bold')).grid(row=1, column=0)
        self.pleasure_text = tk.Label(self.e_frame, text = '0.00', 
        font =('Verdana 11')).grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.e_frame, text = 'Arousal:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.arousal_text = tk.Label(self.e_frame, text = '0.00', 
        font =('Verdana 11')).grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.e_frame, text = 'Dominance:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.dominance_text = tk.Label(self.e_frame, text = '0.00', 
        font =('Verdana 11')).grid(row=3, column=1, sticky=tk.W)

        self.closest_emotion = tk.Label(self.e_frame, text = 'Closest Emotion: Happy', 
        font =('Verdana 11 bold')).grid(row=4, pady=15, columnspan=2)


        # Emotion Classification frame
        tk.Label(self.window, text = '== EMOTION SYNTHESIS ==', 
        font =('Verdana 13 bold')).pack(pady=(20,0))

        self.n_frame = tk.Frame(self.window)
        self.n_frame.pack(fill=tk.X)

        self.n_frame.columnconfigure(0, weight=1)
        self.n_frame.columnconfigure(1, weight=1)

        tk.Label(self.n_frame, text = 'Pleasure:', 
        font =('Verdana 11 bold')).grid(row=1, column=0)
        self.new_pleasure = tk.Entry(self.n_frame, 
        font =('Verdana 11')).grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.n_frame, text = 'Arousal:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.new_arousal = tk.Entry(self.n_frame, 
        font =('Verdana 11')).grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.n_frame, text = 'Dominance:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.new_dominance = tk.Entry(self.n_frame, 
        font =('Verdana 11')).grid(row=3, column=1, sticky=tk.W)

        self.closest_emotion = tk.Button(self.n_frame, text = 'CONFIRM', 
        font =('Verdana 11 bold')).grid(row=6, pady=15, columnspan=2)


        # Emotion Classification frame
        tk.Label(self.window, text = '== STATUS ==', 
        font =('Verdana 13 bold')).pack(pady=(20,0))

        self.s_frame = tk.Frame(self.window)
        self.s_frame.pack(fill=tk.X)

        self.s_frame.columnconfigure(0, weight=1)
        self.s_frame.columnconfigure(1, weight=1)

        tk.Label(self.s_frame, text = 'Animation:', 
        font =('Verdana 11 bold')).grid(row=1, column=0)
        self.animation_statust = tk.Label(self.s_frame, text = 'Not Looped/Over', 
        font =('Verdana 11')).grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.s_frame, text = 'Emotion Prediction:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.emotion_prediction = tk.Label(self.s_frame, text = 'In Progress', 
        font =('Verdana 11')).grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.s_frame, text = 'Motion Synthesis:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.motion_synthesis = tk.Label(self.s_frame, text = 'Not Synthesizing', 
        font =('Verdana 11')).grid(row=3, column=1, sticky=tk.W)


    def update(self):
        self.window.mainloop()


manager = GUIManager()
manager.update()

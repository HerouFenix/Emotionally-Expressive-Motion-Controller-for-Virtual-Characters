import os.path
import tkinter as tk
from turtle import left

EMOTION_COORDINATES = {
    "neutral": (0.0, 0.0, 0.0),
    "angry": (-0.5, 0.6, 0.9),
    "happy": (0.6, 0.5, 0.2),
    "sad": (-0.6, -0.3, -0.3),
    "disgusted": (-0.4, 0.25, -0.1) ,
    "afraid": (-0.35, 0.7, -0.8),
    "pleased": (0.7, 0.2, 0.2),
    "bored": (-0.5, -0.7, -0.25),
    "tired": (0.1, -0.7, -0.2),
    "relaxed": (0.6, -0.55, 0.1),
    "excited": (0.5, 0.7, 0.4),
    "miserable": (-0.85, -0.1, -0.8),
    "nervous": (-0.3, -0.66, -0.7),
    "satisfied": (0.9, -0.25, 0.65),  
}

COLOURS = {
    "green": "#14631e",
    "red": "#5c111f",
    "blue": "#114759",
    "yellow": "#706a19"

}

class GUIManager():
    def __init__(self):
        self.window = tk.Tk()
        self.window.minsize(400,475)
        self.window.title("Emotion Prediction and Synthesis")

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
        font =('Verdana 11'))
        self.pleasure_text.grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.e_frame, text = 'Arousal:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.arousal_text = tk.Label(self.e_frame, text = '0.00', 
        font =('Verdana 11'))
        self.arousal_text.grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.e_frame, text = 'Dominance:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.dominance_text = tk.Label(self.e_frame, text = '0.00', 
        font =('Verdana 11'))
        self.dominance_text.grid(row=3, column=1, sticky=tk.W)

        self.closest_emotion = tk.Label(self.e_frame, text = 'Closest Emotion: neutral', 
        font =('Verdana 11 bold'))
        self.closest_emotion.grid(row=4, pady=15, columnspan=2)


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
        font =('Verdana 11'))
        self.new_pleasure.grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.n_frame, text = 'Arousal:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.new_arousal = tk.Entry(self.n_frame, 
        font =('Verdana 11'))
        self.new_arousal.grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.n_frame, text = 'Dominance:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.new_dominance = tk.Entry(self.n_frame, 
        font =('Verdana 11'))
        self.new_dominance.grid(row=3, column=1, sticky=tk.W)

        self.start_motion_synthesis = tk.Button(self.n_frame, text = 'CONFIRM', 
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
        self.animation_status = tk.Label(self.s_frame, text = 'Not Looped/Over', 
        font =('Verdana 11 bold'))
        self.animation_status.grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.s_frame, text = 'Emotion Prediction:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.emotion_prediction = tk.Label(self.s_frame, text = 'In Progress', 
        font =('Verdana 11 bold'))
        self.emotion_prediction.grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.s_frame, text = 'Motion Synthesis:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.motion_synthesis = tk.Label(self.s_frame, text = 'Not Synthesizing', 
        font =('Verdana 11 bold'))
        self.motion_synthesis.grid(row=3, column=1, sticky=tk.W)

    def update(self):
        self.window.update()

    def _find_closest_emotion(self, pad):
        p, a, d = pad
        dist = lambda key: (p - EMOTION_COORDINATES[key][0]) ** 2 + (a - EMOTION_COORDINATES[key][1]) ** 2 + (d - EMOTION_COORDINATES[key][2]) ** 2
        return min(EMOTION_COORDINATES, key=dist)

    def change_emotion_coordinates(self, pleasure, arousal, dominance):
        self.pleasure_text.config(text = str(pleasure))
        self.arousal_text.config(text = str(arousal))
        self.dominance_text.config(text = str(dominance))
        
        self.closest_emotion.config(text = "Closest Emotion: " + self._find_closest_emotion((pleasure,arousal,dominance)))

    def change_animation_status(self, new_status):
        if(new_status == 0):
            self.animation_status.config(text="Running", fg=COLOURS['yellow'])
        elif(new_status == 1):
            self.animation_status.config(text="Looped", fg=COLOURS['green'])
        elif(new_status == 2):
            self.animation_status.config(text="Stopped", fg=COLOURS['red'])

    def change_emotion_prediction_status(self, new_status):
        if(new_status == 0):
            self.emotion_prediction.config(text="Stopped", fg=COLOURS['red'])
        elif(new_status == 1):
            self.emotion_prediction.config(text="In Progress", fg=COLOURS['yellow'])
        elif(new_status == 2):
            self.emotion_prediction.config(text="Finished", fg=COLOURS['green'])

#manager = GUIManager()
#while True:
#    manager.update()

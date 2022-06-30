import os.path
import tkinter as tk

EMOTION_COORDINATES = {
    (0.05, -0.05, 0.0): "Neutral", # Normal bandai 1/2
    
    (0.1, -0.7, -0.2): "Tired", # Tired bandai 1
    (0.1, -0.75, -0.25): "Tired", # Exhausted bandai 2
    (-0.1, -0.6, -0.15): "Exhausted", # Old bandai 1 & Elderly bandai 2
    
    (-0.5, 0.8, 0.9): "Angry", # Angry bandai 1
    
    (0.8, 0.5, 0.15): "Happy", # Happy bandai 1
    (0.6, 0.4, 0.1): "Happy", # Youthful bandai 2
    
    (-0.6, -0.4, -0.3): "Sad", # Sad bandai 1
    
    (0.4, 0.2, 0.35): "Proud", # Proud bandai 1
    (0.3, 0.3, 0.9): "Confident", # Giant bandai 1 
    (0.25, 0.15, 0.4): "Confident", # Masculine / Masculinity bandai 1
    (0.3, 0.4, 0.6): "Confident", # Masculine bandai 2
    
    (-0.6, 0.7, -0.8): "Afraid", # Not confident bandai 1
    
    (0.1, 0.6, 0.4): "Active", # Active bandai 1/2
    
    ## Kin ##
    #(-0.5, 0.7, 0.9): "Angry", 
    #(0.6, 0.5, 0.2): "Happy", 
    #(-0.6, -0.3, -0.3): "Sad", 
    #(-0.4, 0.25, -0.1): "Disgusted" , 
    #(-0.5, 0.7, -0.8): "Afraid", 
    #(0.0, 0.0, 0.0): "Neutral"  
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
        self.window.minsize(415,515)
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
        font =('Verdana 10 bold'))
        self.closest_emotion.grid(row=4, pady=10, columnspan=2)


        # Emotion Classification frame
        tk.Label(self.window, text = '== EMOTION SYNTHESIS ==', 
        font =('Verdana 13 bold')).pack(pady=(20,0))

        self.n_frame = tk.Frame(self.window)
        self.n_frame.pack(fill=tk.X)

        self.n_frame.columnconfigure(0, weight=1, uniform="oof")
        self.n_frame.columnconfigure(1, weight=1, uniform="oof")
        self.n_frame.columnconfigure(2, weight=1, uniform="oof")

        tk.Label(self.n_frame, text = 'Pleasure:', 
        font =('Verdana 11 bold')).grid(row=1, column=0)
        self.new_pleasure_slider = tk.Scale(
            self.n_frame,
            from_= -1.0,
            to=1.0,
            orient='horizontal',
            resolution = 0.01,
            showvalue=0,
            length=None
        )
        self.new_pleasure_slider.grid(row=1, column=1, sticky=tk.NSEW)
        self.new_pleasure_label = tk.Label(self.n_frame, text = '0.0', 
        font =('Verdana 10'))
        self.new_pleasure_label.grid(row=1, column=2, sticky=tk.NSEW)
        
        tk.Label(self.n_frame, text = 'Arousal:', 
        font =('Verdana 11 bold')).grid(row=2, column=0)
        self.new_arousal_slider = tk.Scale(
            self.n_frame,
            from_= -1.0,
            to=1.0,
            orient='horizontal',
            resolution = 0.01,
            showvalue=0,
            length=None
        )
        self.new_arousal_slider.grid(row=2, column=1, sticky=tk.NSEW)
        self.new_arousal_label = tk.Label(self.n_frame, text = '0.0', 
        font =('Verdana 10'))
        self.new_arousal_label.grid(row=2, column=2, sticky=tk.NSEW)

        tk.Label(self.n_frame, text = 'Dominance:', 
        font =('Verdana 11 bold')).grid(row=3, column=0)
        self.new_dominance_slider = tk.Scale(
            self.n_frame,
            from_= -1.0,
            to=1.0,
            orient='horizontal',
            resolution = 0.01,
            showvalue=0,
            length=None,
        )
        self.new_dominance_slider.grid(row=3, column=1, sticky=tk.NSEW)
        self.new_dominance_label = tk.Label(self.n_frame, text = '0.0', 
        font =('Verdana 10'))
        self.new_dominance_label.grid(row=3, column=2, sticky=tk.NSEW)

        self.start_motion_synthesis = tk.Button(self.n_frame, text = 'CONFIRM', 
        font =('Verdana 11 bold'))
        self.start_motion_synthesis.grid(row=4, pady=15, columnspan=3)

        # Preset Emotions
        # Row 1
        self.tired_button = tk.Button(self.n_frame, text = 'Exhausted', 
        font =('Verdana 10 bold'), width=7, command= lambda: self._change_emotion_by_preset(0))
        self.tired_button.grid(row=5, pady=5, column=0, columnspan=1)

        self.confident_button = tk.Button(self.n_frame, text = 'Confident', 
        font =('Verdana 10 bold'), width=7, command= lambda: self._change_emotion_by_preset(1))
        self.confident_button.grid(row=5, pady=5, column=1, columnspan=1)

        self.angry_button = tk.Button(self.n_frame, text = 'Angry', 
        font =('Verdana 10 bold'), width=7, command= lambda: self._change_emotion_by_preset(2))
        self.angry_button.grid(row=5, pady=5, column=2, columnspan=1)

        # Row 2
        self.afraid_button = tk.Button(self.n_frame, text = 'Afraid', 
        font =('Verdana 10 bold'), width=7, command= lambda: self._change_emotion_by_preset(3))
        self.afraid_button.grid(row=6, pady=5, column=0, columnspan=1)

        self.happy_button = tk.Button(self.n_frame, text = 'Happy', 
        font =('Verdana 10 bold'), width=7, command= lambda: self._change_emotion_by_preset(4))
        self.happy_button.grid(row=6, pady=5, column=1, columnspan=1)

        self.sad_button = tk.Button(self.n_frame, text = 'Sad', 
        font =('Verdana 10 bold'), width=7, command= lambda: self._change_emotion_by_preset(5))
        self.sad_button.grid(row=6, pady=5, column=2, columnspan=1)




        # Emotion Classification frame
        tk.Label(self.window, text = '== STATUS ==', 
        font =('Verdana 13 bold')).pack(pady=(20,0))

        self.s_frame = tk.Frame(self.window)
        self.s_frame.pack(fill=tk.X, pady=(0,20))

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
        self.new_pleasure_label.config(text=str(self.new_pleasure_slider.get()))
        self.new_arousal_label.config(text=str(self.new_arousal_slider.get()))
        self.new_dominance_label.config(text=str(self.new_dominance_slider.get()))

        self.window.update()

    def _change_emotion_by_preset(self, index):
        if(index == 0): # Tired
            self.new_pleasure_slider.set(0.1)
            self.new_arousal_slider.set(-0.75)
            self.new_dominance_slider.set(-0.25)

        elif(index == 1): # Confident
            self.new_pleasure_slider.set(0.3)
            self.new_arousal_slider.set(0.3)
            self.new_dominance_slider.set(0.9)

        elif(index == 2): # Angry
            self.new_pleasure_slider.set(-0.5)
            self.new_arousal_slider.set(0.8)
            self.new_dominance_slider.set(0.9)

        elif(index == 3): # Afraid
            self.new_pleasure_slider.set(-0.6)
            self.new_arousal_slider.set(0.7)
            self.new_dominance_slider.set(-0.8)

        elif(index == 4): # Happy
            self.new_pleasure_slider.set(0.8)
            self.new_arousal_slider.set(0.5)
            self.new_dominance_slider.set(0.15)

        elif(index == 5): # Sad
            self.new_pleasure_slider.set(-0.6)
            self.new_arousal_slider.set(-0.4)
            self.new_dominance_slider.set(-0.3)

        else: # Neutral
            self.new_pleasure_slider.set(0.05)
            self.new_arousal_slider.set(-0.1)
            self.new_dominance_slider.set(0.0)

    def _find_closest_emotion(self, pad):
        p, a, d = pad
        #dist = lambda key: (p - EMOTION_COORDINATES[key][0]) ** 2 + (a - EMOTION_COORDINATES[key][1]) ** 2 + (d - EMOTION_COORDINATES[key][2]) ** 2
        dist = lambda key: (p - key[0]) ** 2 + (a - key[1]) ** 2 + (d - key[2]) ** 2

        closest_coordinates = min(EMOTION_COORDINATES, key=dist)
        distance = (p - closest_coordinates[0]) ** 2 + (a - closest_coordinates[1]) ** 2 + (d - closest_coordinates[2]) ** 2

        return EMOTION_COORDINATES[closest_coordinates], distance

    def change_emotion_coordinates(self, pleasure, arousal, dominance):
        self.pleasure_text.config(text = str(pleasure))
        self.arousal_text.config(text = str(arousal))
        self.dominance_text.config(text = str(dominance))
        
        closest_emotion, closest_emotion_dist = self._find_closest_emotion((pleasure,arousal,dominance))
        if(closest_emotion_dist < 0.03):
            self.closest_emotion.config(fg=COLOURS['green'])
        elif(closest_emotion_dist < 0.06):
            self.closest_emotion.config(fg=COLOURS['yellow'])
        else:
            self.closest_emotion.config(fg=COLOURS['red'])
            
        self.closest_emotion.config(text = "Closest Emotion: " + str(closest_emotion))

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

    def change_motion_synthesizer_status(self, new_status):
        if(new_status == 0):
            self.motion_synthesis.config(text="Not Synthesizing", fg=COLOURS['red'])
        elif(new_status == 1):
            self.motion_synthesis.config(text="In Progress", fg=COLOURS['yellow'])

    def get_pad(self):
        pleasure = float(self.new_pleasure_slider.get())
        arousal = float(self.new_arousal_slider.get())
        dominance = float(self.new_dominance_slider.get())

        """
        if(pleasure > 1.0):
            pleasure = 1.0
            self.new_pleasure.delete(0,tk.END)
            self.new_pleasure.insert(0,"1.0")
        elif(pleasure < -1.0):
            pleasure = -1.0
            self.new_pleasure.delete(0,tk.END)
            self.new_pleasure.insert(0,"-1.0")

        if(arousal > 1.0):
            arousal = 1.0
            self.new_arousal.delete(0,tk.END)
            self.new_arousal.insert(0,"1.0")
        elif(arousal < -1.0):
            arousal = -1.0
            self.new_arousal.delete(0,tk.END)
            self.new_arousal.insert(0,"-1.0")

        if(dominance > 1.0):
            dominance = 1.0
            self.new_pleasure.delete(0,tk.END)
            self.new_pleasure.insert(0,"1.0")
        elif(dominance < -1.0):
            dominance = -1.0
            self.new_dominance.delete(0,tk.END)
            self.new_dominance.insert(0,"-1.0")
        """

        print([pleasure, arousal, dominance])
        return [pleasure, arousal, dominance]

#manager = GUIManager()
#while True:
#    manager.update()

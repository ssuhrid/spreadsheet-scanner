import pyttsx

engine = pyttsx.init()
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[0].id) #change index to change voices
engine.say('choos le')

engine.runAndWait()

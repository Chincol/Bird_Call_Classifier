
import os,sys,time

from PyQt4.QtGui import * 
from PyQt4.QtCore import * 

import numpy as np

from model.load_model import load_music_tagger_model
from model.audio_conv_utils import decode_predictions, preprocess_input

def predict_song_genre(model,audio_path):
    melgram = preprocess_input(audio_path)
    melgrams = np.expand_dims(melgram, axis=0)
    preds = model.predict(melgrams)
    print('Predicted:')
    print(decode_predictions(preds))

class main_window(QWidget):
	def __init__(self,parent=None):
		super(main_window,self).__init__()
		self.setAcceptDrops(True)
		self.init_vars()
		self.init_ui()

	def init_vars(self):
		self.tagger = load_music_tagger_model(weights='msd')

	def init_ui(self):
		self.setWindowTitle("Music Classifier")

		self.setFixedWidth(300)
		self.setFixedHeight(150)

		self.layout = QVBoxLayout(self)

		self.label = QLabel("Drop .mp3 Here",self)
		self.label.move(100,30)
		self.label.setFixedWidth(150)

		self.show()

	def dragMoveEvent(self,event):
		if event.mimeData().hasUrls:
			event.setDropAction(Qt.CopyAction)
			event.accept()
		else:
			event.ignore()

	def dragEnterEvent(self,event):
		if event.mimeData().hasUrls:
			event.accept()
		else:
			event.ignore()

	def dropEvent(self,event):
		if event.mimeData().hasUrls:
			event.setDropAction(Qt.CopyAction)
			event.accept()
			links=[]
			for url in event.mimeData().urls():
				links.append(str(url.toLocalFile()))
			self.classify_song(str(links[0]))
		else:
			event.ignore()

	def classify_song(self,song_path):
		if song_path.find(".mp3")!=-1:
			print("Predicting genre...")
			predict_song_genre(self.tagger,song_path)
		else:
			print(song_path)

def main():
	app = QApplication(sys.argv)
	app.setStyle('plastique')
	window = main_window(app)
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
from __future__ import print_function
from __future__ import absolute_import

# Standard imports
import os,sys,librosa,random
import numpy as np 

# Keras imports 
import keras
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model

TAGS = None # to hold all bird types

def get_bird_names():
	data_dir = "data/VB100/vb100_audio/"
	bird_dirs = os.listdir(data_dir)
	return bird_dirs

def print_sample_counts():

	def get_sample_count(row):
		return row[1]

	data_dir = "data/VB100/vb100_audio/"
	bird_dirs = os.listdir(data_dir)
	bird_data = []

	for b in bird_dirs:
		num_samples = len(os.listdir(data_dir+b))
		print("%s:\t%d samples"%(b,num_samples))
		bird_data.append([b,num_samples])

	print("--")

	bird_data = sorted(bird_data,key=get_sample_count)

	print("Sorted:")
	for b in bird_data:
		print("%s, count: %d"%(b[0],b[1]))

def get_species_count():
	data_dir = "data/VB100/vb100_audio"
	bird_dirs = os.listdir(data_dir)
	return len(bird_dirs)

# Decodes the output of bird tagger model
def decode_predictions(preds, top_n=5):
    results = []
    for pred in preds:
        result = zip(TAGS, pred)
        result = sorted(result, key=lambda x: x[1], reverse=True)
        results.append(result[:top_n])
    return results

# Reads an audio file & outputs a Mel-spectrogram
def preprocess_input(audio_path):
    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

    src, sr = librosa.load(audio_path, sr=SR)
    n_sample = src.shape[0]
    n_sample_wanted = int(DURA * SR)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[(n_sample - n_sample_wanted) / 2:
                  (n_sample + n_sample_wanted) / 2]

    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MELS) ** 2,
              ref_power=1.0)
    return np.expand_dims(x,axis=3)

# Loads in all mp3 data
def load_data(split=0.9,max_classes=None):
	data_dir = "data/VB100/vb100_audio/"
	bird_dirs = os.listdir(data_dir)
	num_classes = len(bird_dirs)

	X = [] # mel-spectrogram inputs
	Y = [] # onehot outputs

	test_X=[]
	test_Y=[]

	if max_classes!=None: num_classes = max_classes 

	for i in range(num_classes):

		cur_bird = bird_dirs[i]
		cur_bird_dir = os.path.join(data_dir,cur_bird)
		cur_bird_samples = os.listdir(cur_bird_dir)

		for c in cur_bird_samples:
			sys.stdout.write("\rLoading data... %d"%(i+1))

			# onehot output
			cur_output = np.zeros(num_classes)
			cur_output[i] = 1.0

			# melspectrogram output
			cur_input = preprocess_input(os.path.join(cur_bird_dir,c))

			# decide whether to place in test or train set
			if random.random()<split:
				X.append(cur_input)
				Y.append(cur_output)
			else:
				test_X.append(cur_input)
				test_Y.append(cur_output)
	
	sys.stdout.write("\nTrain set: %d | Test set: %d\n"%(len(X),len(test_X)))

	train_X = np.array(X)
	train_Y = np.array(Y)
	test_X = np.array(test_X)
	test_Y = np.array(test_Y)

	return train_X,train_Y,test_X,test_Y

# Predicts the bird species which produced audio file
def predict_bird(model,audio_path):
	melgram = preprocess_input(audio_path)
	melgrams = np.expand_dims(melgram,axis=0)
	preds = model.predict(melgrams)
	print("Predicted: ")
	print(decode_predictions(preds))
	return preds 

def train_model(model,train_X,train_Y,test_X,test_Y):
	batch_size = 12
	epochs = 4
	verbose = 1 

	model.fit(train_X,train_Y,batch_size=batch_size,
				epochs=epochs,verbose=verbose,
				validation_data=(test_X,test_Y))
	score = model.evaluate(test_X,test_Y,verbose=0)
	print("Test loss: %0.5f"%score[0])
	print("Test acc: %0.5f"%score[1])

	print("Saving model...")
	model.save("model/conv2D_classifier.h5")

# Builds the audio classification model
def build_model(weights='msd', input_tensor=None, include_top=True, num_outputs=100):

    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Input shape for tensorflow
    input_shape = (96,1366,1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # settings for tensorflow
    channel_axis=3 
    freq_axis=1
    time_axis=2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    # reshaping
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    if include_top: x = Dense(num_outputs, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None: 
    	print("Compiling model...")
    	model.compile(
    			optimizer=keras.optimizers.Adam(lr=5e-3),
    			loss='binary_crossentropy',
    			metrics=['accuracy'])
    	return model
    else:
        model.load_weights(WEIGHTS_PATH, by_name=True)
        return model

def create_model():

	num_bird_types = get_species_count()
	print("Found %d bird types."%num_bird_types)

	print("Building model...")
	model = build_model(weights=None,num_outputs=num_bird_types)

	train_X,train_Y,test_X,test_Y = load_data(max_classes=num_bird_types)

	print("Training model...")
	train_model(model,train_X,train_Y,test_X,test_Y)

	print("Done")

def test_model():
	global TAGS 
	TAGS = get_bird_names()
	
	print("Loading model...")
	from keras.models import load_model
	model = load_model("model/conv2D_classifier.h5")
	
	test_file = "data/VB100/vb100_audio/Stilt_Sandpiper/Stilt_Sandpiper_00001.mp3"
	predict_bird(model,test_file)

def main():
	create_model()
	test_model()


if __name__ == '__main__':
	main()
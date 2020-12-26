#import argparse
import tensorflow.compat.v1 as tf
import ctc_utils
import omr_utils
import skimage
import numpy as np
import os
import sys
from skimage import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vocabulary_file_path = os.path.join(BASE_DIR, 'vocabulary_agnostic.txt')
model_path = os.path.join(BASE_DIR, 'Models', 'agnostic_model.meta')

def detect_symbols(img_data):
	tf.reset_default_graph()
	sess = tf.InteractiveSession()

	# Read the dictionary
	dict_file = open(vocabulary_file_path,'r')
	dict_list = dict_file.read().splitlines()
	int2word = dict()
	for word in dict_list:
	    word_idx = len(int2word)
	    int2word[word_idx] = word
	dict_file.close()

	# Restore weights
	saver = tf.train.import_meta_graph(model_path)
	saver.restore(sess, model_path[:-5])

	graph = tf.get_default_graph()

	input = graph.get_tensor_by_name("model_input:0")
	seq_len = graph.get_tensor_by_name("seq_lengths:0")
	rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
	height_tensor = graph.get_tensor_by_name("input_height:0")
	width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
	logits = tf.get_collection("logits")[0]

	# Constants that are saved inside the model itself
	WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

	decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

	#image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
	image = img_data
	image = ctc_utils.resize(image, HEIGHT)
	image = ctc_utils.normalize(image)
	image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

	seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

	prediction = sess.run(decoded,
	                      feed_dict={
	                          input: image,
	                          seq_len: seq_lengths,
	                          rnn_keep_prob: 1.0,
	                      })

	str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
	return [int2word[w] for w in str_predictions[0]]

if __name__ == '__main__':
	img_path = sys.argv[1]
	img_data = skimage.io.imread(img_path, as_gray=True) # reads image as grayscale
	img_data = skimage.util.img_as_ubyte(img_data)

	#apply binarization method and aligns staff
	img_data = omr_utils.align_staff(img_data, 'otsu')
	img_data = omr_utils.apply_threshold(img_data, 'otsu')
	io.imsave('bin_img.png', img_data)
	line_thickness, staff_space = omr_utils.get_reference_lenghts(img_data)

	symbols = detect_symbols(img_data)
	omr_utils.convert_to_midi(symbols)
	print(symbols)
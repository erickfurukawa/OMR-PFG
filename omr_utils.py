import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import re

import skimage
from skimage import transform, filters, io, measure, restoration, util
from skimage.color import rgb2gray
from skimage.morphology import disk, erosion, dilation, closing

from midiutil.MidiFile import SHARPS, FLATS, MIDIFile, MAJOR

#----------------------------------------------MUSIC RECONSTRUCTION----------------------------------------------

class Score:
	class Accidental:
		nothing = 'nothing'
		natural = 'natural'
		sharp = 'sharp'
		flat = 'flat'
		doubleSharp = 'doubleSharp'
		doubleFlat = 'doubleFlat'

	class Note:
		def __init__(self, note_id, position, duration, pitch):
			self.note_id = note_id
			self.position = position
			self.duration = duration
			self.pitch = pitch

		def __str__(self):
			return "Note: {} Position: {} Duration: {} Midi_value: {}".format(self.note_id, self.position, self.duration, self.pitch)

	MIDI_NOTES_FIRST_OCTAVE = {
		'C': 24,
		'D': 26,
		'E': 28,
		'F': 29,
		'G': 31,
		'A': 33,
		'B': 35
	}

	ACCIDENTAL_TO_VALUE = {
		Accidental.sharp: 1,
		Accidental.doubleSharp: 2,
		Accidental.flat: -1,
		Accidental.doubleFlat: -2,
		Accidental.natural: 0,
		Accidental.nothing: 0
	}

	CLEF_STR_TO_CLEF = {
		'C-L1': 'soprano',
		'C-L2': 'mezzosoprano',
		'C-L3': 'alto',
		'C-L4': 'tenor',
		'C-L5': 'baritone',
		'F-L4': 'bass',
		'F-L5':	'subbass',
		'G-L1':	'french',
		'G-L2':	'treble'
	}

	CLEFS_L0_NOTE = {
		'treble': ('C', 4),
		'french': ('E', 4),
		'subbass': ('C', 2),
		'baritone': ('G', 2),
		'mezzosoprano': ('F', 3),
		'soprano': ('A', 3),
		'bass': ('E', 2),
		'alto': ('D', 3),
		'tenor': ('B', 2)
	}

	NOTE_TO_ID = {
		'whole': 1,
		'half': 2,
		'quarter': 4,
		'eighth': 8,
		'sixteenth': 16,
		'thirty_second': 32,
		'sixty_fourth': 64
	}

	BEAMS_TO_ID = {
		1: 8,
		2: 16,
		3: 32,
		4: 64,
	}

	def __init__(self):
		self.bpm = 100
		self.clef = 'treble'
		self.time_signature = [4,4]
		self.notes = []
		self.key_accidentals = {}
		self.bar_accidentals = {}
		self.clear_all_accidentals()
		self.key_signature = ['natural', 0]
		self.__first_bar = True
		self.first_bar_duration = 0

	def clear_all_accidentals(self):
		for space_line in ['L', 'S']:
			for i in range(-3,9):
				self.bar_accidentals[space_line+str(i)] = self.Accidental.nothing
		for note in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
			self.key_accidentals[note] = self.Accidental.nothing

	def add_key_accidental(self, accidental, position):
		note, _= self.position_to_note(position)
		self.key_accidentals[note] = accidental

	def add_bar_accidental(self, accidental, position):
		position = position[0]+str(position[1])
		self.bar_accidentals[position] = accidental

	def get_key_accidental(self, position):
		note, _= self.position_to_note(position)
		return self.key_accidentals[note]

	def get_bar_accidental(self, position):
		position = position[0]+str(position[1])
		return self.bar_accidentals[position]

	def new_bar(self):
		self.__first_bar = False
		for space_line in ['L', 'S']:
			for i in range(-3,9):
				self.bar_accidentals[space_line+str(i)] = self.Accidental.nothing

	def position_to_note(self, position):
		distance_to_L0 = int(position[1])*2
		if position[0] == 'S':
			distance_to_L0 += 1
		notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

		distance_to_L0 = distance_to_L0 + notes.index(self.CLEFS_L0_NOTE[self.clef][0])
		note_index = distance_to_L0 % 7
		octave = self.CLEFS_L0_NOTE[self.clef][1] + (distance_to_L0 // 7) # floor division
		
		return notes[note_index], octave

	def note_to_midi_value(self, note, accidental, octave):
		notes_per_octave = 12
		if note not in self.MIDI_NOTES_FIRST_OCTAVE.keys():
			raise Exception("Note {} is invalid.".format(note))
		return self.MIDI_NOTES_FIRST_OCTAVE[note] + self.ACCIDENTAL_TO_VALUE[accidental] + (octave-1)*notes_per_octave

	def add_note(self, note_str, rest=False):
		note_id, position = self.detect_note(note_str)
		duration = self.time_signature[1]/note_id # duration in beats
		if self.__first_bar:
			self.first_bar_duration += duration
		if not rest:
			#duration = 60/self.bpm * self.time_signature[1]/note_id #duration in seconds
			note, octave = self.position_to_note(position)
			accidental = self.Accidental.nothing
			if self.get_bar_accidental(position) == self.Accidental.nothing: # no bar accidentals, check key accidentals
				accidental = self.key_accidentals[note]
			else:
				accidental = self.get_bar_accidental(position)
			note_value = self.note_to_midi_value(note, accidental, octave)
			self.notes.append(self.Note(note_id, position, duration, note_value))
		else:
			self.notes.append(self.Note(note_id, position, duration, 0))

	def add_accidental(self, accidental_str):
		position = self.detect_position(accidental_str)
		regex = '(.+?)-.+'
		accidental = re.match(regex, accidental_str).group(1)
		if len(self.notes) > 0:
			self.add_bar_accidental(accidental, position)
		else:
			self.add_key_accidental(accidental, position)
			accidental_no = 0
			which_accidental = self.Accidental.nothing
			for note in self.key_accidentals.keys():
				if self.key_accidentals[note] != self.Accidental.nothing:
					accidental_no += 1
					which_accidental = self.key_accidentals[note]
			self.key_signature = [which_accidental, accidental_no]


	def add_to_time_signature(self, digit_str):
		position = self.detect_position(digit_str)
		distance_to_L0 = position[1]*2
		if position[0] == 'S':
			distance_to_L0 += 1
		digit = int(digit_str[0])

		if digit_str[-2:] == 'L4':
			self.time_signature[0] = digit
		elif digit_str[-2:] == 'L2':
			if digit in [2,4,8,16,32,64]:
				self.time_signature[1] = digit

	def detect_position(self, symbol_str):
		regex = '-(L|S)(-?[0-9])$'
		m = re.search(regex, symbol_str)
		return [m.group(1), int(m.group(2))]

	def add_dot(self, dot_str):
		position = self.detect_position(dot_str)
		if len(self.notes) > 0:
			self.notes[-1].duration += self.notes[-1].duration/2

	def add_clef(self, clef_str):
		self.clef = self.CLEF_STR_TO_CLEF[clef_str]

	def detect_note(self, note_str):
		note_id = 0
		position_match = None
		if note_str.startswith("beamed"):
			regex = ".*([0-9])-.*"
			m = re.match(regex, note_str)
			beams = m.group(1)
			note_id = self.BEAMS_TO_ID[int(beams)]
		else:
			for note in self.NOTE_TO_ID.keys():
				if note_str.startswith(note):
					note_id = self.NOTE_TO_ID[note]

		position = self.detect_position(note_str)
		return note_id, position

def convert_to_midi(symbols_list):
	score = Score()
	for symbol in symbols_list:
		if symbol.startswith("note.") or symbol.startswith("rest."):
			rest = symbol.startswith("rest.")
			score.add_note(symbol[5:], rest)
		elif symbol.startswith("accidental."):
			score.add_accidental(symbol[11:])
		elif symbol.startswith("dot-"):
			score.add_dot(symbol)
		elif symbol.startswith("barline"):
			score.new_bar()
			first_bar = False
		elif symbol.startswith("digit"):
			score.add_to_time_signature(symbol[6:])
		elif symbol.startswith("metersign"):
			if symbol == "metersign.C-L3":
				score.time_signature = [4, 4]
			else:
				score.time_signature = [2, 2]
		elif symbol.startswith("clef"):
			score.add_clef(symbol[5:])

	mf = MIDIFile(1)     # only 1 track
	track = 0   # the only track
	time = 0    # start at the beginning
	mf.addTrackName(track, time, "OMR track")
	mf.addTempo(track, time, score.bpm)

	time_sig_denom = 0
	denominator = 1
	while denominator != score.time_signature[1]: # converts 
		denominator  *= 2
		time_sig_denom += 1
		if time_sig_denom > 6: # out of bounds
			time_sig_denom = 2
			break

	mf.addTimeSignature(track, time, score.time_signature[0], time_sig_denom, 24)
	channel = 0
	volume = 100

	if score.first_bar_duration < score.time_signature[0]: # anacrusis
		time = score.time_signature[0] - score.first_bar_duration
	else:
		time = score.first_bar_duration % score.time_signature[0]

	if score.key_signature[0] != score.Accidental.nothing:
		if score.key_signature[0] == score.Accidental.sharp:
			mf.addKeySignature(track, 0, score.key_signature[1], SHARPS, MAJOR)
		else:
			mf.addKeySignature(track, 0, score.key_signature[1], FLATS, MAJOR)
	for note in score.notes:
		if note.pitch != 0:
			mf.addNote(track, channel, note.pitch, time, note.duration, volume)
		time += note.duration

	# write it to disk
	with open("output.mid", 'wb') as outf:
	    mf.writeFile(outf)

#--------------------------------------------DECORATORS----------------------------------------------

def requires_grayscale(f):
	def wrapper(*args, **kwargs):
		img_data = args[0]
		if img_data.ndim >= 3:
			raise Exception("function {} requires a grayscale image.".format(f.__name__))
		return f(*args, **kwargs)
	return wrapper


def requires_binary(f):
	def wrapper(*args, **kwargs):
		img_data = args[0]
		for intensity in np.unique(img_data):
			if intensity not in [0, 255]:
				raise Exception("function {} requires a binary image.".format(f.__name__))
		return f(*args, **kwargs)
	return requires_grayscale(wrapper)

#--------------------------------------------FUNCTIONS--------------------------------------------
@requires_grayscale
def align_staff(img_data, bin_method='sauvola'):
	best_image = None
	maximum = 0
	for i in range(-40,40):
		new_image = util.img_as_ubyte(transform.rotate(img_data, 0.25*i, mode='wrap'))
		binary_image = apply_threshold(new_image, bin_method)
		horizontal_projection = get_projection(binary_image)
		max_projection = max(horizontal_projection)
		if max_projection > maximum:
			best_image = new_image
			maximum = max_projection
	return best_image

@requires_grayscale
def median_filter(img_data, radius=1):
	filtered_img = filters.median(img_data, disk(radius))
	return filtered_img

@requires_grayscale
def mean_bilateral_filter(img_data, radius=1):
	filtered_img = filters.rank.mean_bilateral(img_data, disk(radius), s0=10, s1=10)
	filtered_img = util.img_as_ubyte(filtered_img)
	return filtered_img

@requires_grayscale
def wiener_filter(img_data, psf=np.ones((5, 5))/5):
	filtered_img, _ = restoration.unsupervised_wiener(img_data, psf)
	return filtered_img

@requires_grayscale
def apply_threshold(img_data, method):
	window_size = 101
	methods_dict = {
		'otsu': (filters.threshold_otsu, {}),
		'niblack': (filters.threshold_niblack, {'window_size':window_size, 'k':0.8}),
		'sauvola': (filters.threshold_sauvola, {'window_size':window_size}),
		'isodata': (filters.threshold_isodata, {}),
		'local_otsu': (filters.rank.otsu, {'selem': disk(10)})
	}
	if methods_dict.get(method) is not None:
		threshold_func = methods_dict[method][0]
		kwargs = methods_dict[method][1]
		threshold = threshold_func(img_data,**kwargs)
		return util.img_as_ubyte(img_data > threshold)
	else:
		raise Exception("binarization method '{}' is not supported.".format(method))

@requires_binary
def get_projection(img_data, col_row=1):
	mask = np.uint8(np.where(img_data == 0, 1, 0))
	count = cv2.reduce(mask, col_row, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
	return count

@requires_binary
def get_reference_lenghts(img_data):
	black_runs = {}
	white_runs = {}
	for i in range(img_data.shape[1]): # for all columns
		col = img_data[:,i]
		for k, g in itertools.groupby(col):
			length = len(list(g))
			if k == 0:
				black_runs[length] = black_runs.get(length, 0) + 1
			else:
				white_runs[length] = white_runs.get(length, 0) + 1
	line_thickness = max(black_runs, key=black_runs.get)
	staff_space = max(white_runs, key=white_runs.get)
	return(line_thickness, staff_space)

@requires_binary
def detect_staves_morph_op(img_data):
	eroded = erosion(img_data, selem=np.ones((20,1)))
	dilated = dilation(eroded, selem=np.ones((1,20)))
	return dilated

@requires_binary
def apply_closing(img_data, size=2):
	selem = disk(size)
	closed = util.invert(img_data)
	closed = closing(closed, selem)
	closed = util.invert(closed)
	return closed

@requires_binary
def get_connected_components(img_data):
	labels = measure.label(img_data, background=1)
	return np.max(labels)
	

#--------------------------------------------UTILS--------------------------------------------
def get_image_diff(img_data1, img_data2):
	if img_data1.shape != img_data2.shape:
		raise Exception("image_diff requires two images with the same dimensions")

	img_diff = np.zeros(shape=(img_data1.shape[0], img_data1.shape[1], 3), dtype=np.uint8)

	for i in range(img_data1.shape[0]):
		for j in range(img_data1.shape[1]):
			if img_data1[i][j] == img_data2[i][j]:
				if img_data1[i][j] != 0:
					img_diff[i][j] = np.array([122, 122, 122])
				else:
					img_diff[i][j] = np.array([0, 0, 0])
			else:
				if img_data1[i][j] != 0:
					img_diff[i][j] = np.array([0, 255, 0])
				else:
					img_diff[i][j] = np.array([255, 0, 0])
	return img_diff

@requires_binary
def horizontal_projection_img(img_data):
	pixel_list = get_projection(img_data)
	img = np.empty(img_data.shape)
	n_cols = img.shape[1]
	for i in range(img.shape[0]):
		npixels = pixel_list[i]
		black_array = np.zeros(npixels)
		white_array = np.full(n_cols-npixels, 255)
		array = np.append(black_array, white_array)
		img[i] = array
	return img

def plot_images(img_list, title_list, show=False, save=False, dest=None, img_name='image'):
	fig, axs = plt.subplots(len(img_list), figsize=(20, 10))
	fig.tight_layout(pad=1.0)
	for i, (image, title) in enumerate(zip(img_list,title_list)):
		if image.ndim == 3:
			fig = axs[i].imshow(image)
		else:
			fig = axs[i].imshow(image, cmap=plt.cm.gray)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		axs[i].set_title(title)
	if show:
		plt.show()
	if save:
		if dest is not None:
			plt.savefig(os.path.join(dest,img_name))
		else:
			raise Exception("cannot save to an empty destination.")
	plt.close()

def __try_all_thresholds(img_data):
	if img_data.ndim == 3:
		gray_img_data = rgb2gray(img_data)
	else:
		gray_img_data = img_data
	otsu_img = apply_threshold(gray_img_data, 'otsu')
	#isodata_img = apply_threshold(gray_img_data, 'isodata')
	niblack_img = apply_threshold(gray_img_data, 'niblack')
	sauvola_img = apply_threshold(gray_img_data, 'sauvola')
	local_otsu_img = apply_threshold(gray_img_data, 'local_otsu')

	img_list = [img_data, otsu_img, niblack_img, sauvola_img, local_otsu_img]
	title_list = ['original', 'otsu', 'niblack', 'sauvola', 'local_otsu']
	return (img_list, title_list)

def __try_all_noise_removal(img_data):
	if img_data.ndim == 3:
		gray_img_data = rgb2gray(img_data)
	else:
		gray_img_data = img_data
	img1 = median_filter(gray_img_data, 2)
	bi10_img = mean_bilateral_filter(gray_img_data, 15)

	img_list = [img_data, img1, bi10_img]
	title_list = ['original', 'median1', 'bilateral10']
	return (img_list, title_list)

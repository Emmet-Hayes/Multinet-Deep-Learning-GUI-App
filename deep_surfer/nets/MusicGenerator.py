import os
import tensorflow as tf
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from scipy.io import wavfile
from scipy import signal
import soundfile #convert 24 bit wav to 16 bit
from pydub import AudioSegment #
from pydub.playback import play as dubPlay #play wav files
import pygame #play midi files
import numpy as np #cool math stuff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt #plot spectrograms and midi files
#import magenta #magical music library headed by google
#import magenta.music as mm
#from magenta.music.sequences_lib import concatenate_sequences
#from magenta.models.melody_rnn import melody_rnn_sequence_generator
#from magenta.models.performance_rnn import performance_sequence_generator
#from magenta.protobuf import generator_pb2
#from magenta.protobuf import music_pb2
#from magenta.models.music_vae import configs
#from magenta.models.music_vae.trained_model import TrainedModel

from deep_surfer.nets.MidiFileVisual import MidiFile #makes it plottable easily

#THIS IS OUR MIXER RIGHT HERE BABY YEAH AUDIO
freq = 44100    # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2    # 1 is mono, 2 is stereo
buffer = 1024    # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)


class MusicGenerator:
  LAST_MUSIC_FILE='deep_surfer/music/performance0.8.mid'
  '''
  @staticmethod
  def make_note_sequence():
    note_sequence = music_pb2.NoteSequence()
    note_sequence.notes.add(pitch=36, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=38, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=46, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=0.25, end_time=0.375, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=0.375, end_time=0.5, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=0.5, end_time=0.625, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=50, start_time=0.5, end_time=0.625, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=36, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=38, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=45, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=36, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=46, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=42, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=48, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
    note_sequence.notes.add(pitch=50, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
    note_sequence.total_time = 1.375
    note_sequence.tempos.add(qpm=60)
    mm.sequence_proto_to_midi_file(note_sequence, 'deep_surfer/music/note_sequence.mid')
    return note_sequence

  @staticmethod
  def continue_melodyrnn(midi_file='default', num_steps=128, temperature=1.0):
    mm.notebook_utils.download_bundle('basic_rnn.mag', 'magentarnn')
    bundle = mm.sequence_generator_bundle.read_bundle_file('deep_surfer/magentarnn/basic_rnn.mag')
    generator_map = melody_rnn_sequence_generator.get_generator_map()
    melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
    melody_rnn.initialize()
    print('melodyrnn is initialized\n')
    # Model options. Change these to get different generated sequences! 
    if midi_file is 'default':
      input_sequence = MusicGenerator.make_note_sequence() # change this to teapot if you want
    else: input_sequence = mm.midi_to_sequence_proto(midi_file)
    # Set the start time to begin on the next step after the last note ends.
    last_end_time = (max(n.end_time for n in input_sequence.notes)
                      if input_sequence.notes else 0)
    qpm = input_sequence.tempos[0].qpm 
    seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter
    total_seconds = num_steps * seconds_per_step

    generator_options = generator_pb2.GeneratorOptions()
    generator_options.args['temperature'].float_value = temperature
    generate_section = generator_options.generate_sections.add(
      start_time=last_end_time + seconds_per_step,
      end_time=total_seconds)
    # Ask the model to continue the sequence.
    continued_sequence = melody_rnn.generate(input_sequence, generator_options)
    mm.sequence_proto_to_midi_file(continued_sequence, 'deep_surfer/music/continued_sequence.mid')
    MusicGenerator.LAST_MUSIC_FILE = 'deep_surfer/music/continued_sequence.mid'
    print("Done with melodyrnn\n")

  @staticmethod
  def generate_musicvae(checkpoint='deep_surfer/musicvae/cat-drums_2bar_small.lokl.ckpt',
    config_map='cat-drums_2bar_small', batch_size=4, n=2, length=80, temperature=1.0):
    print("Initializing Music VAE...")
    music_vae = TrainedModel( configs.CONFIG_MAP[config_map], 
          batch_size=batch_size, checkpoint_dir_or_path='deep_surfer/musicvae/cat-drums_2bar_small.lokl.ckpt')
    generated_sequences = music_vae.sample(n=n, length=length, temperature=temperature)
    for idx, ns in enumerate(generated_sequences):
      mm.sequence_proto_to_midi_file(ns, 'deep_surfer/music/generated_sequence' + str(idx) + '.mid')
      MusicGenerator.LAST_MUSIC_FILE = 'deep_surfer/music/generated_sequence' + str(idx) + '.mid'
    print("Done with vae\n")

  @staticmethod
  def random_multitrackvae(BATCH_SIZE=4, Z_SIZE=256, TOTAL_STEPS=512, 
    BAR_SECONDS=2.0, CHORD_DEPTH=49, SAMPLE_RATE=44100):
    
    def slerp(p0, p1, t): # Spherical linear interpolation.
      omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
      so = np.sin(omega)
      return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1
    
    def chord_encoding(chord): # Chord encoding tensor.
      index = mm.TriadChordOneHotEncoding().encode_event(chord)
      c = np.zeros([TOTAL_STEPS, CHORD_DEPTH])
      c[0,0] = 1.0
      c[1:,index] = 1.0
      return c

    def trim_sequences(seqs, num_seconds=BAR_SECONDS): # Trim sequences to exactly one bar.
      for i in range(len(seqs)):
        seqs[i] = mm.extract_subsequence(seqs[i], 0.0, num_seconds)
        seqs[i].total_time = num_seconds
    
    def fix_instruments_for_concatenation(note_sequences): # Consolidate instrument numbers by MIDI program.
      instruments = {}
      for i in range(len(note_sequences)):
        for note in note_sequences[i].notes:
          if not note.is_drum:
            if note.program not in instruments:
              if len(instruments) >= 8:
                instruments[note.program] = len(instruments) + 2
              else:
                instruments[note.program] = len(instruments) + 1
            note.instrument = instruments[note.program]
          else:
            note.instrument = 9
    config = configs.CONFIG_MAP['cat-drums_2bar_small']
    model = TrainedModel(
        config, batch_size=BATCH_SIZE,
        checkpoint_dir_or_path='./deep_surfer/musicvae/cat-drums_2bar_small.lokl.ckpt')
    model._config.data_converter._max_tensors_per_input = None
    temperature = 0.2 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
    seqs = model.sample(n=BATCH_SIZE, length=TOTAL_STEPS, temperature=temperature)
    trim_sequences(seqs)
    i = 0
    for seq in seqs:
      mm.sequence_proto_to_midi_file(seq, "./deep_surfer/music/random_seq" + str(i) + ".mid")
      i += 1
    #@title Interpolation Between Random Samples
    num_bars = 32 #@param {type:"slider", min:4, max:64, step:1}
    temperature = 0.3 #@param {type:"slider", min:0.01, max:1.5, step:0.01}
    z1 = np.random.normal(size=[Z_SIZE])
    z2 = np.random.normal(size=[Z_SIZE])
    z = np.array([slerp(z1, z2, t)
                  for t in np.linspace(0, 1, num_bars)])
    seqs = model.decode(length=TOTAL_STEPS, z=z, temperature=temperature)
    trim_sequences(seqs)
    fix_instruments_for_concatenation(seqs)
    interp_ns = concatenate_sequences(seqs)
    mm.sequence_proto_to_midi_file(interp_ns, "./deep_surfer/music/random_interpolation.mid")
    MusicGenerator.LAST_MUSIC_FILE = "./deep_surfer/music/random_interpolation.mid"
    print("done with random multitrack interpolation vae\n")

  @staticmethod
  def performance_rnn(temperature=1.0):
    BUNDLE_DIR = './deep_surfer/magentarnn/'
    MODEL_NAME = 'performance_with_dynamics'
    BUNDLE_NAME = MODEL_NAME + '.mag'
    mm.notebook_utils.download_bundle(BUNDLE_NAME, BUNDLE_DIR)
    bundle = mm.sequence_generator_bundle.read_bundle_file(os.path.join(BUNDLE_DIR, BUNDLE_NAME))
    generator_map = performance_sequence_generator.get_generator_map()
    generator = generator_map[MODEL_NAME](checkpoint=None, bundle=bundle)
    generator.initialize()
    generator_options = generator_pb2.GeneratorOptions()
    generator_options.args['temperature'].float_value = temperature  # Higher is more random; 1.0 is default. 
    generate_section = generator_options.generate_sections.add(start_time=0, end_time=30)
    sequence = generator.generate(music_pb2.NoteSequence(), generator_options)
    mm.sequence_proto_to_midi_file(sequence, "./deep_surfer/music/performance.mid")
    MusicGenerator.LAST_MUSIC_FILE = "./deep_surfer/music/performance.mid"
    print("Done with performance rnn\n")
  '''
  @staticmethod
  def wav_to_spectrogram(notepadwidget):
    wav_tuple = QFileDialog.getOpenFileName(notepadwidget, "Open wav file", 
      os.getenv('HOME'), "Audio (*.wav)")
    wav_file = wav_tuple[0]
    data, samplerate = soundfile.read(wav_file[:-4] + '.wav') #converts to 16 bit wav
    soundfile.write(wav_file[:-4] + '16bit.wav', data, samplerate, subtype='PCM_16' )
    sound = AudioSegment.from_wav(wav_file[:-4] + '16bit.wav')
    sound = sound.set_channels(1) # sets it to mono
    sound.export(wav_file[:-4] + '16bitmono.wav', format="wav")
    sample_rate, samples = wavfile.read(wav_file[:-4] + '16bitmono.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.show()

  @staticmethod
  def play_music(music_file):
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print("Music file %s loaded!" % music_file)
    except pygame.error:
        print("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    pygame.mixer.music.play()

  @staticmethod
  def play_wav(wav_file):
    print("in it\n")
    sound = AudioSegment.from_wav(wav_file)
    dubPlay(sound)
    print("did it?")

  @staticmethod
  def play_last_generated():
    MusicGenerator.play_music(MusicGenerator.LAST_MUSIC_FILE)

  @staticmethod
  def visual_last_generated():
    midiRoll = MidiFile(MusicGenerator.LAST_MUSIC_FILE)
    midiRoll.drawRoll()

if __name__ == '__main__':
  #print('Magenta version: ' + magenta.__version__)
  print('Tensorflow version: ' + tf.__version__)
  print('Testing the functions of the generator here...')
  #MusicGenerator.continue_melodyrnn()
  #MusicGenerator.generate_musicvae()
  #MusicGenerator.random_multitrackvae()
  #MusicGenerator.performance_rnn()
  print('if you are reading this, everything worked correctly.\n')

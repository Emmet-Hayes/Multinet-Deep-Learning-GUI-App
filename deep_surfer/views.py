from django.shortcuts import render
from django.http import HttpResponse
from deep_surfer.nets.ImageClassifier import ImageClassifier
from deep_surfer.nets.ImageGenerator import ImageGenerator
from deep_surfer.nets.DeepDream import DeepDream
from deep_surfer.nets.TextGenerator import TextGenerator
from django.core.files.storage import FileSystemStorage
import platform

#this dict will holds all element to be looked up in the templates(html)
PARAMS = {'tg_epochs':15, 'tg_generate':400, 'tg_temper':1.0, 'tg_seq':40,
    'ic_steps':4000, 'ic_lrate':0.01, 'ic_flip':False, 'ic_random':False,
    'ig_channels':3, 'ig_batch':64, 'ig_epochs':1000, 'ig_rdims':100, 
    'ig_lrate':2e-4, 'ig_weights':0.01, 'ig_diters':5, 'ig_giters':1,
    'ig_crate':500, 'ig_srate':50, 'dd_layer':'mixed4b', 'dd_render':10,
    'dd_octave':4, 'dd_scaled':1.4}

def games(request):
  return render(request, 'deep_surfer/games.html')

def chess(request):
  return render(request, 'deep_surfer/chess.html')

def twentyfourtyeight(request):
  return render(request, 'deep_surfer/2048.html') 

def tetris(request):
  return render(request, 'deep_surfer/tetris.html')

def tg(request):
  tg_handler(request)
  return render(request, 'deep_surfer/tg.html', PARAMS)

def tg_handler(request):
  file_list = []
  upfiles = request.FILES.getlist('tgfile')
  fs = FileSystemStorage()
  for i, f in enumerate(upfiles):
    filename = fs.save(f.name, f)
    file_url = fs.url(filename)
    if platform.system() is "Windows":
      file_url = file_url[1:] #remove the / for windows
    file_list.append(file_url)
    PARAMS['tg_file_url_' + str(i)] = file_list[i]

def ic(request):
  return render(request, 'deep_surfer/ic.html')

def dd(request):
  return render(request, 'deep_surfer/dd.html')

def ig(request):
  return render(request, 'deep_surfer/ig.html')

def mg(request):
  return render(request, 'deep_surfer/mg.html')

def trainTG(request):
  try:
    if 'tg_file_url_0'in PARAMS:
      textfile = PARAMS['tg_file_url_0']
      generated_text = TextGenerator.train_text_generator(
        file_path=textfile)
    else:
      generated_text = TextGenerator.train_text_generator()
    PARAMS['tg_train_complete'] = generated_text
  except Exception as e:
    template = "An exception of type {0} occured. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    PARAMS['tg_run_complete'] = message + "\nsorry :("
  return render(request, 'deep_surfer/tgtrain.html', PARAMS)

def runTG(request):
  try:
    if all(k in PARAMS for k in ('tg_file_url_0', 'tg_file_url_1')):
      textfile = PARAMS['tg_file_url_0']
      modelfile = PARAMS['tg_file_url_1']
      generated_text = TextGenerator.run_text_generator(
        file_path=textfile, ckpt_path=modelfile)
    elif 'tg_file_url_0' in PARAMS:
      textfile = PARAMS['tg_file_url_0']
      generated_text = TextGenerator.run_text_generator(
        file_path=textfile)
    else:
      print("in this one")
      generated_text = TextGenerator.run_text_generator()
    PARAMS['tg_run_complete'] = generated_text
  except ValueError as ve:
    PARAMS['tg_run_complete'] = 'This brain doesn\'t work with this text :(\n' + str(ve.args)
  except Exception as e:
    template = "An exception of type {0} occured. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    PARAMS['tg_run_complete'] = message + "\nsorry :("
  return render(request, 'deep_surfer/tgrun.html', PARAMS)

def openIC(request):
  ImageClassifier.run_image_classifier(self, False)
  return render(request, 'deep_surfer/ic.html', PARAMS)

def openICAlt(request):
  ImageClassifier.run_image_classifier(self, True)
  return render(request, 'deep_surfer/netparams.html', PARAMS)

def trainIC(request):
  ImageClassifier.retrain_image_classifier(retrain_image_dir, self, True, training_steps, 
      learn_rate, print_misclass, flip_l_r, rnd_crop, rnd_scale, rnd_brightness)
  return render(request, 'deep_surfer/netparams.html', PARAMS)

def trainIG(request):    
  ImageGenerator.train(self, height, width, channel, batch_size, epoch, random_dim, learn_rate, 
      clip_weights, d_iters, g_iters, save_ckpt_rate, save_img_rate)
  return render(request, 'deep_surfer/netparams.html', PARAMS)

def genDD(request):
  DeepDream.run(self, dream_layer, naive_render_iter, naive_step,
      deep_render_iter, deep_step, octave_number, octave_scaled, downsize, img_noise_size, 
      imagenet_mean_init, grad_tile_size, strip_const_size)  
  return render(request, 'deep_surfer/netparams.html', PARAMS)

def index(request):
  return render(request, 'deep_surfer/index.html')
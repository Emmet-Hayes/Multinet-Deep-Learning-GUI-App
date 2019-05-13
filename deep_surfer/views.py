from django.shortcuts import render
from django.http import HttpResponse
from deep_surfer.nets.ImageClassifier import ImageClassifier
from deep_surfer.nets.ImageGenerator import ImageGenerator
from deep_surfer.nets.DeepDream import DeepDream
from deep_surfer.nets.TextGenerator import TextGenerator
from django.core.files.storage import FileSystemStorage
import platform
import os
from django.db import models

#this dict will holds all element to be looked up in the templates(html)
PARAMS = {'tg_epochs':15, 'tg_generate':400, 'tg_temper':1.0, 'tg_seq':40,
    'ic_steps':4000, 'ic_lrate':0.01, 'ic_flip':False, 'ic_random':False,
    'ig_channels':3, 'ig_batch':64, 'ig_epochs':1000, 'ig_rdims':100, 
    'ig_lrate':2e-4, 'ig_weights':0.01, 'ig_diters':5, 'ig_giters':1,
    'ig_crate':500, 'ig_srate':50, 'dd_layer':'mixed4b', 'dd_render':10,
    'dd_octave':4, 'dd_scaled':1.4}

def param_handler(request):
  if request.method == "POST":
    if request.POST.get('tg_epochs'):
      PARAMS['tg_epochs'] = request.POST.get('tg_epochs')
    if request.POST.get('tg_generate'):
      PARAMS['tg_generate'] = request.POST.get('tg_generate')
    if request.POST.get('tg_temper'):
      PARAMS['tg_temper'] = request.POST.get('tg_temper')
    if request.POST.get('tg_seq'):
      PARAMS['tg_seq'] = request.POST.get('tg_seq')
    if request.POST.get('dd_layer'):
      PARAMS['dd_layer'] = request.POST.get('dd_layer')
    if request.POST.get('dd_render'):
      PARAMS['dd_render'] = request.POST.get('dd_render')
    if request.POST.get('dd_octave'):
      PARAMS['dd_octave'] = request.POST.get('dd_octave')
    if request.POST.get('dd_scaled'):
      PARAMS['dd_scaled'] = request.POST.get('dd_scaled')
    if request.POST.get('ic_steps'):
      PARAMS['ic_steps'] = request.POST.get('ic_steps')
    if request.POST.get('ic_lrate'):
      PARAMS['ic_lrate'] = request.POST.get('ic_lrate')
    if request.POST.get('ic_flip'):
      PARAMS['ic_flip'] = request.POST.get('ic_flip')
    if request.POST.get('ic_random'):
      PARAMS['ic_random'] = request.POST.get('ic_random')

def file_handler(request):
  file_list = []
  upfiles = request.FILES.getlist('aifile')
  fs = FileSystemStorage()
  for i, f in enumerate(upfiles):
    filename = fs.save(f.name, f)
    file_url = fs.url(filename)[1:]
    file_list.append(file_url)
    PARAMS['file_url_' + str(i)] = file_list[i]
    print('value of file_url_' + str(i) + ': ' + str(PARAMS['file_url_' + str(i)]) + '\n')

def games(request):
  return render(request, 'deep_surfer/games.html')

def chess(request):
  return render(request, 'deep_surfer/chess.html')

def twentyfourtyeight(request):
  return render(request, 'deep_surfer/2048.html') 

def tetris(request):
  return render(request, 'deep_surfer/tetris.html')

def tg(request):
  file_handler(request)
  param_handler(request)
  return render(request, 'deep_surfer/tg.html', PARAMS)

def ic(request):
  file_handler(request),
  param_handler(request)
  return render(request, 'deep_surfer/ic.html', PARAMS)

def dd(request):
  file_handler(request)
  param_handler(request)
  return render(request, 'deep_surfer/dd.html', PARAMS)

def ig(request):
  file_handler(request)
  param_handler(request)
  return render(request, 'deep_surfer/ig.html', PARAMS)

def mg(request):
  file_handler(request)
  param_handler(request)
  return render(request, 'deep_surfer/mg.html', PARAMS)

def trainTG(request):
  file_handler(request)
  param_handler(request)
  try:
    if 'file_url_0'in PARAMS:
      textfile = PARAMS['file_url_0']
      generated_text = TextGenerator.train_text_generator(textfile,
        PARAMS['tg_epochs'], PARAMS['tg_generate'], PARAMS['tg_temper'], seq_length = PARAMS['tg_seq'])
    else:
      generated_text = TextGenerator.train_text_generator(train_epochs=PARAMS['tg_epochs'],
        num_generate=PARAMS['tg_generate'], temperature=PARAMS['tg_temper'], seq_length=PARAMS['tg_seq'])
    PARAMS['tg_train_complete'] = generated_text
  except Exception as e:
    template = "An exception of type {0} occured. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    PARAMS['tg_train_complete'] = message + "\nsorry :("
  return render(request, 'deep_surfer/tg.html', PARAMS)

def runTG(request):
  file_handler(request)
  try:
    if all(k in PARAMS for k in ('file_url_0', 'file_url_1')):
      textfile = PARAMS['file_url_0']
      modelfile = PARAMS['file_url_1']
      generated_text = TextGenerator.run_text_generator(textfile, modelfile,
        PARAMS['tg_generate'], PARAMS['tg_temper'])
    elif 'file_url_0' in PARAMS:
      textfile = PARAMS['file_url_0']
      generated_text = TextGenerator.run_text_generator(textfile,
        num_generate=PARAMS['tg_generate'], temperature=PARAMS['tg_temper'])
    else:
      generated_text = TextGenerator.run_text_generator(
        num_generate=PARAMS['tg_generate'], temperature=PARAMS['tg_temper'])
    PARAMS['tg_run_complete'] = generated_text
  except ValueError as ve:
    PARAMS['tg_run_complete'] = 'This brain doesn\'t work with this text :(\n' + str(ve.args)
  except Exception as e:
    template = "An exception of type {0} occured. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    PARAMS['tg_run_complete'] = message + "\nsorry :("
  return render(request, 'deep_surfer/tg.html', PARAMS)

def genDD(request):
  file_handler(request)
  try:
    if 'file_url_0' in PARAMS:
        imagefile = PARAMS['file_url_0']
        generated_image = DeepDream.run(
          file_path=imagefile)
        PARAMS['dd_run_complete'] = generated_image
    else:
        print("no file found to generate on\n")
        PARAMS['dd_run_failed'] = "no file found to generate on :(\n"
  except Exception as e:
    template = "An exception of type {0} occured. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    PARAMS['dd_run_failed'] = message + "\nsorry :("
  return render(request, 'deep_surfer/dd.html', PARAMS)

def classifyIC(request):
  file_handler(request)
  try:
    if all(k in PARAMS for k in ('file_url_0', 'file_url_1')):
      imagefile = PARAMS['file_url_0']
      modelfile = PARAMS['file_url_1']
      PARAMS['ic_run_complete'] = ImageClassifier.run_image_classifier(file_path=imagefile,
        model_path=modelfile)
    if 'file_url_0' in PARAMS:
      imagefile = PARAMS['file_url_0']
      PARAMS['ic_run_complete'] = ImageClassifier.run_image_classifier(file_path=imagefile)
    else:
      PARAMS['ic_run_failed'] ="no image file found to classify :(\n"
  except Exception as e:
    template = "An exception of type {0} occured. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    PARAMS['ic_run_failed'] = message + "\nsorry :("
  return render(request, 'deep_surfer/ic.html', PARAMS)

def trainIC(request):
  file_handler(request)
  param_handler(request)
  ImageClassifier.retrain_image_classifier(retrain_image_dir, self, True, training_steps, 
      learn_rate, print_misclass, flip_l_r, rnd_crop, rnd_scale, rnd_brightness)
  return render(request, 'deep_surfer/ic.html', PARAMS)

def trainIG(request):    
  ImageGenerator.train(self, height, width, channel, batch_size, epoch, random_dim, learn_rate, 
      clip_weights, d_iters, g_iters, save_ckpt_rate, save_img_rate)
  return render(request, 'deep_surfer/ig.html', PARAMS)

def index(request):
  return render(request, 'deep_surfer/index.html')
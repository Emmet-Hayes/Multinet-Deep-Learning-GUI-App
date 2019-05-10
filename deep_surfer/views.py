from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.urls import reverse
from django.template import Template, Context
from .models import Choice, Question
from deep_surfer.nets.ImageClassifier import ImageClassifier
from deep_surfer.nets.ImageGenerator import ImageGenerator
from deep_surfer.nets.DeepDream import DeepDream
from deep_surfer.nets.TextGenerator import TextGenerator
from django.conf import settings
from django.core.files.storage import FileSystemStorage
#from deep_surfer.nets.MultinetWindow import MultinetWindow
#from PyQt5.QtWidgets import QApplication
import sys

PARAMS = {'tg_epochs':15, 'tg_generate':400, 'tg_temper':1.0, 'tg_seq':40,
    'ic_steps':4000, 'ic_lrate':0.01, 'ic_flip':False, 'ic_random':False,
    'ig_channels':3, 'ig_batch':64, 'ig_epochs':1000, 'ig_rdims':100, 
    'ig_lrate':2e-4, 'ig_weights':0.01, 'ig_diters':5, 'ig_giters':1,
    'ig_crate':500, 'ig_srate':50, 'dd_layer':'mixed4b', 'dd_render':10,
    'dd_octave':4, 'dd_scaled':1.4, 
    'tgtxtfile':'/', 'tgmodfile':'/'}

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'deep_surfer/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'deep_surfer/simple_upload.html')

def mainapp(request):
  return HttpResponse("Thanks for trying out my generator!\n")

def games(request):
  return render(request, 'deep_surfer/games.html')

def chess(request):
  return render(request, 'deep_surfer/chess.html')

def twentyfourtyeight(request):
  return render(request, 'deep_surfer/2048.html') 

def tetris(request):
  return render(request, 'deep_surfer/tetris.html')

def netparams(request):
  dealwithuploads(request)
  return render(request, 'deep_surfer/netparams.html', PARAMS)

def dealwithuploads(request):
  if request.method == 'POST' and request.FILES['tgtxtfile']:
    upfile = request.FILES['tgtxtfile']
    fs = FileSystemStorage()
    filename = fs.save(upfile.name, upfile)
    PARAMS['tgtxt_file_url'] = fs.url(filename)
  elif request.method == 'POST' and request.FILES['tgmodfile']:
    upfile = request.FILES['tgmodfile']
    fs = FileSystemStorage()
    filename = fs.save(upfile.name, upfile)
    PARAMS['tgmod_file_url'] = fs.url(filename)
  '''
  if 'tgtxtfile' in request.POST:
    upfile = request.FILES.get('tgtxtfile')
    fs = FileSystemStorage()
    filename = fs.save(upfile.name, upfile)
    PARAMS['tgtxtfile'] = fs.url(filename)
  if 'tgmodfile' in request.POST:
    upfile = request.FILES.get('tgmodfile')
    fs = FileSystemStorage()
    filename = fs.save(upfile.name, upfile)
    PARAMS['tgmodfile'] = fs.url(filename)
  '''

def trainTG(request):
  return HttpResponse('Text Generator is Done!\n\n' + TextGenerator.train_text_generator())

def runTG(request):
  return HttpResponse('Text Generator is Done!\n\n' + TextGenerator.run_text_generator())
  #return render(request, 'deep_surfer/netparams.html', generatedText)

def openIC(request):
  ImageClassifier.run_image_classifier(self, False)
  return render(request, 'deep_surfer/netparams.html')

def openICAlt(request):
  ImageClassifier.run_image_classifier(self, True)
  return render(request, 'deep_surfer/netparams.html')

def trainIC(request):
  ImageClassifier.retrain_image_classifier(retrain_image_dir, self, True, training_steps, 
      learn_rate, print_misclass, flip_l_r, rnd_crop, rnd_scale, rnd_brightness)
  return render(request, 'deep_surfer/netparams.html')

def trainIG(request):    
  ImageGenerator.train(self, height, width, channel, batch_size, epoch, random_dim, learn_rate, 
      clip_weights, d_iters, g_iters, save_ckpt_rate, save_img_rate)
  return render(request, 'deep_surfer/netparams.html')

def genDD(request):
  DeepDream.run(self, dream_layer, naive_render_iter, naive_step,
      deep_render_iter, deep_step, octave_number, octave_scaled, downsize, img_noise_size, 
      imagenet_mean_init, grad_tile_size, strip_const_size)  
  return render(request, 'deep_surfer/netparams.html')

def index(request):
  latest_question_list = Question.objects.order_by('-pub_date')[:5] #last 5
  context = {'latest_question_list': latest_question_list}
  return render(request, 'deep_surfer/index.html', context)

def detail(request, question_id):
  question = get_object_or_404(Question, pk=question_id)
  return render(request, 'deep_surfer/detail.html', { 'question': question})

def results(request, question_id):
  question = get_object_or_404(Question, pk=question_id)
  return render(request, 'deep_surfer/results.html', {'question': question})

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'deep_surfer/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('deep_surfer:results', args=(question.id,)))
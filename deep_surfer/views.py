from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.urls import reverse
from django.template import Template, Context
from .models import Choice, Question
from deep_surfer.nets.ImageClassifier import ImageClassifier
from deep_surfer.nets.ImageGenerator import ImageGenerator
from deep_surfer.nets.DeepDream import DeepDream
from deep_surfer.nets.TextGenerator import TextGenerator
#from deep_surfer.nets.MultinetWindow import MultinetWindow
#from PyQt5.QtWidgets import QApplication
import sys

# THIS IS WHERE EVERYTHING GETS CALLED ON THE deep_surfer/mainapp/ page!!!!
def mainapp(request):
 # app = QApplication(sys.argv)
#  labelWindow = MultinetWindow()
#  sys.exit(app.exec_())
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
  '''
  fp = open('/deep_surfer/templates/deep_surfer/netparams.html')
  t = Template(fp.read())
  fp.close()
  html = t.render(Context({'num_epochs' : 15, 'num_generate' : 400}))
  return HttpResponse(html) 
  '''
  return render(request, 'deep_surfer/netparams.html', {'num_epochs':15, 'num_generate':400})

def runTG(request):
  TextGenerator.train_text_generator(self, train_epochs, num_generate, temperature,
      trim_text, embedding_dim, step_size, seq_length, BATCH_SIZE)
  return render(request, 'deep_surfer/netparams.html')


def loadTG(request):
  TextGenerator.run_text_generator(self,
        num_generate, temperature, trim_text, embedding_dim,
        seq_length, step_size)
  return render(request, 'deep_surfer/netparams.html')

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
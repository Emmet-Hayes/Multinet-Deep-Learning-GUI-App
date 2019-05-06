from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.urls import reverse
from .models import Choice, Question

from deep_surfer.nets.MultinetWindow import MultinetWindow
from PyQt5.QtWidgets import QApplication
import sys
#
#
#
# THIS IS WHERE EVERYTHING GETS CALLED ON THE deep_surfer/mainapp/ page!!!!
#
#
def mainapp(request):
  app = QApplication(sys.argv)
  labelWindow = MultinetWindow()
  sys.exit(app.exec_())
  return HttpResponse("Thanks for trying out my generator!\n")

def games(request):
  return render(request, 'deep_surfer/games.html')

def chess(request):
  return render(request, 'deep_surfer/chess.html')

def twentyfourtyeight(request):
  return render(request, 'deep_surfer/2048.html') 

def tetris(request):
  return render(request, 'deep_surfer/tetris.html')

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
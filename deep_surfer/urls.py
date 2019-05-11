from django.urls import path #defines a django path object
from django.conf.urls import url
from django.contrib import admin
from . import views
import platform
from django.conf import settings
from django.conf.urls.static import static
'''
  ****path() is passed 4 args.****
  ****  only 2 are required   ****
  route - string containing a URLpattern, Django starts at 1st
  pattern and goes down list, comparing requestURL with match
  view - on a match, calls this view arg func with an 
  HttpRequest object as the 1st arg and captured vals from 
  route from keyword args.
  kwargs - unused for now
  name - names the URL  for reference elsewhere in django code
'''



urlpatterns = [
    path('', views.index, name='index'), #index is view func name
    path('index', views.index, name='index'),
    #url(r'^admin/', admin.site.urls),
    path('trainTG', views.trainTG, name="trainTG"),
    path('runTG', views.runTG, name="runTG"),
    path('openIC', views.openIC, name="openIC"),
    path('openICAlt', views.openICAlt, name="openICAlt"),
    path('trainIC', views.trainIC, name="trainIC"),
    path('trainIG', views.trainIG, name="trainIG"),
    path('genDD', views.genDD, name="genDD"),
    path('games/', views.games, name='games'),
    path('games/chess', views.chess, name='chess'),
    path('games/2048', views.twentyfourtyeight, name='twentyfortyeight'),
    path('games/tetris', views.tetris, name='tetris'),
    path('tg', views.tg, name="tg"),
    path('ic', views.ic, name="ic"),
    path('dd', views.dd, name="dd"),
    path('ig', views.ig, name="ig"),
    path('mg', views.mg, name="mg")
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
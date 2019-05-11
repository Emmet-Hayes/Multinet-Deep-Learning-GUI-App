from django.urls import path #defines a django path object
from django.conf.urls import url
from django.contrib import admin
from . import views #.py
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
#app_name = 'deep_surfer'

base = ''
if platform.system() is "Windows":
  base = '/'
  print(base)
#elif platform.system() is "Linux":
#  base = 'deep_surfer'

urlpatterns = [
    path(base, views.index, name='index'), #index is view func name
    path(base + 'index', views.index, name='index'),
    #url(r'^admin/', admin.site.urls),
    path(base + 'trainTG', views.trainTG, name="trainTG"),
    path(base + 'runTG', views.runTG, name="runTG"),
    path(base + 'openIC', views.openIC, name="openIC"),
    path(base + 'openICAlt', views.openICAlt, name="openICAlt"),
    path(base + 'trainIC', views.trainIC, name="trainIC"),
    path(base + 'trainIG', views.trainIG, name="trainIG"),
    path(base + 'genDD', views.genDD, name="genDD"),
    path(base + 'games/', views.games, name='games'),
    path(base + 'games/chess', views.chess, name='chess'),
    path(base + 'games/2048', views.twentyfourtyeight, name='twentyfortyeight'),
    path(base + 'games/tetris', views.tetris, name='tetris'),
    path(base + 'tg', views.tg, name="tg"),
    path(base + 'ic', views.ic, name="ic"),
    path(base + 'dd', views.dd, name="dd"),
    path(base + 'ig', views.ig, name="ig"),
    path(base + 'mg', views.mg, name="mg")
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
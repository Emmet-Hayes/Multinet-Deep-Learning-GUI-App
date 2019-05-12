from django.contrib import admin
from django.urls import include, path
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
  name - names the URL for reference elsewhere in django code
'''
urlpatterns = [
    path('deep_surfer/', include('deep_surfer.urls')),
    path('media/', include('deep_surfer.urls')),
    path('admin', admin.site.urls),
]
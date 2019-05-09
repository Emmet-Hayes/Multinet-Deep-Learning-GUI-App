import datetime
from django.db import models
from django.utils import timezone

# Create your models here.
class Question(models.Model):
  question_text = models.CharField(max_length=200)
  pub_date = models.DateTimeField('date published')
  def __str__(self): #MAKES IT RETURN AS A STRING NICELY
    return question_text

  def was_published_recently(self):
    return pub_date >= timezone.now() - datetime.timedelta(days=1)

class Choice(models.Model):
  question = models.ForeignKey(Question, on_delete=models.CASCADE)
  choice_text = models.CharField(max_length=200)
  votes = models.IntegerField(default=0)

  def __str__(self):
    return choice_text

class TextGeneratorParameterWeb(models.Model):
    num_epochs=models.IntegerField(default=15)
    num_generate=models.IntegerField(default=800)
    temperature=models.IntegerField(default=1.0)
    trim_text = models.IntegerField(default=1)
    embedding_dim=models.IntegerField(default=128)
    step_size=models.IntegerField(default=3)
    seq_length=models.IntegerField(default=40)
    BATCH_SIZE=models.IntegerField(default=128)

class ImageClassifierParameterWeb(models.Model):
  training_steps= models.IntegerField(default=4000) #IC hyperparameters are initialized here
  #learn_rate=models.DecimalField(default=0.01)
  learn_rate=models.FloatField(default=0.01)
  print_misclass=models.BooleanField(default=False)
  flip_l_r=models.BooleanField(default=False)
  rnd_crop=models.BooleanField(default=False)
  rnd_scale=models.BooleanField(default=False)
  rnd_brightness=models.BooleanField(default=False)

class DeepDreamParameterWeb(models.Model):
  dream_layer=models.CharField(default="mixed3a", max_length=9)
  naive_render_iter=models.IntegerField(default=20)
  naive_step=models.FloatField(default=1.0)
  deep_render_iter=models.IntegerField(default=10)
  deep_step = models.FloatField(default=1.5) 
  octave_number=models.IntegerField(default=4)
  octave_scaled=models.FloatField(default=1.4)
  downsize=models.FloatField(default=255.0)
  img_noise_size=models.IntegerField(default=224) 
  imagenet_mean_init=models.FloatField(default=117.0)
  grad_tile_size=models.IntegerField(default=256)
  strip_const_size=models.IntegerField(default=32)

class ImageGeneratorParameterWeb(models.Model):
    resize_side_length=models.IntegerField(default=256)
    height=models.IntegerField(default=128)
    width=models.IntegerField(default=128)
    channels=models.IntegerField(default=3)
    batch_size=models.IntegerField(default=64)
    epochs=models.IntegerField(default=1000)
    random_dim=models.IntegerField(default=100)
    #learn_rate=models.DecimalField(default=2e-4)
    learn_rate=models.FloatField(default=2e-4)
    clip_weights=models.FloatField(default=0.01)
    d_iters=models.IntegerField(default=5)
    g_iters=models.IntegerField(default=1)
    save_ckpt_rate=models.IntegerField(default=500)
    save_img_rate=models.IntegerField(default=50)
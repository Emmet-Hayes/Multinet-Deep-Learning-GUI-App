'''
'''

#from PyQt5.QtGui import QIcon #GUI importz bb
#from PyQt5.QtWidgets import QWidget, QPushButton, QInputDialog, QFormLayout, QLineEdit
#from PyQt5.QtWebKit import QWebView

"""
class ImageClassifierParameterDialog(QWidget): #gui hyperparameters for image classifier retrainer
  def __init__(self, parent = None):
    super(ImageClassifierParameterDialog, self).__init__(parent)
    self.training_steps=4000 #IC hyperparameters are initialized here
    self.learn_rate=0.01 
    self.print_misclass=False
    self.flip_l_r=False
    self.rnd_crop=0 
    self.rnd_scale=0
    self.rnd_brightness=0
    self.setWindowTitle("Image Classifier Parameters")
    self.setWindowIcon(QIcon('deep_surfer/icons/cnn-icon.png'))
    layout = QFormLayout()
    self.btnTS = QPushButton("Training Steps") #each line of param in gui is 6 lines of code
    self.btnTS.clicked.connect(self.setTrainSteps)
    self.leTS = QLineEdit()
    self.leTS.setText(str(self.training_steps))
    self.leTS.setReadOnly(True)
    layout.addRow(self.btnTS,self.leTS)
    self.btnLR = QPushButton("Learning Rate")
    self.btnLR.clicked.connect(self.setLearnRate)
    self.leLR = QLineEdit()
    self.leLR.setText(str(self.learn_rate))
    self.leLR.setReadOnly(True)
    layout.addRow(self.btnLR,self.leLR)
    self.btnMC = QPushButton("Show Misclassified Images")
    self.btnMC.clicked.connect(self.setMisclass)
    self.leMC = QLineEdit()
    self.leMC.setText("No")
    self.leMC.setReadOnly(True)
    layout.addRow(self.btnMC,self.leMC)
    self.btnFL = QPushButton("Flip Left and Right")
    self.btnFL.clicked.connect(self.setFlipLR)
    self.leFL = QLineEdit()
    self.leFL.setText("No")
    self.leFL.setReadOnly(True)
    layout.addRow(self.btnFL,self.leFL)
    self.btnRC = QPushButton("Random Crop")
    self.btnRC.clicked.connect(self.setRndCrop)
    self.leRC = QLineEdit()
    self.leRC.setText("No")
    self.leRC.setReadOnly(True)
    layout.addRow(self.btnRC,self.leRC)
    self.btnRS = QPushButton("Random Scale")
    self.btnRS.clicked.connect(self.setRndScale)
    self.leRS = QLineEdit()
    self.leRS.setText("No")
    self.leRS.setReadOnly(True)
    layout.addRow(self.btnRS,self.leRS)
    self.btnRB = QPushButton("Random Brightness")
    self.btnRB.clicked.connect(self.setRndBrightness)
    self.leRB = QLineEdit()
    self.leRB.setText("No")
    self.leRB.setReadOnly(True)
    layout.addRow(self.btnRB,self.leRB)
    self.setLayout(layout) #set the layout to the window after you set it up

  def setTrainSteps(self): #gui collects total training steps from user input
    num,ok = QInputDialog.getInt(self,"Training Steps",
      "Enter a Positive Integer (Don't do 20000 steps if you don't have a GPU for 4 hours to spare.)")
    if ok:
      self.training_steps=num
      self.leTS.setText(str(num))

  def setLearnRate(self): #all these set functions are the same except for type of input
    num,ok = QInputDialog.getDouble(self,"Learning Rate",
      "Enter a Positive Decimal Number")
    if ok: 
      self.learning_rate=num
      self.leLR.setText(str(num))

  def setMisclass(self):
    items = ("Yes", "No")
    item, ok = QInputDialog.getItem(self, "Misclassified Test Images", 
      "Would you like the network to print misclassified test images?", items, 0, False)
    if ok and item: 
      if item is "Yes": self.print_misclass=True
      elif item is "No": self.print_misclass=False
      else: return
      self.leMC.setText(str(item))

  def setFlipLR(self):
    items = ("Yes", "No")
    item, ok = QInputDialog.getItem(self, "Flip Left and Right", 
       "Would you like the network to flip images left to right?", items, 0, False)
    if ok and item: 
      if item is "Yes": self.flip_l_r=True
      elif item is "No": self.flip_l_r=False
      else: return 
      self.leFL.setText(str(item)) 

  def setRndCrop(self):
    items = ("Yes", "No")
    item, ok = QInputDialog.getItem(self, "Random Cropping", 
       "Would you like the network to randomly crop test images?", items, 0, False)
    if ok and item: 
      if item is "Yes": self.rnd_crop=True
      elif item is "No": self.rnd_crop=False
      else: return 
      self.leRC.setText(str(item)) 

  def setRndScale(self):
    items = ("Yes", "No")
    item, ok = QInputDialog.getItem(self, "Random Scale", 
       "Would you like the network to randomly scale test images?", items, 0, False)
    if ok and item:
      if item is "Yes": self.rnd_scale=True
      elif item is "No": self.rnd_scale=False
      else: return 
      self.leRS.setText(str(item)) 

  def setRndBrightness(self):
    items = ("Yes", "No")
    item, ok = QInputDialog.getItem(self, "Random Brightness", 
       "Would you like the network to randomly brighten test images?", items, 0, False)
    if ok and item: 
      if item is "Yes": self.rnd_brightness=True
      elif item is "No": self.rnd_brightness=False
      else: return 
      self.leRB.setText(str(item))

class DeepDreamParameterDialog(QWidget): #gui hyperparameters for deep dream image processing
  def __init__(self, parent = None):
    super(DeepDreamParameterDialog, self).__init__(parent)
    self.dream_layer='mixed4c'
    self.naive_render_iter=20
    self.naive_step=1.0
    self.deep_render_iter=10
    self.deep_step = 1.5 
    self.octave_number=4
    self.octave_scaled=1.4
    self.downsize=255.0
    self.img_noise_size=224 
    self.imagenet_mean_init=117.0
    self.grad_tile_size=256
    self.strip_const_size=32
    self.setWindowTitle("Deep Dream Parameters")
    self.setWindowIcon(QIcon('deep_surfer/icons/deep-dream-icon.png'))
    layout = QFormLayout()
    self.btnDL = QPushButton("Dream Layer")
    self.btnDL.clicked.connect(self.setDreamLayer)
    self.leDL = QLineEdit()
    self.leDL.setText(str(self.dream_layer))
    self.leDL.setReadOnly(True)
    layout.addRow(self.btnDL,self.leDL)
    self.btnNRI = QPushButton("Naive Render Iterator")
    self.btnNRI.clicked.connect(self.setNRI)
    self.leNRI = QLineEdit()
    self.leNRI.setText(str(self.naive_render_iter))
    self.leNRI.setReadOnly(True)
    layout.addRow(self.btnNRI,self.leNRI)
    self.btnNS = QPushButton("Naive Step")
    self.btnNS.clicked.connect(self.setNaiveStep)
    self.leNS = QLineEdit()
    self.leNS.setText(str(self.naive_step))
    self.leNS.setReadOnly(True)
    layout.addRow(self.btnNS,self.leNS)
    self.btnDRI = QPushButton("Deep Render Iterator")
    self.btnDRI.clicked.connect(self.setDRI)
    self.leDRI = QLineEdit()
    self.leDRI.setText(str(self.deep_render_iter))
    self.leDRI.setReadOnly(True)
    layout.addRow(self.btnDRI,self.leDRI)
    self.btnDS = QPushButton("Deep Step")
    self.btnDS.clicked.connect(self.setDeepStep)
    self.leDS = QLineEdit()
    self.leDS.setText(str(self.deep_step))
    self.leDS.setReadOnly(True)
    layout.addRow(self.btnDS,self.leDS)
    self.btnON = QPushButton("Octave Number")
    self.btnON.clicked.connect(self.setOctaveNumber)
    self.leON = QLineEdit()
    self.leON.setText(str(self.octave_number))
    self.leON.setReadOnly(True)
    layout.addRow(self.btnON,self.leON)
    self.btnOS = QPushButton("Octave Scale")
    self.btnOS.clicked.connect(self.setOctaveScale)
    self.leOS = QLineEdit()
    self.leOS.setText(str(self.octave_scaled))
    self.leOS.setReadOnly(True)
    layout.addRow(self.btnOS, self.leOS)
    self.btnDZ = QPushButton("Downsize")
    self.btnDZ.clicked.connect(self.setDownsize)
    self.leDZ = QLineEdit()
    self.leDZ.setText(str(self.downsize))
    self.leDZ.setReadOnly(True)
    layout.addRow(self.btnDZ,self.leDZ)
    self.btnIN = QPushButton("Image Noise")
    self.btnIN.clicked.connect(self.setImageNoise)
    self.leIN = QLineEdit()
    self.leIN.setText(str(self.img_noise_size))
    self.leIN.setReadOnly(True)
    layout.addRow(self.btnIN,self.leIN)
    self.btnIM = QPushButton("Image Net Mean")
    self.btnIM.clicked.connect(self.setImagenetMean)
    self.leIM = QLineEdit()
    self.leIM.setText(str(self.imagenet_mean_init))
    self.leIM.setReadOnly(True)
    layout.addRow(self.btnIM,self.leIM)
    self.btnGT = QPushButton("Grad Tile Size")
    self.btnGT.clicked.connect(self.setGradTileSize)
    self.leGT = QLineEdit()
    self.leGT.setText(str(self.grad_tile_size))
    self.leGT.setReadOnly(True)
    layout.addRow(self.btnGT,self.leGT)
    self.btnSC = QPushButton("Strip Const Size")
    self.btnSC.clicked.connect(self.setStripConstSize)
    self.leSC = QLineEdit()
    self.leSC.setText(str(self.strip_const_size))
    self.leSC.setReadOnly(True)
    layout.addRow(self.btnSC,self.leSC)
    self.setLayout(layout)

  def setDreamLayer(self):
    dream_layers = ("mixed4c", "conv2d0", "conv2d1", "conv2d2", "mixed3a", "mixed3b", "mixed4a",
     "mixed4b", "mixed4d", "mixed4e", "mixed5a", "mixed5b", "maxpool4")
    item,ok = QInputDialog.getItem(self,"Dream Layer",
      "Enter the dream layer that you want to dream on.", dream_layers, 0, False)
    if ok and item:
      self.dream_layer = item
      self.leDL.setText(item)

  def setNRI(self):
    num,ok = QInputDialog.getInt(self,"Naive Render Iterations",
      "Enter the amount of iterations you want to render in the naive state.", 20)
    if ok: 
      self.naive_render_iter=num
      self.leNRI.setText(str(num))

  def setNaiveStep(self):
    num,ok = QInputDialog.getDouble(self,"Naive Step",
      "Enter the step size of each iteration in the naive state.", 1.0)
    if ok: 
      self.naive_step=num
      self.leNS.setText(str(num))

  def setDRI(self):
    num,ok = QInputDialog.getInt(self,"Deep Dream Render Iterations",
      "Enter the amount of iterations you want to render in the deep dream state.", 10)
    if ok: 
      self.deep_render_iter=num
      self.leDRI.setText(str(num))

  def setDeepStep(self):
    num,ok = QInputDialog.getDouble(self,"Deep Step",
      "Enter the step size of each iteration in the deep dream state.", 1.5)
    if ok: 
      self.deep_step=num
      self.leDS.setText(str(num))

  def setOctaveNumber(self):
    num,ok = QInputDialog.getInt(self,"Octave Number",
      "Enter the amount of octaves you'd like to run on the image (3 or 4 is a normal dose)", 4)
    if ok: 
      self.octave_number=num
      self.leON.setText(str(num))

  def setOctaveScale(self):
    num,ok = QInputDialog.getDouble(self,"Octave Scale",
      "Enter the scale of each octave (floating decimal number)", 1.4)
    if ok: 
      self.octave_scaled=num
      self.leOS.setText(str(num))

  def setDownsize(self):
    num,ok = QInputDialog.getDouble(self,"Overload Truncation",
      "Enter the downsizing variable (255/256 preferred)", 255.0)
    if ok: 
      self.downsize=num
      self.leDZ.setText(str(num))

  def setImageNoise(self):
    num,ok = QInputDialog.getInt(self,"Image Noise",
      "Enter the size (x by x) of the initial image noise tile.", 224)
    if ok: 
      self.img_noise_size=num
      self.leIN.setText(str(num))

  def setImagenetMean(self):
    num,ok = QInputDialog.getDouble(self,"Imagenet Mean",
      "Enter the initial mean to use on imagenet gradient.", 117.0)
    if ok: 
      self.leIM.setText(str(num))
      self.imagenet_mean_init=num

  def setGradTileSize(self):
    num, ok = QInputDialog.getInt(self,"Gradient Tile Size",
      "Enter the size of the tiles used on imagenet gradient.", 256)
    if ok: 
      self.leGT.setText(str(num))
      self.grad_tile_size=num

  def setStripConstSize(self):
    num, ok = QInputDialog.getInt(self,"",
      "Enter the size of the buffer for tensor constants.", 32)
    if ok: 
      self.leSC.setText(str(num))
      self.strip_const_size=num

class TextGeneratorParameterDialog(QWidget): #gui hyperparameters for lstm text gen
  def __init__(self, parent = None):
    super(TextGeneratorParameterDialog, self).__init__(parent)
    self.num_epochs=15
    self.num_generate=400
    self.temperature=1.0
    self.trim_text = 1
    self.embedding_dim=128
    self.step_size=3
    self.seq_length=40
    self.BATCH_SIZE=128
    self.setWindowTitle("Text Generator Parameters")
    self.setWindowIcon(QIcon('deep_surfer/icons/text-gen-icon.png'))
    layout = QFormLayout()
    self.btnNE = QPushButton("Epochs of Training")
    self.btnNE.clicked.connect(self.setTrainEpochs)
    self.leNE = QLineEdit()
    self.leNE.setText(str(self.num_epochs))
    self.leNE.setReadOnly(True)
    layout.addRow(self.btnNE,self.leNE)
    self.btnNG = QPushButton("Generated Character Count")
    self.btnNG.clicked.connect(self.setNumGenerate)
    self.leNG = QLineEdit()
    self.leNG.setText(str(self.num_generate))
    self.leNG.setReadOnly(True)
    layout.addRow(self.btnNG,self.leNG)
    self.btnTP = QPushButton("Temperature")
    self.btnTP.clicked.connect(self.setTemperature)
    self.leTP = QLineEdit()
    self.leTP.setText(str(self.temperature))
    self.leTP.setReadOnly(True)
    layout.addRow(self.btnTP,self.leTP)
    self.btnTT = QPushButton("Trim Start/End Characters")
    self.btnTT.clicked.connect(self.setTrimText)
    self.leTT = QLineEdit()
    self.leTT.setText(str(self.trim_text))
    self.leTT.setReadOnly(True)
    layout.addRow(self.btnTT,self.leTT)
    self.btnED = QPushButton("Embedding Dimensions of Network")
    self.btnED.clicked.connect(self.setEmbeddingDim)
    self.leED = QLineEdit()
    self.leED.setText(str(self.embedding_dim))
    self.leED.setReadOnly(True)
    self.btnSS = QPushButton("Training Step Size")
    self.btnSS.clicked.connect(self.setStepSize)
    self.leSS = QLineEdit()
    self.leSS.setText(str(self.step_size))
    self.leSS.setReadOnly(True)
    layout.addRow(self.btnSS,self.leSS)
    self.btnSL = QPushButton("Character Sequence Length (Training)")
    self.btnSL.clicked.connect(self.setSeqLength)
    self.leSL = QLineEdit()
    self.leSL.setText(str(self.seq_length))
    self.leSL.setReadOnly(True)
    layout.addRow(self.btnSL,self.leSL)
    self.btnBS = QPushButton("Batch Size (Training)")
    self.btnBS.clicked.connect(self.setBatchSize)
    self.leBS = QLineEdit()
    self.leBS.setText(str(self.BATCH_SIZE))
    self.leBS.setReadOnly(True)
    layout.addRow(self.btnBS,self.leBS)
    self.setLayout(layout)

  def setTrainEpochs(self):
    num,ok = QInputDialog.getInt(self,"Training Epochs",
      "Enter the amount of epochs you would like to train for.")
    if ok: 
      self.num_epochs=num
      self.leNE.setText(str(num))

  def setNumGenerate(self):
    num,ok = QInputDialog.getInt(self,"Generated Character Count",
      "Enter the amount of characters you want to generate.")
    if ok: 
      self.num_generate=num
      self.leNG.setText(str(num))

  def setTemperature(self):
    num,ok = QInputDialog.getDouble(self,"Learning Rate",
      "Enter a Positive Decimal Number")
    if ok: 
      self.temperature=num
      self.leTP.setText(str(num))

  def setTrimText(self):
    num,ok = QInputDialog.getInt(self,"Trim Character Count",
      "Enter the amount of characters you want to trim from beginning/end of file being read.")
    if ok: 
      self.trim_text=num
      self.leTT.setText(str(num))

  def setEmbeddingDim(self):
    num,ok = QInputDialog.getInt(self,"Embedding Dimensions of Network (Training)",
      "Enter the amount of embedding dimensions you want in your network.")
    if ok: 
      self.embedding_dim=num
      self.leED.setText(str(num))

  def setStepSize(self):
    num,ok = QInputDialog.getInt(self,"Training Step Size",
      "Set the size of training steps for the network.")
    if ok: 
      self.step_size=num
      self.leSS.setText(str(num))

  def setSeqLength(self):
    num,ok = QInputDialog.getInt(self,"Sequence Length (Training)",
      "Enter the length of character sequences you want the network to create.")
    if ok: 
      self.seq_length=num
      self.leSL.setText(str(num))

  def setBatchSize(self):
    num,ok = QInputDialog.getInt(self,"Batch Size (Training)",
      "Enter the size of batches you want the network to look at.")
    if ok: 
      self.BATCH_SIZE=num
      self.leBS.setText(str(num))

class ImageGeneratorParameterDialog(QWidget): #gui hyperparameters for wgan image gen
  def __init__(self, parent = None):
    super(ImageGeneratorParameterDialog, self).__init__(parent)
    self.resize_side_length=256
    self.height=128
    self.width=128
    self.channels=3
    self.batch_size=64
    self.epochs=1000
    self.random_dim=100
    self.learn_rate=2e-4
    self.clip_weights=0.01
    self.d_iters=5
    self.g_iters=1
    self.save_ckpt_rate=500
    self.save_img_rate=50
    self.setWindowTitle("Image Generator Parameters")
    self.setWindowIcon(QIcon('deep_surfer/icons/gan-icon.png'))
    layout = QFormLayout()
    self.btnSL = QPushButton("Resize Side Length")
    self.btnSL.clicked.connect(self.setResize)
    self.leSL = QLineEdit()
    self.leSL.setText(str(self.resize_side_length))
    self.leSL.setReadOnly(True)
    layout.addRow(self.btnSL,self.leSL)
    self.btnHG = QPushButton("Height")
    self.btnHG.clicked.connect(self.setHeight)
    self.leHG = QLineEdit()
    self.leHG.setText(str(self.height))
    self.leHG.setReadOnly(True)
    layout.addRow(self.btnHG,self.leHG)
    self.btnWD = QPushButton("Width")
    self.btnWD.clicked.connect(self.setWidth)
    self.leWD = QLineEdit()
    self.leWD.setText(str(self.width))
    self.leWD.setReadOnly(True)
    layout.addRow(self.btnWD,self.leWD)
    self.btnCH = QPushButton("Channels")
    self.btnCH.clicked.connect(self.setChannels)
    self.leCH = QLineEdit()
    self.leCH.setText(str(self.channels))
    self.leCH.setReadOnly(True)
    layout.addRow(self.btnCH,self.leCH)
    self.btnBS = QPushButton("Batch Size")
    self.btnBS.clicked.connect(self.setBatchSize)
    self.leBS = QLineEdit()
    self.leBS.setText(str(self.batch_size))
    self.leBS.setReadOnly(True)
    layout.addRow(self.btnBS, self.leBS)
    self.btnTE = QPushButton("Epochs")
    self.btnTE.clicked.connect(self.setTrainEpochs)
    self.leTE = QLineEdit()
    self.leTE.setText(str(self.epochs))
    self.leTE.setReadOnly(True)
    layout.addRow(self.btnTE,self.leTE)
    self.btnRD = QPushButton("Initial Random Dimensions")
    self.btnRD.clicked.connect(self.setRandomDim)
    self.leRD = QLineEdit()
    self.leRD.setText(str(self.random_dim))
    self.leRD.setReadOnly(True)
    layout.addRow(self.btnRD,self.leRD)
    self.btnLR = QPushButton("Learning Rate")
    self.btnLR.clicked.connect(self.setLearnRate)
    self.leLR = QLineEdit()
    self.leLR.setText(str(self.learn_rate))
    self.leLR.setReadOnly(True)
    layout.addRow(self.btnLR,self.leLR)
    self.btnCW = QPushButton("Clip Weights")
    self.btnCW.clicked.connect(self.setClipWeights)
    self.leCW = QLineEdit()
    self.leCW.setText(str(self.clip_weights))
    self.leCW.setReadOnly(True)
    layout.addRow(self.btnCW,self.leCW)
    self.btnDI = QPushButton("Discriminator Iteration Rate")
    self.btnDI.clicked.connect(self.setDIR)
    self.leDI = QLineEdit()
    self.leDI.setText(str(self.d_iters))
    self.leDI.setReadOnly(True)
    layout.addRow(self.btnDI,self.leDI)
    self.btnGI = QPushButton("Generator Iteration Rate")
    self.btnGI.clicked.connect(self.setGIR)
    self.leGI = QLineEdit()
    self.leGI.setText(str(self.g_iters))
    self.leGI.setReadOnly(True)
    layout.addRow(self.btnGI, self.leGI)
    self.btnSC = QPushButton("Save Checkpoint Every "+ str(self.save_ckpt_rate) + " epochs")
    self.btnSC.clicked.connect(self.setSaveCkptRate)
    self.leSC = QLineEdit()
    self.leSC.setText(str(self.save_ckpt_rate))
    self.leSC.setReadOnly(True)
    layout.addRow(self.btnSC,self.leSC)
    self.btnSI = QPushButton("Save Generated Image Every " + str(self.save_img_rate) + " epochs")
    self.btnSI.clicked.connect(self.setSaveImgRate)
    self.leSI = QLineEdit()
    self.leSI.setText(str(self.save_img_rate))
    self.leSI.setReadOnly(True)
    layout.addRow(self.btnSI,self.leSI)
    self.setLayout(layout)

  def setResize(self):
    num,ok = QInputDialog.getInt(self,"Resize Side Length",
      "Enter the side length of the image files after using the resizing tool in advanced.",
      256)
    if ok: 
      self.resize_side_length=num
      self.leSL.setText(str(num))

  def setHeight(self):
    num,ok = QInputDialog.getInt(self,"Height",
      "Enter the height of the network architectures during training.", 128)
    if ok: 
      self.height=num
      self.leHG.setText(str(num))

  def setWidth(self):
    num,ok = QInputDialog.getInt(self,"Width",
      "Enter the width of the network architectures.", 128)
    if ok:
      self.width=num
      self.leWD.setText(str(num))

  def setChannels(self):
    num,ok = QInputDialog.getInt(self,"Channels",
      "Enter the number of color channels for the network", 3)
    if ok: 
      self.leCH.setText(str(num))
      self.channels=num

  def setBatchSize(self):
    num,ok = QInputDialog.getInt(self,"Batch Size",
      "Set the size of batches during wGAN training.", 64)
    if ok: 
      self.leBS.setText(str(num))
      self.batch_size=num

  def setTrainEpochs(self):
    num,ok = QInputDialog.getInt(self,"Training Epochs",
      "Enter the amount of epochs you would like to train for.", 1000)
    if ok: 
      self.leTE.setText(str(num))
      self.epochs=num

  def setRandomDim(self):
    num,ok = QInputDialog.getInt(self,"Initial Random Dimensions",
      "Enter the random dimension you would like to put into the generator.", 100)
    if ok: 
      self.random_dim=num
      self.leRD.setText(str(num))

  def setLearnRate(self):
    num,ok = QInputDialog.getDouble(self,"Learning Rate",
      "Enter the rate at which the network is learning.", 2e-4)
    if ok:
      self.batch_size=num
      self.leLR.setText(str(num))

  def setClipWeights(self):
    num,ok = QInputDialog.getDouble(self,"Clip Weights",
      "Enter the amount to clip off the weights during each training step.", 0.01)
    if ok: 
      self.clip_weights=num
      self.leCW.setText(str(num))

  def setDIR(self):
    num,ok = QInputDialog.getInt(self,"Discriminator Iteration Rate",
      "Enter how many iterations you want the discriminator to take at each combined step.", 5)
    if ok: 
      self.d_iters=num
      self.leDI.setText(str(num))

  def setGIR(self):
    num,ok = QInputDialog.getInt(self,"Generator Iteration Rate",
      "Enter how many iterations you want the generator to take at each combined step.", 1)
    if ok:
      self.g_iters=num
      self.leGI.setText(str(num))

  def setSaveCkptRate(self):
    num,ok = QInputDialog.getInt(self,"Save Checkpoint Rate",
      "Enter the rate at which checkpoints will save by step.", 500)
    if ok: 
      self.save_ckpt_rate=num
      self.leSC.setText(str(num))

  def setSaveImgRate(self):
    num,ok = QInputDialog.getInt(self,"Save Image Rate",
        "Enter the rate at which generated images will save by step.", 50)
    if ok: 
      self.save_img_rate=num
      self.leSI.setText(str(num))
"""
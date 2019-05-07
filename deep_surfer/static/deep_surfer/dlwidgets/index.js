
//create IC model and grab the webcam
const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

let recognizer;
// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
let examples = [];

async function setupWebcam() {
  console.log("in setupWebcam\n");
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
        reject();
    }
  });
}

function collect(label) {
  if (recognizer.isListening()) {
    return recognizer.stopListening();
  }
  if (label == null) {
    return;
  }
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    examples.push({vals, label});
    document.querySelector('#console').textContent =
      `${examples.length} examples collected`;
  }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
  });
}

function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

function predictWord() {
  // Array of words that the recognizer is trained to recognize.
  const words = recognizer.wordLabels();
  recognizer.listen(({scores}) => {
    // Turn scores into a list of (score,word) pairs.
    scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
    // Find the most probable word.
    scores.sort((s1, s2) => s2.score - s1.score);
    document.querySelector('#console').textContent = scores[0].word;
    }, {probabilityThreshold: 0.75});
}

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
  toggleButtons(false);
  const ys = tf.oneHot(examples.map(e => e.label), 3);
  const xsShape = [examples.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);
  
  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector('#console').textContent =
          `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleButtons(true);
}

function buildModel() {
  model = tf.sequential();
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 8,
    kernelSize: [NUM_FRAMES, 3],
    activation: 'relu',
    inputShape: INPUT_SHAPE
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
}

function toggleButtons(enable) {
  document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}

async function moveSlider(labelTensor) {
  const label = (await labelTensor.data())[0];
  document.getElementById('console').textContent = label;
  if (label == 2) {
    return;
  }
  let delta = 0.1;
  const prevValue = +document.getElementById('output').value;
  document.getElementById('output').value =
  prevValue + (label === 0 ? -delta : delta);
}

function listen() {
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById('listen').textContent = 'Listen';
    return;
  }
  toggleButtons(false);
  document.getElementById('listen').textContent = 'Stop';
  document.getElementById('listen').disabled = false;
  
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
    const probs = model.predict(input);
    const predLabel = probs.argMax(1);
    await moveSlider(predLabel);
    tf.dispose([input, probs, predLabel]);
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
}

async function imapp() {
  cam = await setupWebcam();
  var v = document.getElementsByTagName("video");
  v[0].style.filter = "opacity(100%)";
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');
    
    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };
  
  var watch = true; //flag to break the infinite loop
  document.getElementById('console').innerText = `poopy popopopopopooooop`;
  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('class-d').addEventListener('click', () => addExample(3));
  document.getElementById('class-e').addEventListener('click', () => addExample(4));
  document.getElementById('wicstop').addEventListener('click', () => watch = false);
  while (watch) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);
      
      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
    prediction: ${classes[result.classIndex]}\n
    probability: ${result.confidences[result.classIndex]}
      `;
    }
    
    await tf.nextFrame();
  }
  webcamElement.srcObject.getTracks()[0].stop();

  v[0].style.filter = "opacity(20%)";
}

async function app() {
  document.getElementById('wicstart').addEventListener('click', () => imapp());
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  buildModel();
}

app();

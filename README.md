# ML
Machine Learning with Python (TensorFlow)


Sound Classification is a widely used application in the field of Audio Deep Learning.
Its objective is to classify sounds and predict their corresponding categories.
This type of problem finds practical applications in various scenarios.
For example, it can be used to classify music clips and determine the genre of the music, or to classify short spoken phrases by different speakers and identify the speaker based on their voice.

Starting with sound files, convert them into spectrograms, input them into CNN Plus Linear Classifier model, and produce predictions about the class to which the sound belongs.

You can find various datasets that are suitable for different types of sounds.
These datasets come with a significant number of audio samples, and each sample is assigned a class label indicating the type of sound it represents, depending on the specific problem you're working on.
These class labels can be derived from different sources.
Sometimes, they can be extracted from the filename of the audio sample or the name of the sub-folder where the file is stored.
Alternatively, the class labels may be provided in a separate metadata file, typically in formats like TXT, JSON, or CSV.

Prepare training data SOPs:
1. Download the dataset
2. Prepare training data
3. Pre-process training data
4. Build Model
5. Train model
6. Inference

Training data for this presentation will be simple :

The features (X) are the audio file paths.
The target lwh4le (y) are the class names.

Audio pre-processing : Define Transforms
This training data with audio file paths is to be loaded from the file so that is is in a format that the model expects.

This audio pre-processing will be run dynamically at runtime when read and load the audio files.

Then, at runtime, as we train the model one batch at a time, process it by applying a series of transforms to the audio that will keep audio data for only one batch in memory at a time.

Convert to two channels tasks
Some audio files are mono while most of them are stereo or aka 2 audio channels.

Standardize sampling rate 
Convert all audio to the same sampling rate so that ll arrays have the same dimeensions.

Resize to the same length
Add method by resizing and extending it's duration by padding it with silence, or by truncating it.

Data Augmentaion to the raw audio signal by applying a time shift to shift the audio to the left or right by a rendom amount.

Mel Spectrogram
We then convert the augmented audio to a Mel Spectrogram.
They capture the essential features of the audio and are often the most suitable way to input audio data into deep learning modelsa Mel Spectrogram is,
why they are crucial for audio deep learning, as well as how they are generated and how to tune them for getting the best performance from your models.

Data Augmentation: Time and Frequency Masking
Now we can do another round of augmentation, 
this time on the Mel Spectrogram rather than on the raw audio.
We will use a technique called SpecAugment that uses these two methods:
Frequency mask — randomly mask out a range of consecutive frequencies by adding horizontal bars on the spectrogram.
Time mask — similar to frequency masks, except that we randomly block out ranges of time from the spectrogram by using vertical bars.

Define custom data loader
we start training, the Data Loader will randomly fetch one batch of input Features containing the list of audio file names and run the pre-processing audio transforms on each audio file.
It will also fetch a batch of the corresponding target Labels containing the class IDs.
Thus it will output one batch of training data at a time, which can directly be fed as input to our deep learning model.

After transformation of the audio file , each batch will have two tensors,
one for the X feature data containing the Mel Spectrograms and the other for the y target labels containing numeric Class IDs.
The batches are picked randomly from the training data for each training epoch.
Each batch has a shape of (batch_sz, num_channels, Mel freq_bands, time_steps)

We then visualize one item from the batch.
We see the Mel Spectrogram with vertical and horizontal stripes showing the Frequency and Time Masking data augmentation.

Create Model
The data processing steps that we just did are the most unique aspects of our audio classification problem.
From here on, the model and training procedure are quite similar to what is commonly used in a standard image classification problem and are not specific to audio deep learning.
Since our data now consists of Spectrogram images,
we build a CNN classification architecture to process them.
It has four convolutional blocks which generate the feature maps.
That data is then reshaped into the format we need so it can be input into the linear classifier layer, which finally outputs the predictions for the 10 classes.

Training
We are now ready to create the training loop to train the model.
We define the functions for the optimizer, loss, and scheduler to dynamically vary our learning rate as training progresses,
which usually allows training to converge in fewer epochs.
We train the model for several epochs, processing a batch of data in each iteration.
We keep track of a simple accuracy metric which measures the percentage of correct predictions.

Inference
Ordinarily, as part of the training loop, we would also evaluate our metrics on the validation data.
We would then do inference on unseen data, perhaps by keeping aside a test dataset from the original data. However,
for the purposes of this demo, we will use the validation data for this purpose.
We run an inference loop taking care to disable the gradient updates. 
The forward pass is executed with the model to get predictions, but we do not need to backpropagate or run the optimizer.

Conclusion
We have now seen an end-to-end example of sound classification
which is one of the most foundational problems in audio deep learning.




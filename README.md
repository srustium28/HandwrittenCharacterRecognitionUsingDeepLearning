# Handwritten Charcater using Deep Learning with Tensorflow
### Description
Used Convolutional Recurrent Neural Network to recognize the Handwritten Character without pre segmentation into words or characters. Use CTC loss Function to train.
CRN- 
 Why Deep Learning?
 Deep Learning self extracts features with a deep neural networks and classify itself. 
 Compare to traditional Algorithms it performance increase with Amount of Data.



## <i> Basic Intuition on How it Works.
![Step_wise_detail](images/Step_wise_detail_of_workflow.png?raw=true "Step_Wise Detail")
* First Use Convolutional Recurrent Neural Network to extract the important features from the handwritten line text Image.
* The output before CNN FC layer (512x100x8) is passed to the BLSTM which is for sequence dependency and time-sequence operations.
* Then CTC LOSS [Alex Graves](https://www.cs.toronto.edu/~graves/icml_2006.pdf) is used to train the RNN which eliminate the Alignment problem in Handwritten, since handwritten have different alignment of every writers. We just gave the what is written in the image (Ground Truth Text) and BLSTM output, then it calculates loss simply as `-log("gtText")`; aim to minimize negative maximum likelihood path.
* Finally CTC finds out the possible paths from the given labels. Loss is given by for (X,Y) pair is: ![Ctc_Loss](images/CtcLossFormula.png?raw=true "CTC loss for the (X,Y) pair")
* Finally CTC Decode is used to decode the output during Prediction.
</i>


#### Detail Project Workflow
Model Architecture


* Project consists of Three steps:
  1. Multi-scale feature Extraction --> Convolutional Neural Network 7 Layers
  2. Sequence Labeling (BLSTM-CTC)  --> Recurrent Neural Network (2 layers of LSTM) with CTC 
  3. Transcription --> Decoding the output of the RNN (CTC decode)
!
# Requirements
1. Tensorflow 
2. Flask
3. Numpy
4. OpenCv 3
5. Spell Checker `autocorrect` >=0.3.0 ``pip install autocorrect``

#### Dataset Used
* IAM dataset has been downloaded
* Place the downloaded files inside data directory  



Run in Web with Flask
```markdown
$ python upload.py
Validation character error rate of saved model: 8.654728%
Python: 3.6.4 
Tensorflow:
Init with stored values from ../model/snapshot-24
Without Correction clothed leaf by leaf with the dioappoistmest
With Correction clothed leaf by leaf with the dioappoistmest
```
**Prediction output on IAM Test Data**


**Prediction output on Self Test Data**


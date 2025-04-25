# MAST

This is the code of "**MAST: Video Polyp Segmentation with a Mixture-Attention Siamese Transformer**".

<img src="imgs/model_structure.jpg" alt="model_structure" style="zoom:10%;" />

## Weights and Prediction

You can find the weights and predictions of our model in [Google Drive](https://drive.google.com/drive/folders/101fumxq6i72edyUBCFBqnpGvj739TUDg?usp=sharing).

If you want to test and evaluate, please make a directory called "weight" and put the weights in it. Run "MyTesting.py" to generate predictions, and then run "eval/vps_evaluator.py" to generate metrics.

## Evaluation

<img src="imgs/SUN-SEG-Easy.png" alt="SUN-SEG-Easy" style="zoom:10%;" />

<img src="imgs/SUN-SEG-Hard.png" alt="SUN-SEG-Hard" style="zoom:10%;" />

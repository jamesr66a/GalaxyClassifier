Galaxy Classifier
================

Description
----------
A quick classifier model I wrote in TensorFlow for my Astrophysics class and to practice using the framework. Still a work in progress but I've had fun and it's been a great learning experience

File Structure
-------------
* `images` contains the training images I used, taken from www.spacetelescope.org under Creative Commons license. (Their licensing link is broken but they use the CC BY logo. Feel free to contact me if there are any issues)
* `train` contains output files from the TensorFlow runs. The repository currently contains checkpoint files after a run of 6000 steps on my laptop.
* `scale.py` is a quick python script to scale all the images down to 64x64 to reduce computational complexity.
* `train.py` is the main file that defines the model. Run it with no arguments to train or run it with a string (any string) in argv[1] to evaluate (I still have to make the eval better, hang tight)

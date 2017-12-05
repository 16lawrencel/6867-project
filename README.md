TODO:
* debug a3c (and dqn?)

Problems:
* Currently, a3c and dqn perform poorly on Pong after substantial training (around 10 million steps).
* At best, a3c gets small negative score (>= -3), but most of the time still gets < -10 score.
* dqn just doesn't really work.

Things that I did in a3c code:
* 20-step look ahead
* decaying learn rate
* gradient norm clipping (max gradient is 40, like in paper)
* training batch of 10000 (neural net stores data points collected by agents until size reaches 10000, then updates on entire data set at once)
* frame skipping (change FRAME_SKIP, paper uses 4)
* checkpoints

Potential Bugs:
* Need to normalize R's
* Wrong update

Changes:
* Fixed preprocessing - copied from another implementation for now
* Fixed frame updating - agents did not update frames (so they never annealed eps to 0)

My dqn code is based on code from the following:
* https://www.tensorflow.org/tutorials/layers
* https://github.com/tflearn/tflearn/blob/master/examples/reinforcement_learning/atari_1step_qlearning.py
* The conv net code in hw3


a3c code is based on the following:
* https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
* dqn code


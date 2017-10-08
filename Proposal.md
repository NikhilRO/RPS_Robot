## Rock-Paper-Scissors Robot God (tentative)

### What is it?
A mechanical hand that can form rock-paper scissors gestures and recognize those of a human’s; it uses this information to play the game against human players.

### Software components:
* OpenCV for hand gesture recognition
* The OpenCV C API will be used for camera capture in order to detect the shapes that the robot sees live
* Hardware components:
* Acquired Raspberry Pi will be used to connect the robot hand to a webcam
* Need robotic parts (specification needed) for the hand
* Need webcam

### Prototype Plan:
* Get camera and computer to communicate with one another (possibly first detect movement in the frame)
* Get camera and computer to recognize/approximate different hand gestures (RPC) and output what the computer's move would be (Eg 1 for rock, 2 for paper, 3 scissors)
* Make robotic hand and use it with the previous prototype

### Challenges:
* Getting the software to recognize, analyze and differentiate between three gestures
* Also, getting it to differentiate between rock or initial three/two hand movements (when you say Rock-Paper-Scissors-Go and pump your fist 3 times before playing)
* Designing the hand to move fast enough to match human reaction time
* Differentiating between valid or invalid moves
* Learning how to get hardware to interact with software, playing around with different inputs, outputs, and algorithms to optimize efficiency (kind of sketch?)

### Useful Links:
* [C Library for Camera](http://docs.opencv.org/3.3.0/dd/d01/group__videoio__c.html#gae38819ff8d6fa81e72c1ee032aa86284)
* [Inspiration for our idea](http://www.k2.t.u-tokyo.ac.jp/fusion/Janken/index-e.html)




 

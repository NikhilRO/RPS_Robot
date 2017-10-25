import gab.opencv.*; //<>// //<>//
import processing.video.*;
import java.awt.*;

Capture video;
OpenCV opencv;

int max, a, prev;

PImage canny;

void setup() {
  frameRate(4);
  max = 0;
  a = 0;
  prev =0;

  size(640, 480);
  video = new Capture(this, 640/4, 480/4);
  opencv = new OpenCV(this, 640/4, 480/4);

  //opencv.startBackgroundSubtraction(5, 3, 0.5);

  video.start();
}

void draw() {
  opencv.loadImage(video);

  opencv.findCannyEdges(50, 100);
  canny = opencv.getSnapshot();
  image(canny, 0, 0, width, height);

  canny.loadPixels();
  for (int i = 0; i < canny.width; i++) { 
    for (int j =0; j < canny.height; j++) {
      if (canny.pixels[(canny.width*j)+i] == color(255, 255, 255) && prev == 0) { 
        a += 1;
        prev = 1;
      } else if (canny.pixels[(canny.width*j)+i] != color(255, 255, 255)){
        prev = 0;
      }
    }
    if (a > max) { 
      max = a;
    }
    a=0;
  }
  canny.updatePixels();
  println(max);
  if(max<10){ println("scissors");
  } else if (max < 14){ println("rock");
  } else { println("paper"); }
  max=0;
  
}

void captureEvent(Capture c) {
  c.read();
}
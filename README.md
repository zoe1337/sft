# Shitty Face Tracker

It's crap but I spent more than 20 hours writing it.

## UwU What is this
It is a primitive face tracker, using OpenCV's built-in haarcascades classifier and minimum filtering.

You could surely do better. I'm sharing it because there is not enough crap on the internet, and I have better things to do.

I tried using state-of-the-art face detection algorithms and all of them have a documentation as pleasant as a big steaming pile of shit.

This Thing opens the camera, grabs frames, chews on them, and then outputs the frames to a v4l2 loopback device. Problem is, some video conferenceing stuff - jitsi, for example - won't list it because it isn't a capture device, but a video source. So this would still need some work by someone who isn't me.

## But why
Badly managed ADHD, appreciation for Ghost in the Shell, the Fediverse, and memes

Playing with machine vision can be fun, and can be also very frustrating. Ultimately it is a lot like sausage. Once you learn how it's made, you don't want it anymore.

## How to use
You sure about that? Alright then.

### Install Dependencies
Somehow get these things on your computer:
 - Linux, or something that speaks V4L2 APIs reasonably well
 - Video camera with a working driver
 - Recent-ish OpenCV, python3, opencv, python-v4l2, python-imageio, v4l2loopback, python-argparse

 I use arch btw. If you also do so, you can just  
 `yay -S v4l2loopback-dkms python-imageio python-v4l2 python-argparse opencv`

### Clone this repository

### Run
You might need to tweak variables if it breaks. But it's just `./laughing-cat.py`

If you want to see a window with the processed video, use the `--gui` option.

If you don't want the v4l2loopback (why is it not called v4l3oopback?) use the `--noloopback` option. 

Using both options makes the Shitty Face Tracker do absolutely nothing. I'm not sure if that makes it less useful.

## Q&A
### This adds latency to the video!
Yes. Ideally not more than 2 frame latency, but YMMV.

### How do you detect faces?
I don't! OpenCV does with the haarcascade classifier.

### Why did you then write 300 lines of code around it?
Because alpha compositing is apparently not important enough to be part of OpenCV.

### How does the face tracker work?
Mostly it does not!

The idea is to have a stateful face object for each face on the image. On each new frame I use haarcascades to detect faces (without apriori knowledge). The array of those faces (as rectangles) are then iterated over and I try to find close ones, update the face tracker with those positions, eliminate duplicates, and then add any new face which might have appeared.

My original idea was to use a sliding window of just 3 frames, and if on 2 or 3 of those frames a face was detected, it is considered valid. Otherwise it is discarded as noise.

### You don't know what you're talking about!
Most certainly.

### This consumes a lot of CPU! It can be done more efficiently.
Yes but I can't bother.

### You don't have anything better to spend your time on?
In fact, I do. But it's difficult if not impossible to control what I find engaging in the moment.

### You can babble a lot!
Indeed! Follow me on [fedi](https://chaos.social/@uint8_t) for more!

### Do you like pull requests?
I do.

### Aren't you just procrastinating somegthing right now?
Alright, alright. I will finish this l
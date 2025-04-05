# Pi_imx500_detection_2

Pi + Pi imx500 AI camera. Tested on Pi4 and Pi5.

It is a modified version of imx500_object_detection_demo.py

Captures videos as .h264 and converts to .mp4 

Runs a pre-capture buffer, user defined length

you can set the objects to detect in line 49, objects = ["cat","bear","bird"], the objects must be in coco_labels.txt file

Copy imx500_detect_2.py into /home/USER/picamera2/examples/imx500

sudo apt install python3-opencv -y

Videos saved to ram and then copied to /home/USERNAME/Videos

stills saved to /home/USER/Pictures

to run ... python3 imx500_detect_2.py

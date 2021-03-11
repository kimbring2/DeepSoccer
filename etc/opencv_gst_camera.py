import traitlets
import atexit
import cv2
import threading
import numpy as np
from .camera_base import CameraBase


class OpenCvGstCamera(CameraBase):
    value = traitlets.Any()

    # config
    width = traitlets.Integer(default_value=640).tag(config=True)
    height = traitlets.Integer(default_value=360).tag(config=True)
    fps = traitlets.Integer(default_value=21).tag(config=True)
    capture_width = traitlets.Integer(default_value=640).tag(config=True)
    capture_height = traitlets.Integer(default_value=360).tag(config=True)

    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        super().__init__(self, *args, **kwargs)
        
        path_video = "/home/kimbring2/video_folder/image_1.avi"
        self.video_out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 360))

        self.frame_list = []

        try:
            #print("self._gst_str(): " + str(self._gst_str()))
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)

            re, image = self.cap.read()

            if not re:
                raise RuntimeError('Could not read image from camera.')

            self.value = image
            self.start()
        except:
            self.stop()
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        #print("_capture_frames")
        while True:
            re, image = self.cap.read()
            if re:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                self.value = rotated_image 
                self.frame_list.append(rotated_image)
                if len(self.frame_list) > 10000:
                    self.frame_list = []
            else:
                break

            #self.restart()
                
    def _gst_str(self):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (self.capture_width, self.capture_height, self.fps, self.width, self.height)
    
    def start(self):
        #print("self.cap.isOpened(): " + str(self.cap.isOpened()))
        if self.cap.isOpened():
            #print("Thread start")
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()
            
    def restart(self):
        self.stop()
        self.start()
    
    def save_video(self):
        #print("len(self.frame_list): " + str(len(self.frame_list)))
        frame_array = np.array(self.frame_list)
        #print("frame_array.shape: " + str(frame_array.shape))
        for i in range(len(frame_array)):
            #print("frame_array[i].shape: " + str(frame_array[i].shape))
            self.video_out.write(frame_array[i])

        self.video_out.release()

    @staticmethod
    def instance(*args, **kwargs):
        return OpenCvGstCamera(*args, **kwargs)

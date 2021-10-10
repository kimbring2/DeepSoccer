import traitlets
import atexit
import cv2
import threading
import numpy as np
from .camera_base import CameraBase


class OpenCvGstCamera(CameraBase):
    
    value = traitlets.Any()
    
    # config
    width = traitlets.Integer(default_value=816).tag(config=True)
    height = traitlets.Integer(default_value=616).tag(config=True)
    fps = traitlets.Integer(default_value=30).tag(config=True)
    capture_width = traitlets.Integer(default_value=816).tag(config=True)
    capture_height = traitlets.Integer(default_value=616).tag(config=True)
    record_step = 0
    record_flag = False

    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        super().__init__(self, *args, **kwargs)
        
        path_video = "/home/kimbring2/Desktop/content_video.avi"
        self.video_out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, (816, 616))

        try:
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
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
                if self.record_flag == True:
                    self.video_out.write(image)
                    #cv2.imwrite("/home/kimbring2/Desktop/image_out_" + str(self.step) + ".jpg", image)
                    self.record_step += 1
            else:
                break
                
    def _gst_str(self):
        return 'nvarguscamerasrc sensor-mode=3 ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                self.capture_width, self.capture_height, self.fps, self.width, self.height)
    def record_start(self):
        self.record_flag = True

    def record_stop(self):
        self.record_flag = False

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
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
        
    @staticmethod
    def instance(*args, **kwargs):
        return OpenCvGstCamera(*args, **kwargs)

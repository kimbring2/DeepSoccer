# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the bEase image

import cv2
import numpy as np
import tensorflow as tf

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

imported_seg = tf.saved_model.load("/home/kimbring2/segmentation_model_official/")
f_seg = imported_seg.signatures["serving_default"]
seg_test_input = np.zeros([1,256,256,3])
seg_test_tensor = tf.convert_to_tensor(seg_test_input, dtype=tf.float32)
print(f_seg(seg_test_tensor))

#imported_style = tf.saved_model.load("/home/kimbring2/Desktop/style_model")
#f_style = imported_style.signatures["serving_default"]
#style_test_input = np.zeros([1,256,256,3])
#style_test_tensor = tf.convert_to_tensor(style_test_input, dtype=tf.float32)
#f_style(style_test_tensor)['output_1']

path_raw_video = '/home/kimbring2/Desktop/raw_video.avi'
path_dilation_video = '/home/kimbring2/Desktop/dilation_video.avi'
path_blue_video = '/home/kimbring2/Desktop/blue_video.avi'
path_green_video = '/home/kimbring2/Desktop/green_video.avi'
path_segmented_video = '/home/kimbring2/Desktop/segmented_video.avi'

fps = 5
raw_video_out = cv2.VideoWriter(path_raw_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,720))
dilation_video_out = cv2.VideoWriter(path_dilation_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,720))
blue_video_out = cv2.VideoWriter(path_blue_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,720))
green_video_out = cv2.VideoWriter(path_green_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,720))
segmented_video_out = cv2.VideoWriter(path_segmented_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,720))

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    #print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        #window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        #while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        while True:
            print("True")
                
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            raw_video_out.write(frame)
            img_ = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            #cv2.imwrite("original_image.jpg", img_)
            
            img_ = cv2.normalize(img_, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA).astype(np.float32)
            
            resized = np.array([img_])
            input_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
            output = f_seg(input_tensor)['conv2d_transpose_4'].numpy()
            #output_style = f_style(input_tensor)['output_1'].numpy()[0]
            #print("output_style.shape: " + str(output_style.shape))
            #cv2.imwrite("output_style.jpg", output_style)
            
            pred_mask = tf.keras.preprocessing.image.array_to_img(create_mask(output))
            pred_mask = np.array(pred_mask)
            #print("pred_mask.shape: " + str(pred_mask.shape))
            #print("pred_mask: " + str(pred_mask))
            ret, thresh = cv2.threshold(pred_mask, 126, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5, 5), np.uint8)
            erodition_image = cv2.erode(thresh, kernel, iterations=2)  #// make dilation image
            dilation_image = cv2.dilate(erodition_image, kernel, iterations=2)  #// make dilation image
            dilation_image = cv2.resize(dilation_image, dsize=(1280,720), interpolation=cv2.INTER_AREA)
            dilation_image = np.float32(dilation_image)
            #dilation_image = cv2.resize(np.float32(dilation_image), dsize=(1280,720), interpolation=cv2.INTER_AREA)
            dilation_image_rgb = cv2.cvtColor(dilation_image, cv2.COLOR_GRAY2RGB)
            print("np.uint8(dilation_image_rgb).shape: " + str(np.uint8(dilation_image_rgb).shape))
            dilation_video_out.write(np.uint8(dilation_image_rgb))
            dilation_image = dilation_image != 255.0
            
            # converting from BGR to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red color
            low_red = np.array([120, 155, 84])
            high_red = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv_frame, low_red, high_red)
            red = cv2.bitwise_and(frame, frame, mask=red_mask)

            # Blue color
            low_blue = np.array([110, 130, 2])
            high_blue = np.array([126, 255, 255])
            blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
            kernel = np.ones((10, 10), np.uint8)
            blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)  #// make dilation image
            blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
            #cv2.imwrite("blue.jpg", blue)
            blue_video_out.write(blue)

            # Green color
            low_green = np.array([25, 52, 72])
            high_green = np.array([60, 255, 255])
            green_mask = cv2.inRange(hsv_frame, low_green, high_green)
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.dilate(green_mask, kernel, iterations=1)  #// make dilation image
            green = cv2.bitwise_and(frame, frame, mask=green_mask)
            #cv2.imwrite("green.jpg", green)
            green_video_out.write(green)

            mask = dilation_image - green_mask - blue_mask
            #mask = green_mask + dilation_image

            result = cv2.bitwise_and(frame, frame, mask=mask)
            #cv2.imwrite("segmented_image.jpg", result)
            
            result_mean = np.mean(result)

            keep_mask_0 = result[:,:,0] == 0
            keep_mask_1 = result[:,:,1] == 0
            keep_mask_2 = result[:,:,2] == 0
            keep_mask = keep_mask_0 + keep_mask_1 + keep_mask_2
            #print("keep_mask: " + str(keep_mask))
            result[keep_mask] = result_mean
            #print("output: " + str(output))
            
            #print("result.shape: " + str(result.shape))
            #cv2.imwrite("segmented_image.jpg", result)
            segmented_video_out.write(result)
            '''
            combined_image = np.zeros((360, 720, 3))

            img_resized = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
            output = cv2.resize(output, (360, 360), interpolation=cv2.INTER_AREA)
            
            #print("output: " + str(output))

            combined_image[:,0:360,:] = img_resized
            combined_image[:,360:,:] = output
                
            combined_image = combined_image.astype(np.uint8)
            #cv2.imshow("CSI Camera", combined_image)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
            '''
        cap.release()
        #cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()

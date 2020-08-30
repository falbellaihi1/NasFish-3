import re
import threading

import cv2 as cv
import numpy as np
import os
from time import time

from win32gui import GetWindowText, GetForegroundWindow

from src import chat_box
from src.windowcapture import WindowCapture
from src.vision import Vision
from src.hsvfilter import HsvFilter
import pydirectinput
from PIL import Image
import pytesseract
# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Window_name = 'Chaja'
# initialize the WindowCapture class
wincap = WindowCapture(Window_name)
# initialize the Vision class
vision_limestone = Vision('../img/silver_left.png')
vision_limestone_right = Vision('../img/silver_right.png')
# initialize the trackbar window
vision_limestone.init_control_gui()

# limestone HSV filter
hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)



# def process_screen_shot(screenshot):
#     print('checking box')
#     num_image = Image.fromarray(screenshot, mode='RGB')
#
#     num_image.save('img/1.png')
#     image = cv.imread('img/1.png', 0)
#
#     # image_proc = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     x = 0
#     y = wincap.h - 36
#     h = wincap.h -100
#     w = wincap.w - 500
#
#     ROI = image[y:y + h, x:x + w]
#     thresh = cv.threshold(ROI, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
#     result = 255 - thresh
#     kernel = np.ones((1, 1))
#     imgDil = cv.dilate(result, kernel, iterations=1)
#     data = pytesseract.image_to_string(imgDil, config='--psm 6 --oem 3')
#     cv.imshow('chat commands box', imgDil)
#     result_text = "".join(re.split("[^a-zA-Z ]*", data))
#
#     print(result_text)
#     return data



timer = 9
loop_time = time()

def welcome_threading():
    print("thread")
    threading.Timer(5, welcome_threading).start()
    return ''
MATCHED_SOMETHING_IN_CHAT = False
commands = welcome_threading()
timer=13

def main():
    while(True):
        # print(wincap.w)
        # print(wincap.h)


        # get an updated image of the game
        screenshot = wincap.get_screenshot()
        # pre-process the image
        chat = chat_box
        chat.crop_chat(screenshot, wincap.w, wincap.h)
        #print(chat)

        processed_image = vision_limestone.apply_hsv_filter(screenshot)

        processed_image_right = vision_limestone_right.apply_hsv_filter(screenshot)

        # do edge detection
        edges_image = vision_limestone.apply_edge_filter(processed_image)
        edges_image_right = vision_limestone_right.apply_edge_filter(processed_image_right)


        keypoint_image = edges_image

        keypoint_image_right =edges_image_right
        # crop the image to remove the ui elements

        x, y, w, h = [10, 10, 700,500]

        xr, yr, wr, hr = [650, 10, 700,500]

        keypoint_image = keypoint_image[y:y+h, x:x+w]
        keypoint_image_right = keypoint_image_right[yr:yr + hr, xr:xr + wr]

        #left
        kp1, kp2, matches, match_points = vision_limestone.match_keypoints(keypoint_image)
        ## right
        kp1r, kp2r, matches_right, match_points_right = vision_limestone_right.match_keypoints(keypoint_image_right)

        match_left_image = cv.drawMatches(
            vision_limestone.needle_img,
            kp1,
            keypoint_image,
            kp2,
            matches,
            None)
        # RIGHT MATCH IMAGE
        match_right_image = cv.drawMatches(

            vision_limestone_right.needle_img,
            kp1r,
            keypoint_image_right,
            kp2r,
            matches_right,
            None

        )

        if match_points:
            # find the center point of all the matched features
            center_point = vision_limestone.centeroid(match_points)
            # account for the width of the needle image that appears on the left
            center_point[0] += vision_limestone.needle_w
            # drawn the found center point on the output image
            match_left_image = vision_limestone.draw_crosshairs(match_left_image, [center_point])

        if match_points_right:
           # print(match_points_right)
            # find the center point of all the matched features
            center_point_right = vision_limestone.centeroid(match_points_right)
            # account for the width of the needle image that appears on the left
            center_point_right[0] += vision_limestone_right.needle_w
            # drawn the found center point on the output image
            match_right_image = vision_limestone_right.draw_crosshairs(match_right_image, [center_point_right])


        # display the processed image

        # cv.imshow('Keypoint Search', match_left_image)
        # cv.imshow('Keypoint righ', match_right_image)
        match = ['cought the hook', 'something cought', 'something cought the hook', 'something cought the', 'cought the hook']






        if (GetWindowText(GetForegroundWindow()) == 'Chaja'):
            #print('begaining')
            if len(match_points) >= 8:
                print('fish fight left')
                pydirectinput.press('a')
                timer -= 1
                print(timer)

            if len(match_points_right) >= 8:
                print('fish fight right')
                pydirectinput.press('d')
                timer -= 1
                print(timer)

            if (timer == 0):
                # pydirectinput.press('Enter')
                print('== 0')
                timer = 12

        loop_time = time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


run = main()
run.runall()

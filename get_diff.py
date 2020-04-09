#!/usr/bin/env python3

import argparse
import cv2
from skimage.measure import compare_ssim
import imutils
from datetime import datetime, timezone, timedelta
import time
import numpy as np

speed = timedelta(milliseconds=33)

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
cap = cv2.VideoCapture(0)

kernel = np.ones((5,5),np.float32)/25
# read base image
base = cv2.imread('base.png',-1)
base_blur = cv2.medianBlur(base, 21)


while(True):
    start = datetime.now(timezone.utc)
    ret, live = cap.read()

    # Our operations on the frame come here
    #live = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    live_blur = cv2.medianBlur(live, 21)

    # compare images
    # https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    (score, diff) = compare_ssim(live_blur, base_blur, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")

    b,g,r = cv2.split(diff)

    # blue
    thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    color_mask_b = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    color_mask_b = cv2.medianBlur(color_mask_b, 11)
    masked_b = cv2.bitwise_and(live, color_mask_b)
    masked_and = cv2.bitwise_and(live, color_mask_b)

    # green
    thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    color_mask_g = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    color_mask_g = cv2.medianBlur(color_mask_g, 11)
    masked_g = cv2.bitwise_and(live, color_mask_g)
    masked_and = cv2.bitwise_and(masked_and, color_mask_g)

    # red
    thresh = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    color_mask_r = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    color_mask_r = cv2.medianBlur(color_mask_r, 11)
    masked_r = cv2.bitwise_and(live, color_mask_r)
    masked_and = cv2.bitwise_and(masked_and, color_mask_r)

    mask_or = cv2.bitwise_or(color_mask_b, color_mask_g, color_mask_r)
    masked_or = cv2.bitwise_and(live, mask_or)

    # Display the resulting frame
    cv2.imshow('base', base)
    cv2.imshow('base blur', base_blur)
    cv2.imshow('live', live)
    cv2.imshow('live blur', live_blur)
    #cv2.imshow('diff', thresh)
    cv2.imshow('masked and', masked_and)
    cv2.imshow('masked or', masked_or)
    cv2.imshow('masked blue', masked_b)
    cv2.imshow('masked green', masked_g)
    cv2.imshow('masked red', masked_r)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    now = datetime.now(timezone.utc)
    timediff = now - start
    to_wait = speed - timediff
    if to_wait > timedelta(seconds=0):
        time.sleep(to_wait.total_seconds())

cap.release()
cv2.destroyAllWindows()



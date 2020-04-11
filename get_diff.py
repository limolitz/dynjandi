#!/usr/bin/env python3

import sys
import argparse
import cv2
from skimage.metrics import structural_similarity
from datetime import datetime, timezone, timedelta
import time
import numpy as np
import subprocess as sp
from threading import Thread, Event
from queue import Queue
# based on
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

FULL_WIDTH = 1280
FULL_HEIGHT = 720
SMALL_SIZE = (426, 240)

# https://answers.opencv.org/question/92450/processing-camera-stream-in-opencv-pushing-it-over-rtmp-nginx-rtmp-module-using-ffmpeg/
# ffmpeg -f rawvideo -vcodec rawvideo -i - f v4l2 -vcodec rawvideo /dev/video5
FFMPEG_COMMAND = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{FULL_WIDTH}x{FULL_HEIGHT}',
    '-pix_fmt', 'bgr24',
    '-i', '-',
    '-codec', 'copy',
    '-f', 'v4l2',
    '/dev/video3'
]


class ImageCompareThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = Queue()
        self.result = None
        self.stoprequest = Event()

    def run(self):
        # read base image
        base = cv2.imread('base.png', -1)
        base_small = cv2.resize(base, SMALL_SIZE)

        while not self.stoprequest.isSet():
            # Wait for next message
            live = self.queue.get()
            live_small = cv2.resize(live, SMALL_SIZE)

            start = datetime.now(timezone.utc)

            # compare images
            # https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
            (score, diff) = structural_similarity(base_small, live_small, full=True, multichannel=True)
            diff = (diff * 255).astype("uint8")
            ssim_time = datetime.now(timezone.utc)

            b, g, r = cv2.split(diff)

            # blue
            thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            color_mask_b = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # green
            thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            color_mask_g = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # red
            thresh = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            color_mask_r = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            mask_and = cv2.bitwise_and(color_mask_b, color_mask_g, color_mask_r)
            mask_or = cv2.bitwise_or(color_mask_b, color_mask_g, color_mask_r)

            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#
            contours, hierarchy = cv2.findContours(
                cv2.cvtColor(mask_or, cv2.COLOR_BGR2GRAY).copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            largest_contour = None
            area = 0
            for cnt in contours:
                # print(cv2.moments(cnt))
                if cv2.contourArea(cnt) > area:
                    largest_contour = cnt
                    area = cv2.contourArea(cnt)

            # hull = cv2.convexHull(largest_contour)

            contour_mask = np.zeros(base_small.shape, np.uint8)

            # https://stackoverflow.com/a/50022122
            contour_mask = cv2.drawContours(contour_mask, [largest_contour], -1, (255, 255, 255), -1)

            # Display the resulting frame
            #cv2.imshow('base', base)
            #cv2.imshow('base blur', base_blur)
            #cv2.imshow('live blur', live_blur)
            #cv2.imshow('contour mask', img)
            #cv2.imshow('mask or', mask_or)
            #cv2.imshow('mask and', mask_and)
            #cv2.imshow('masked contour', masked_contour)
            #cv2.imwrite("contour.png", contour)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            now = datetime.now(timezone.utc)
            timediff = now - start
            ssim = ssim_time - start
            rest = now - ssim_time
            print(
                f"Processing took {timediff.total_seconds()*1000:.0f} ms, "
                f"ssim {ssim.total_seconds()*1000:.0f}, "
                f"the rest {rest.total_seconds()*1000:.0f}."
            )

            contour_mask_big = cv2.resize(contour_mask, (1280, 720))
            contour_mask_big = cv2.medianBlur(contour_mask_big, 41)
            self.result = contour_mask_big

    def put_message(self, message):
        self.queue.put(message)

    def join(self, timeout=None):
        self.stoprequest.set()
        super().join(timeout)


class MaskingThread(Thread):
    def __init__(self, worker_thread: Thread, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoprequest = Event()
        self.worker_thread = worker_thread
        self.cap = cv2.VideoCapture()

    def run(self):
        contour_mask = None

        green = np.zeros((FULL_HEIGHT, FULL_WIDTH, 3), np.uint8)
        # Fill image with green color for greenscreen
        green[:] = (0, 255, 0)

        cap = self.cap
        cap.open(0, apiPreference=cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30.0)

        proc = sp.Popen(FFMPEG_COMMAND, stdin=sp.PIPE, stderr=sp.PIPE, shell=False)

        ret, live = cap.read()
        self.worker_thread.put_message(live)

        while not self.stoprequest.isSet():
            ret, live = cap.read()

            if self.worker_thread.result is not None:
                contour_mask = self.worker_thread.result
                self.worker_thread.result = None
                self.worker_thread.put_message(live)
            if contour_mask is not None:
                # https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Convert uint8 to float
                foreground = live.astype(float)
                background = green.astype(float)

                # Normalize the alpha mask to keep intensity between 0 and 1
                alpha = contour_mask.astype(float) / 255

                # Multiply the foreground with the alpha matte
                foreground = cv2.multiply(alpha, foreground)

                # Multiply the background with ( 1 - alpha )
                background = cv2.multiply(1.0 - alpha, background)

                # Add the masked foreground and background.
                outImage = cv2.add(foreground, background).astype('uint8')

                proc.stdin.write(outImage.tostring())
        cap.release()

    def join(self, timeout=None):
        self.stoprequest.set()
        super().join(timeout)


def main(argv=None):
    image_compare_thread = ImageCompareThread()
    image_compare_thread.start()

    masking_thread = MaskingThread(image_compare_thread)
    masking_thread.start()

    try:
        pass
    except KeyboardInterrupt:
        print("Exiting...")
        masking_thread.join()
        image_compare_thread.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())

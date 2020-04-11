#!/usr/bin/env python3

import sys
import os
import argparse
import cv2
from skimage.metrics import structural_similarity
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import subprocess as sp
from threading import Thread, Event
from queue import Queue
import curses

FULL_WIDTH = 1280
FULL_HEIGHT = 720
SMALL_SIZE = (426, 240)

utc = timezone.utc

# https://answers.opencv.org/question/92450/processing-camera-stream-in-opencv-pushing-it-over-rtmp-nginx-rtmp-module-using-ffmpeg/
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


class ThreadWithSettingsAndMessages(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = {}
        self.message_queue = Queue()

    def set_setting(self, setting, value):
        self.settings[setting] = value

    def get_setting(self, setting):
        return self.settings[setting]

    def toggle_setting(self, setting):
        self.settings[setting] = not self.settings[setting]
        return self.settings[setting]

    def has_message(self) -> bool:
        return self.message_queue.qsize() > 0

    def get_message(self):
        # don't block
        if self.has_message:
            return self.message_queue.get()
        return None


class ImageCompareThread(ThreadWithSettingsAndMessages):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work_queue = Queue()
        self.result = None
        self.stoprequest = Event()
        self.settings = {
            'use_hull': False,
            'use_and': False,
        }

    def run(self):
        # read base image
        base = cv2.imread('base.png', -1)
        base_small = cv2.resize(base, SMALL_SIZE)

        while not self.stoprequest.isSet():
            # Wait for next message
            live = self.work_queue.get()
            live_small = cv2.resize(live, SMALL_SIZE)

            start = datetime.now(utc)

            # compare images
            # https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
            # TODO: this takes to the most of the processing time, make more efficient
            (score, diff) = structural_similarity(base_small, live_small, full=True, multichannel=True)
            diff = (diff * 255).astype("uint8")
            ssim_time = datetime.now(utc)

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

            mask = None
            if self.settings['use_and']:
                mask = cv2.bitwise_and(color_mask_b, color_mask_g, color_mask_r)
            else:
                mask = cv2.bitwise_or(color_mask_b, color_mask_g, color_mask_r)

            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#
            contours, hierarchy = cv2.findContours(
                cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            largest_contour = None
            area = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > area:
                    largest_contour = cnt
                    area = cv2.contourArea(cnt)

            contour = None
            if self.settings['use_hull'] is True:
                contour = cv2.convexHull(largest_contour)
            else:
                contour = largest_contour

            contour_mask = np.zeros(base_small.shape, np.uint8)

            # https://stackoverflow.com/a/50022122
            contour_mask = cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), -1)

            # Display the resulting frame
            #cv2.imshow('base', base)
            #cv2.imshow('base blur', base_blur)
            #cv2.imshow('live blur', live_blur)
            #cv2.imshow('contour mask', img)
            #cv2.imshow('mask or', mask_or)
            #cv2.imshow('mask and', mask_and)
            #cv2.imshow('masked contour', masked_contour)
            #cv2.imwrite("contour.png", contour)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            now = datetime.now(utc)
            timediff = now - start
            ssim = ssim_time - start
            rest = now - ssim_time
            message = (
                f"SSIM took {ssim.total_seconds()*1000:.0f} ms, "
                f"the rest {rest.total_seconds()*1000:.0f} ms, "
                f"total: {timediff.total_seconds()*1000:.0f} ms."
            )
            self.message_queue.put(message)

            contour_mask_big = cv2.resize(contour_mask, (1280, 720))
            contour_mask_big = cv2.medianBlur(contour_mask_big, 41)
            self.result = contour_mask_big
        cv2.destroyAllWindows()

    def put_work(self, message):
        self.work_queue.put(message)

    def join(self, timeout=None):
        self.stoprequest.set()
        super().join(timeout)


class MaskingThread(ThreadWithSettingsAndMessages):
    def __init__(self, worker_thread: Thread, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoprequest = Event()
        self.worker_thread = worker_thread
        self.cap = cv2.VideoCapture()
        self.settings = {
            'use_green': True,
        }
        self.frame_count = 0

    def run(self):
        contour_mask = None

        green = np.zeros((FULL_HEIGHT, FULL_WIDTH, 3), np.uint8)
        # Fill image with green color for greenscreen
        green[:] = (0, 255, 0)
        base_blur = cv2.imread('base_blur.png', -1)

        # loosely based on
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

        cap = self.cap
        # try to have higher perf by using V4L2 here
        # https://stackoverflow.com/a/55779890
        cap.open(0, apiPreference=cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        proc = sp.Popen(FFMPEG_COMMAND, stdin=sp.PIPE, stderr=sp.PIPE, shell=False)

        ret, live = cap.read()
        self.worker_thread.put_work(live)

        while not self.stoprequest.isSet():
            start = datetime.now(utc)
            ret, live = cap.read()
            reading = datetime.now(utc)

            if self.worker_thread.result is not None:
                contour_mask = self.worker_thread.result
                self.worker_thread.result = None
                self.worker_thread.put_work(live)
            # TODO: better wait for first result from other thread
            if contour_mask is not None:
                # based on
                # https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
                # Convert uint8 to float
                foreground = live.astype(float)
                background = None
                if self.settings['use_green']:
                    background = green.astype(float)
                else:
                    background = base_blur.astype(float)

                # Normalize the alpha mask to keep intensity between 0 and 1
                alpha = contour_mask.astype(float) / 255

                # Multiply the foreground with the alpha matte
                foreground = cv2.multiply(alpha, foreground)

                # Multiply the background with ( 1 - alpha )
                background = cv2.multiply(1.0 - alpha, background)

                # Add the masked foreground and background.
                outImage = cv2.add(foreground, background).astype('uint8')

                masking = datetime.now(utc)

                self.frame_count += 1
                proc.stdin.write(outImage.tostring())
                writing = datetime.now(utc)

                reading_diff = reading - start
                masking_diff = masking - reading
                writing_diff = writing - masking
                total_diff = writing - start
                message = (
                    f"Reading took {reading_diff.total_seconds()*1000:.0f} ms, "
                    f"masking took {masking_diff.total_seconds()*1000:.0f} ms, "
                    f"writing took {writing_diff.total_seconds()*1000:.0f} ms, "
                    f"total: {total_diff.total_seconds()*1000:.0f} ms."
                )
                self.message_queue.put(message)
        cap.release()

    def join(self, timeout=None):
        self.stoprequest.set()
        super().join(timeout)

    def get_framecount(self):
        frame_count = self.frame_count
        self.frame_count = 0
        return frame_count


def handle_input(
    stdscr, image_compare_thread: ThreadWithSettingsAndMessages, masking_thread: ThreadWithSettingsAndMessages
):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    stdscr.bkgd(curses.color_pair(1))
    stdscr.nodelay(True)
    stdscr.refresh()

    # Hull info
    win1 = curses.newwin(5, 20, 0, 0)
    win1.bkgd(curses.color_pair(2))
    win1.box()
    win1.addstr(2, 2, "(H)ull inactive")
    win1.refresh()

    # And/Or info
    win2 = curses.newwin(5, 20, 0, 20)
    win2.bkgd(curses.color_pair(2))
    win2.box()
    win2.addstr(2, 2, "(A)nd inactive")
    win2.refresh()

    green_win = curses.newwin(5, 20, 0, 40)
    green_win.bkgd(curses.color_pair(2))
    green_win.box()
    green_win.addstr(2, 2, "(G)reen active")
    green_win.refresh()

    win3 = curses.newwin(5, 20, 0, 100)
    win3.bkgd(curses.color_pair(2))
    win3.box()
    win3.addstr(2, 2, "(Q)uit")
    win3.refresh()

    last_fps_update = datetime.now(utc)

    fps_win = curses.newwin(5, 20, 5, 00)
    fps_win.bkgd(curses.color_pair(2))
    fps_win.box()
    fps_win.refresh()

    # message windows
    masking_message_win = curses.newwin(10, 120, 10, 0)
    masking_message_win.bkgd(curses.color_pair(2))
    masking_message_win.box()
    masking_message_win.refresh()
    masking_message_queue = Queue()

    compare_message_win = curses.newwin(10, 120, 20, 0)
    compare_message_win.bkgd(curses.color_pair(2))
    compare_message_win.box()
    compare_message_win.refresh()
    compare_message_queue = Queue()

    while True:
        key = ''
        try:
            key = str(stdscr.getkey())
        except curses.error:
            pass
        if key == 'h':
            win1.clear()
            win1.box()
            if image_compare_thread.toggle_setting('use_hull'):
                win1.addstr(2, 2, "(H)ull active")
            else:
                win1.addstr(2, 2, "(H)ull inactive")
            win1.refresh()
        elif key == 'a':
            win2.clear()
            win2.box()
            if image_compare_thread.toggle_setting('use_and'):
                win2.addstr(2, 2, "(A)nd active")
            else:
                win2.addstr(2, 2, "(A)nd inactive")
            win2.refresh()
        elif key == 'g':
            green_win.clear()
            green_win.box()
            if masking_thread.toggle_setting('use_green'):
                green_win.addstr(2, 2, "(G)reen active")
            else:
                green_win.addstr(2, 2, "(G)reen inctive")
            green_win.refresh()
        if key == 'q':
            break
        if key == os.linesep:
            break

        if masking_thread.has_message():
            masking_message_win.clear()
            masking_message_win.box()
            masking_message_queue.put(masking_thread.get_message())
            messages = []
            for i in range(0, min(8, masking_message_queue.qsize())):
                message = masking_message_queue.get()
                masking_message_win.addstr(i + 1, 2, message)
                messages.append(message)
            masking_message_win.refresh()
            for message in messages:
                masking_message_queue.put(message)

        if image_compare_thread.has_message():
            compare_message_win.clear()
            compare_message_win.box()
            compare_message_queue.put(image_compare_thread.get_message())
            messages = []
            for i in range(0, min(8, compare_message_queue.qsize())):
                message = compare_message_queue.get()
                compare_message_win.addstr(i + 1, 2, message)
                messages.append(message)
            compare_message_win.refresh()
            for message in messages:
                compare_message_queue.put(message)

        # update fps counter once per second
        if datetime.now(utc) - last_fps_update > timedelta(seconds=1):
            frame_count = masking_thread.get_framecount()
            fps_win.clear()
            fps_win.box()
            fps_win.addstr(2, 2, f"{frame_count} fps")
            fps_win.refresh()
            last_fps_update = datetime.now(utc)

        stdscr.refresh()


def main(argv=None):
    image_compare_thread = ImageCompareThread()
    image_compare_thread.start()

    masking_thread = MaskingThread(image_compare_thread)
    masking_thread.start()

    try:
        curses.wrapper(handle_input, image_compare_thread, masking_thread)
    except KeyboardInterrupt:
        print("Exiting...")
    image_compare_thread.join()
    masking_thread.join()


if __name__ == '__main__':
    sys.exit(main())

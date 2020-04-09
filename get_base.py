#!/usr/bin/env python3

import argparse
import cv2
from datetime import datetime, timezone, timedelta

cap = cv2.VideoCapture(0)

start = datetime.now(timezone.utc)

while(True):
    # wait a few seconds to stabilize
    ret, frame = cap.read()
    now = datetime.now(timezone.utc)
    if now - start > timedelta(seconds=3):
        break

# Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the resulting frame
cv2.imshow('frame',gray)
if cv2.waitKey(1) & 0xFF == ord('q'):
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

cv2.imwrite("base.png", gray)

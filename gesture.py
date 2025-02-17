import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from fontTools.ttx import process

import numpy as np


def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle
def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return 0  # Handle cases where the input is invalid
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - x1)  # Euclidean distance
    return np.interp(L, [0, 1], [0, 1000])  # Normalize distance


mpHands=mp.solutions.hands
hands=mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
def okay_thumbs_up(landmark_list,thumb_index_dist):
    folded_fingers = (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) < 50 and
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50
    )


    thumb_extended = get_angle(landmark_list[1], landmark_list[2], landmark_list[4]) > 90


    thumb_far_from_index = thumb_index_dist > 50


    return folded_fingers and thumb_extended and thumb_far_from_index

def rock_sign(landmark_list,thumb_index_dist):
    thumb_far_from_index = thumb_index_dist > 90
    fingers=(
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12])<50 and
        get_angle(landmark_list[13], landmark_list[14], landmark_list[16])<50
    )
    index_extended=get_angle(landmark_list[5], landmark_list[6], landmark_list[8])>=90
    pinky_extended=get_angle(landmark_list[17], landmark_list[18], landmark_list[20])>=90
    return thumb_far_from_index and fingers and index_extended and pinky_extended
def bruh(landmark_list,thumb_index_dist):
    thumb_far_from_index = thumb_index_dist < 40
    fingers=(
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12])<50 and
        get_angle(landmark_list[13], landmark_list[14], landmark_list[16])<50
    )
    index_extended=get_angle(landmark_list[5], landmark_list[6], landmark_list[8])>=90
    pinky_extended=get_angle(landmark_list[17], landmark_list[18], landmark_list[20])>=90
    return thumb_far_from_index and fingers and index_extended and pinky_extended

def stop(landmark_list,thumb_index_dist):
    thumb_far_from_index = thumb_index_dist > 50
    fingers=(
            get_angle(landmark_list[1], landmark_list[2], landmark_list[4]) >= 90 and
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) >= 90 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) >= 90 and
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) >= 90 and
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) >= 90
    )
    return fingers and thumb_far_from_index
def got_it(landmark_list):
    # Check for thumb and index finger forming a circle
    thumb_index_close = get_distance([landmark_list[4], landmark_list[8]]) < 20  # Adjust the threshold as needed

    # Check if other fingers are extended
    middle_finger_extended = get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) >= 160
    ring_finger_extended = get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) >= 160
    pinky_finger_extended = get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) >= 160

    # Combine conditions
    fingers_extended = middle_finger_extended and ring_finger_extended and pinky_finger_extended
    return thumb_index_close and fingers_extended








def detect_gesture(frame,landmark_list,processed):
    if len(landmark_list)>=21:
        thumb_index_dist=get_distance([landmark_list[4],landmark_list[5]])
        if okay_thumbs_up(landmark_list,thumb_index_dist):
            cv2.putText(frame, "okay,thumbs up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif rock_sign(landmark_list,thumb_index_dist):
            cv2.putText(frame, "yoooooo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif stop(landmark_list,thumb_index_dist):
            cv2.putText(frame,"stop!!!",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif bruh(landmark_list,thumb_index_dist):
            cv2.putText(frame, "bruh :/", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif got_it(landmark_list):
            cv2.putText(frame, "got it ;)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




def main():
    cap=cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame=cv2.flip(frame,1)
            frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            processed=hands.process(frameRGB)
            landmark_list=[]
            if processed.multi_hand_landmarks:
                hand_landmarks=processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame,hand_landmarks,mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x,lm.y))
                detect_gesture(frame,landmark_list,processed)
            cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF==ord('x'):
                break;

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
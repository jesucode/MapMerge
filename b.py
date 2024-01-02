import cv2
import numpy as np

cap = cv2.VideoCapture('mapa.mp4')

def is_motion(frame1, frame2, threshold=25):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    count = np.sum(thresh == 255)
    return count > 0




static_frames = []
prev_frame = None
actual_frame = None

was_motion = False
is_motionb = False

framen = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[135:-102, :]

    if actual_frame is None and prev_frame is None: 
        prev_frame = frame.copy()
        continue

    if actual_frame is None : 
        actual_frame = frame.copy()
        was_motion = is_motion(prev_frame, actual_frame)
        continue
        
    is_motionb =is_motion(frame, actual_frame)

    if is_motionb  and not was_motion:
            print("appending" , framen)
            framen += 1
            static_frames.append(prev_frame.copy())
            cv2.imwrite(f'./f/static_frame_{framen}.png', prev_frame.copy())
    
    was_motion = is_motionb
    prev_frame = actual_frame
    actual_frame = frame
    

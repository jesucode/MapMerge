import pyautogui
import time
import random

num_presses_h = 12
num_presses_v = 10

def go_horizontal(direct):
    duration = 0.701  # in seconds
    delay_range2 = 0.001  # in seconds
    mean_delay = 1.1234  # in seconds
    delay_range = 0.50565  # in seconds


    pyautogui.keyDown(direct)  # simulate key down event
    time.sleep(duration + random.uniform(-delay_range2, delay_range2))  # hold down key for 2 seconds
    pyautogui.keyUp(direct)  # simulate key up event
    delay = mean_delay + random.uniform(-delay_range, delay_range)
    time.sleep(delay)  # wait for random delay before next press

def go_down():
    direct  = "down"
    duration = 1.241  # in seconds
    delay_range2 = 0.0001  # in seconds
    mean_delay = 0.8123  # in seconds
    delay_range = 0.5323  # in seconds

    pyautogui.keyDown(direct)  # simulate key down event
    time.sleep(duration + random.uniform(-delay_range2, delay_range2))  # hold down key for 2 seconds
    pyautogui.keyUp(direct)  # simulate key up event
    delay = mean_delay + random.uniform(-delay_range, delay_range)
    time.sleep(delay)  # wait for random delay before next press
time.sleep(5)


direc  = "right"
for i in range(num_presses_v):
    for j in range (num_presses_h):
        go_horizontal(direc)
    go_down()
    direc = "left" if direc  =="right" else "right" 


#for i in range(num_presses):
#    direct  = "right"
#    go_horizontal(direct)


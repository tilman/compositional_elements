from .types import *
from .generate.bisection import get_angle, get_bisection_cone, get_bisection_point, get_angle_in_respect_to_x, get_bisection_point_from_angle
import cv2
import numpy as np
# import numpy.typing as npt

title_window = 'bisec test'
height = 400
width = 600
#bgr
blue = (200, 100, 50)
darkblue = (150, 60, 30)
green = (100, 200, 50)
red = (100, 50, 220)
purple = (200, 50, 200)
yellow = (200, 250, 20)

a = Keypoint(50, 100)
b = Keypoint(200, 200)
c = Keypoint(40, 350)

img = np.array([[[220,220,220]]*width]*height, np.uint8)

def on_ax(val):
    global a
    a.x = val
    update()
def on_ay(val):
    global a
    a.y = val
    update()
def on_bx(val):
    global b
    b.x = val
    update()
def on_by(val):
    global b
    b.y = val
    update()
def on_cx(val):
    global c
    c.x = val
    update()
def on_cy(val):
    global c
    c.y = val
    update()


def pp(img, k: Keypoint, color):
    cv2.circle(img, (k.x, k.y), 2, color, -1)
    return img

def pl(img, k, l, color):
    cv2.line(img, (k.x, k.y), (l.x, l.y), color, 1)
    return img

def update():
    global img
    global a
    global b
    global c
    # reset window
    img = np.array([[[220,220,220]]*500]*400, np.uint8)
    print("")
    print("print(a, b, c)", a, b, c)
    img = pp(img, a, red)
    img = pl(img, a, b, red)
    img = pp(img, b, blue)
    img = pl(img, c, b, green)
    img = pp(img, c, green)

    phi = get_angle(a,b,c)
    
    bisect_point = get_bisection_point(a,b,c)
    img = pp(img, bisect_point, purple)
    img = pl(img, bisect_point, b, purple)

    cone = get_bisection_cone(a,b,c)
    # img = pp(img, cone2, yellow)
    # img = pl(img, cone2, b, yellow)
    cv2.polylines(img, [np.array(cone.exterior.coords[:-1], np.int)], True, yellow, 1)

    # done => get_angle_ground_normed
    # b_horizontal_reference = get_horizantal_b_reference(a,b,c)
    # gamma = get_angle(b_horizontal_reference, b, bisect_point)
    gamma = get_angle_in_respect_to_x(a,b,c)



    print(np.rad2deg(phi))
    print(np.rad2deg(gamma))
    # img = pp(img, b_horizontal_reference, darkblue)
    print("bisect_point",bisect_point)
    cv2.imshow(title_window, img)
    cv2.waitKey(25)

cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
cv2.createTrackbar('a x', title_window , a.x, width, on_ax)
cv2.createTrackbar('a y', title_window , a.y, height, on_ay)
cv2.createTrackbar('b x', title_window , b.x, width, on_bx)
cv2.createTrackbar('b y', title_window , b.y, height, on_by)
cv2.createTrackbar('c x', title_window , c.x, width, on_cx)
cv2.createTrackbar('c y', title_window , c.y, height, on_cy)
update()
cv2.waitKey(0)
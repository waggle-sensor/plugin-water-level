import glob
import cv2
import numpy as np
import argparse
from PIL import Image

import time

from waggle import Plugin
from waggle.data.vision import VideoCapture, resolve_device
from waggle.data.timestamp import get_timestamp

TOPIC_WATERLEVEL = "env.water.level"

def get_coordinates(args):
    original_coordinates = []
    new_coordinates = []
    pallet = []
    for c in args.roi_coordinates.strip().split(' '):
        x, y = c.split(',')
        original_coordinates.append([float(x), float(y)])
    for c in args.new_coordinates.strip().split(' '):
        x, y = c.split(',')
        new_coordinates.append([float(x), float(y)])
    x, y = args.pallet.strip().split(',')
    pallet.append((int(x), int(y)))
    return original_coordinates, new_coordinates, pallet


def calculation(i, args):
    match = {}
    pix = []
    h = []
    with open('mapping.txt', 'r') as f:
        for line in f:
            a = line.strip().split(',')
            match[int(a[0])] = int(a[1])
            pix.append(int(a[0]))
            h.append(int(a[1]))

    original_coordinates, new_coordinates, pallet = get_coordinates(args)

    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pts1 = np.float32(original_coordinates)
    pts2 = np.float32(new_coordinates)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, pallet[0])
    


    hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    avgs = np.average(s[300,:])
    avgv = np.average(v[300,:])


    if avgv < 170 and avgs > 20:
        filtersize = (40,40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filtersize)
        graydst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        tophat_image = cv2.morphologyEx(graydst, cv2.MORPH_TOPHAT, kernel)

        ret,new_tophat_image = cv2.threshold(tophat_image,45,255,cv2.THRESH_BINARY)

        kernel2 = np.ones((4,4), np.int8)
        closing = cv2.morphologyEx(new_tophat_image, cv2.MORPH_CLOSE, kernel2)
    else:
        graydst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        
        ret,new_tophat_image = cv2.threshold(graydst,85,255,cv2.THRESH_BINARY)

        kernel2 = np.ones((4,4), np.int8)
        closing = cv2.morphologyEx(new_tophat_image, cv2.MORPH_CLOSE, kernel2)
        

    wp = []
    for i in range(len(closing)):
        p = 0
        for j in range(len(closing[0])):
            if closing[i][j] == 0:
                p += 1
        wp.append(p)

    target = 0
    for i in range(len(wp)):
        if wp[i] > args.threshold:
            target = i
            break

    line = cv2.line(dst.copy(), (0,target), (100,target), (0,255,0), 1)

    if target == 0:
        return 'too dark', line


    hh = 0
    lh = 0
    measure = 0
    current = 0
    flag = False
    for i in range(len(pix)-1):
        if target == pix[i]:
            current = h[i]
            flag = True
        elif target == pix[i+1]:
            current = h[i+1]
            flag = True
        elif target < pix[i] and target > pix[i+1]:
            hh = pix[i]
            lh = pix[i+1]
            measure = h[i]
    if flag == True:
        flag = False
        return current, line
    else:
        return round((10/(hh-lh)*(hh-target) + measure), 2), line



def run(args):
    camera = Camera(args.stream)
    while True:
        sample = camera.snapshot()
        image = sample.data
        timestamp = sample.timestamp

        result_value, result_image = calculation(image, args)
        print(result_value)
    
        with Plugin() as plugin:
            plugin.publish(TOPIC_WATERLEVEL, result_value, timestamp=timestamp)
            print(f"Water level: {result_value} at time: {timestamp}")
            cv2.imwrite('watermarker.jpg', result_image)
            plugin.upload_file('watermarker.jpg')
            print('saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    parser.add_argument(
        '-threshold', dest='threshold',
        action='store', default=90, type=float,
        help='Cloud pixel determination threshold')
    parser.add_argument(
        '-roi-coordinates', dest='roi_coordinates',
        action='store', type=str, default="448,280 500,800 520,800 470,280",
        help='X,Y Coordinates of region of interest for perspective transform')
    parser.add_argument(
        '-new-coordinates', dest='new_coordinates',
        action='store', type=str, default="0,0 0,600 100,600 100,0",
        help='X,Y Coordinates of new region of interest for perspective transform')
    parser.add_argument(
        '-pallet', dest='pallet',
        action='store', type=str, default="100,780",
        help='X,Y Length of new pallet for perspective transform')
    run(parser.parse_args())



import numpy as np
import cv2 as cv
def morfologi(img, se):
    dst_erosi = cv.erode(img, se, iterations = 1)
    dst_dilate = cv.dilate(img, se, iterations = 1)
    dst_opening = cv.morphologyEx(img, cv.MORPH_OPEN, se, iterations = 1)
    dst_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, se, iterations = 1)
    return dst_erosi, dst_dilate, dst_opening, dst_closing
def main():
    img1 = cv.imread('circbw.tif',0)
    s_element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst_erosi, dst_dilate, dst_opening, dst_closing = morfologi(img1, s_element)
    cv.imshow('dst_erosi',dst_erosi)
    cv.imshow('dst_dilate',dst_dilate)
    cv.imshow('dst_opening',dst_opening)
    cv.imshow('dst_closing',dst_closing)
    cv.waitKey(0)
    cv.destroyAllWindows()
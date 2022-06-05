#-*-coding:utf-8-*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


# settings for LBP
radius = 2
n_points = 8 * radius
METHOD = 'uniform'
plt.rcParams['font.size'] = 9

def getdataTest (img):
    img_lbp = local_binary_pattern(img, n_points, radius, METHOD)
    img_lbp_hist,bins = np.histogram(img_lbp.ravel(),256,[0,256])
    img_lbp_hist=np.transpose(img_lbp_hist[0:18,np.newaxis])
    return img_lbp_hist

#Muat data gambar batik ceplok
tc1_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC1.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc2_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC2.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc3_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC3.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc4_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC4.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc5_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC5.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc6_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC6.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc7_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC7.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc8_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC8.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc9_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC9.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc10_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC10.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc11_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC11.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc12_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC12.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc13_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC13.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc14_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC14.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc15_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC15.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc16_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC16.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc17_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC17.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc18_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC18.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc19_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC19.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc20_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC20.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc21_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC21.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc22_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC22.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc23_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC23.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc24_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC24.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc25_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC25.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc26_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC26.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc27_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC27.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc28_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC28.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc29_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC29.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc30_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC30.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc31_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC31.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc32_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC32.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc33_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC33.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc34_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC34.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc35_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC35.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc36_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC36.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc37_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC37.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc38_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC38.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc39_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC39.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc40_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC40.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc41_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC41.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc42_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC42.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc43_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC43.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc44_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC44.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc45_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC45.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc46_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC46.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc47_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC47.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc48_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC48.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc49_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC49.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc50_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC50.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc51_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC51.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc52_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC52.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc53_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC53.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc54_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC54.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc55_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC55.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc56_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC56.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc57_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC57.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc58_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC58.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc59_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC59.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc60_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC60.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc61_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC61.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc62_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC62.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc63_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC63.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc64_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC64.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc65_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC65.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc66_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC66.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc67_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC67.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc68_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC68.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc69_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC69.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc70_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC70.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc71_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC71.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc72_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC72.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc73_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC73.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc74_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC74.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc75_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC75.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc76_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC76.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc77_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC77.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc78_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC78.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc79_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC79.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc80_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC80.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc81_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC81.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc82_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC82.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc83_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC83.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc84_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC84.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc85_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC85.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc86_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC86.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc87_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC87.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc88_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC88.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc89_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC89.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc90_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC90.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc91_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC91.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc92_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC92.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc93_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC93.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc94_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC94.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc95_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC95.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc96_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC96.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc97_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC97.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc98_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC98.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc99_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC99.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc100_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC100.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc101_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC101.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc102_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC102.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc103_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC103.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc104_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC104.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc105_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC105.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc106_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC106.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc107_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC107.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc108_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC108.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc109_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC109.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc110_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC110.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc111_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC111.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc112_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC112.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc113_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC113.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc114_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC114.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc115_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC115.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc116_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC116.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc117_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC117.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc118_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC118.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc119_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC119.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc120_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC120.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc121_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC121.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc122_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC122.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc123_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC123.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc124_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC124.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc125_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC125.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc126_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC126.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc127_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC127.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc128_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC128.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc129_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC129.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tc130_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Ceplok/TC130.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)





#Muat data gambar batik parang
tp1_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP1.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp2_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP2.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp3_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP3.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp4_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP4.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp5_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP5.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp6_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP6.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp7_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP7.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp8_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP8.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp9_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP9.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp10_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP10.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp11_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP11.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp12_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP12.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp13_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP13.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp14_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP14.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp15_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP15.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp16_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP16.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp17_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP17.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp18_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP18.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp19_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP19.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp20_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP20.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp21_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP21.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp22_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP22.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp23_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP23.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp24_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP24.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp25_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP25.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp26_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP26.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp27_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP27.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp28_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP28.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp29_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP29.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp30_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP30.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp31_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP31.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp32_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP32.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp33_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP33.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp34_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP34.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp35_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP35.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp36_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP36.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp37_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP37.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp38_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP38.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp39_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP39.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp40_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP40.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp41_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP41.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp42_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP42.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp43_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP43.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp44_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP44.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp45_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP45.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp46_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP46.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp47_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP47.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp48_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP48.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp49_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP49.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp50_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP50.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp51_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP51.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp52_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP52.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp53_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP53.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp54_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP54.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp55_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP55.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp56_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP56.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp57_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP57.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp58_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP58.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp59_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP59.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp60_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP60.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp61_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP61.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp62_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP62.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp63_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP63.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp64_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP64.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp65_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP65.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp66_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP66.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp67_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP67.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp68_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP68.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp69_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP69.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp70_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP70.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp71_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP71.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp72_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP72.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp73_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP73.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp74_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP74.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp75_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP75.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp76_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP76.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp77_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP77.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp78_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP78.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp79_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP79.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp80_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP80.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp81_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP81.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp82_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP82.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp83_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP83.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp84_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP84.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp85_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP85.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp86_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP86.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp87_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP87.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp88_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP88.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp89_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP89.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp90_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP90.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp91_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP91.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp92_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP92.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp93_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP93.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp94_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP94.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp95_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP95.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp96_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP96.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp97_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP97.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp98_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP98.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp99_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP99.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp100_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP100.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp101_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP101.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp102_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP102.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp103_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP103.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp104_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP104.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp105_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP105.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp106_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP106.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp107_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP107.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp108_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP108.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp109_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP109.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp110_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP110.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp111_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP111.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp112_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP112.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp113_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP113.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp114_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP114.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp115_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP115.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp116_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP116.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp117_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP117.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp118_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP118.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp119_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP119.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp120_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP120.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp121_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP121.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp122_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP122.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp123_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP123.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp124_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP124.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp125_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP125.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp126_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP126.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp127_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP127.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp128_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP128.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp129_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP129.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
tp130_lbp = local_binary_pattern(cv.imread('Dataset_projek/Training/Parang/TP130.jpg', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)

#histogram dari LBP untuk data kelas ceplok
tc1_lbp_hist,bins = np.histogram(tc1_lbp.ravel(),256,[0,256])
tc2_lbp_hist,bins = np.histogram(tc2_lbp.ravel(),256,[0,256])
tc3_lbp_hist,bins = np.histogram(tc3_lbp.ravel(),256,[0,256])
tc4_lbp_hist,bins = np.histogram(tc4_lbp.ravel(),256,[0,256])
tc5_lbp_hist,bins = np.histogram(tc5_lbp.ravel(),256,[0,256])
tc6_lbp_hist,bins = np.histogram(tc6_lbp.ravel(),256,[0,256])
tc7_lbp_hist,bins = np.histogram(tc7_lbp.ravel(),256,[0,256])
tc8_lbp_hist,bins = np.histogram(tc8_lbp.ravel(),256,[0,256])
tc9_lbp_hist,bins = np.histogram(tc9_lbp.ravel(),256,[0,256])
tc10_lbp_hist,bins = np.histogram(tc10_lbp.ravel(),256,[0,256])
tc11_lbp_hist,bins = np.histogram(tc11_lbp.ravel(),256,[0,256])
tc12_lbp_hist,bins = np.histogram(tc12_lbp.ravel(),256,[0,256])
tc13_lbp_hist,bins = np.histogram(tc13_lbp.ravel(),256,[0,256])
tc14_lbp_hist,bins = np.histogram(tc14_lbp.ravel(),256,[0,256])
tc15_lbp_hist,bins = np.histogram(tc15_lbp.ravel(),256,[0,256])
tc16_lbp_hist,bins = np.histogram(tc16_lbp.ravel(),256,[0,256])
tc17_lbp_hist,bins = np.histogram(tc17_lbp.ravel(),256,[0,256])
tc18_lbp_hist,bins = np.histogram(tc18_lbp.ravel(),256,[0,256])
tc19_lbp_hist,bins = np.histogram(tc19_lbp.ravel(),256,[0,256])
tc20_lbp_hist,bins = np.histogram(tc20_lbp.ravel(),256,[0,256])
tc21_lbp_hist,bins = np.histogram(tc21_lbp.ravel(),256,[0,256])
tc22_lbp_hist,bins = np.histogram(tc22_lbp.ravel(),256,[0,256])
tc23_lbp_hist,bins = np.histogram(tc23_lbp.ravel(),256,[0,256])
tc24_lbp_hist,bins = np.histogram(tc24_lbp.ravel(),256,[0,256])
tc25_lbp_hist,bins = np.histogram(tc25_lbp.ravel(),256,[0,256])
tc26_lbp_hist,bins = np.histogram(tc26_lbp.ravel(),256,[0,256])
tc27_lbp_hist,bins = np.histogram(tc27_lbp.ravel(),256,[0,256])
tc28_lbp_hist,bins = np.histogram(tc28_lbp.ravel(),256,[0,256])
tc29_lbp_hist,bins = np.histogram(tc29_lbp.ravel(),256,[0,256])
tc30_lbp_hist,bins = np.histogram(tc30_lbp.ravel(),256,[0,256])
tc31_lbp_hist,bins = np.histogram(tc31_lbp.ravel(),256,[0,256])
tc32_lbp_hist,bins = np.histogram(tc32_lbp.ravel(),256,[0,256])
tc33_lbp_hist,bins = np.histogram(tc33_lbp.ravel(),256,[0,256])
tc34_lbp_hist,bins = np.histogram(tc34_lbp.ravel(),256,[0,256])
tc35_lbp_hist,bins = np.histogram(tc35_lbp.ravel(),256,[0,256])
tc36_lbp_hist,bins = np.histogram(tc36_lbp.ravel(),256,[0,256])
tc37_lbp_hist,bins = np.histogram(tc37_lbp.ravel(),256,[0,256])
tc38_lbp_hist,bins = np.histogram(tc38_lbp.ravel(),256,[0,256])
tc39_lbp_hist,bins = np.histogram(tc39_lbp.ravel(),256,[0,256])
tc40_lbp_hist,bins = np.histogram(tc40_lbp.ravel(),256,[0,256])
tc41_lbp_hist,bins = np.histogram(tc41_lbp.ravel(),256,[0,256])
tc42_lbp_hist,bins = np.histogram(tc42_lbp.ravel(),256,[0,256])
tc43_lbp_hist,bins = np.histogram(tc43_lbp.ravel(),256,[0,256])
tc44_lbp_hist,bins = np.histogram(tc44_lbp.ravel(),256,[0,256])
tc45_lbp_hist,bins = np.histogram(tc45_lbp.ravel(),256,[0,256])
tc46_lbp_hist,bins = np.histogram(tc46_lbp.ravel(),256,[0,256])
tc47_lbp_hist,bins = np.histogram(tc47_lbp.ravel(),256,[0,256])
tc48_lbp_hist,bins = np.histogram(tc48_lbp.ravel(),256,[0,256])
tc49_lbp_hist,bins = np.histogram(tc49_lbp.ravel(),256,[0,256])
tc50_lbp_hist,bins = np.histogram(tc50_lbp.ravel(),256,[0,256])
tc51_lbp_hist,bins = np.histogram(tc51_lbp.ravel(),256,[0,256])
tc52_lbp_hist,bins = np.histogram(tc52_lbp.ravel(),256,[0,256])
tc53_lbp_hist,bins = np.histogram(tc53_lbp.ravel(),256,[0,256])
tc54_lbp_hist,bins = np.histogram(tc54_lbp.ravel(),256,[0,256])
tc55_lbp_hist,bins = np.histogram(tc55_lbp.ravel(),256,[0,256])
tc56_lbp_hist,bins = np.histogram(tc56_lbp.ravel(),256,[0,256])
tc57_lbp_hist,bins = np.histogram(tc57_lbp.ravel(),256,[0,256])
tc58_lbp_hist,bins = np.histogram(tc58_lbp.ravel(),256,[0,256])
tc59_lbp_hist,bins = np.histogram(tc59_lbp.ravel(),256,[0,256])
tc60_lbp_hist,bins = np.histogram(tc60_lbp.ravel(),256,[0,256])
tc61_lbp_hist,bins = np.histogram(tc61_lbp.ravel(),256,[0,256])
tc62_lbp_hist,bins = np.histogram(tc62_lbp.ravel(),256,[0,256])
tc63_lbp_hist,bins = np.histogram(tc63_lbp.ravel(),256,[0,256])
tc64_lbp_hist,bins = np.histogram(tc64_lbp.ravel(),256,[0,256])
tc65_lbp_hist,bins = np.histogram(tc65_lbp.ravel(),256,[0,256])
tc66_lbp_hist,bins = np.histogram(tc66_lbp.ravel(),256,[0,256])
tc67_lbp_hist,bins = np.histogram(tc67_lbp.ravel(),256,[0,256])
tc68_lbp_hist,bins = np.histogram(tc68_lbp.ravel(),256,[0,256])
tc69_lbp_hist,bins = np.histogram(tc69_lbp.ravel(),256,[0,256])
tc70_lbp_hist,bins = np.histogram(tc70_lbp.ravel(),256,[0,256])
tc71_lbp_hist,bins = np.histogram(tc71_lbp.ravel(),256,[0,256])
tc72_lbp_hist,bins = np.histogram(tc72_lbp.ravel(),256,[0,256])
tc73_lbp_hist,bins = np.histogram(tc73_lbp.ravel(),256,[0,256])
tc74_lbp_hist,bins = np.histogram(tc74_lbp.ravel(),256,[0,256])
tc75_lbp_hist,bins = np.histogram(tc75_lbp.ravel(),256,[0,256])
tc76_lbp_hist,bins = np.histogram(tc76_lbp.ravel(),256,[0,256])
tc77_lbp_hist,bins = np.histogram(tc77_lbp.ravel(),256,[0,256])
tc78_lbp_hist,bins = np.histogram(tc78_lbp.ravel(),256,[0,256])
tc79_lbp_hist,bins = np.histogram(tc79_lbp.ravel(),256,[0,256])
tc80_lbp_hist,bins = np.histogram(tc80_lbp.ravel(),256,[0,256])
tc81_lbp_hist,bins = np.histogram(tc81_lbp.ravel(),256,[0,256])
tc82_lbp_hist,bins = np.histogram(tc82_lbp.ravel(),256,[0,256])
tc83_lbp_hist,bins = np.histogram(tc83_lbp.ravel(),256,[0,256])
tc84_lbp_hist,bins = np.histogram(tc84_lbp.ravel(),256,[0,256])
tc85_lbp_hist,bins = np.histogram(tc85_lbp.ravel(),256,[0,256])
tc86_lbp_hist,bins = np.histogram(tc86_lbp.ravel(),256,[0,256])
tc87_lbp_hist,bins = np.histogram(tc87_lbp.ravel(),256,[0,256])
tc88_lbp_hist,bins = np.histogram(tc88_lbp.ravel(),256,[0,256])
tc89_lbp_hist,bins = np.histogram(tc89_lbp.ravel(),256,[0,256])
tc90_lbp_hist,bins = np.histogram(tc90_lbp.ravel(),256,[0,256])
tc91_lbp_hist,bins = np.histogram(tc91_lbp.ravel(),256,[0,256])
tc92_lbp_hist,bins = np.histogram(tc92_lbp.ravel(),256,[0,256])
tc93_lbp_hist,bins = np.histogram(tc93_lbp.ravel(),256,[0,256])
tc94_lbp_hist,bins = np.histogram(tc94_lbp.ravel(),256,[0,256])
tc95_lbp_hist,bins = np.histogram(tc95_lbp.ravel(),256,[0,256])
tc96_lbp_hist,bins = np.histogram(tc96_lbp.ravel(),256,[0,256])
tc97_lbp_hist,bins = np.histogram(tc97_lbp.ravel(),256,[0,256])
tc98_lbp_hist,bins = np.histogram(tc98_lbp.ravel(),256,[0,256])
tc99_lbp_hist,bins = np.histogram(tc99_lbp.ravel(),256,[0,256])
tc100_lbp_hist,bins = np.histogram(tc100_lbp.ravel(),256,[0,256])
tc101_lbp_hist,bins = np.histogram(tc101_lbp.ravel(),256,[0,256])
tc102_lbp_hist,bins = np.histogram(tc102_lbp.ravel(),256,[0,256])
tc103_lbp_hist,bins = np.histogram(tc103_lbp.ravel(),256,[0,256])
tc104_lbp_hist,bins = np.histogram(tc104_lbp.ravel(),256,[0,256])
tc105_lbp_hist,bins = np.histogram(tc105_lbp.ravel(),256,[0,256])
tc106_lbp_hist,bins = np.histogram(tc106_lbp.ravel(),256,[0,256])
tc107_lbp_hist,bins = np.histogram(tc107_lbp.ravel(),256,[0,256])
tc108_lbp_hist,bins = np.histogram(tc108_lbp.ravel(),256,[0,256])
tc109_lbp_hist,bins = np.histogram(tc109_lbp.ravel(),256,[0,256])
tc110_lbp_hist,bins = np.histogram(tc110_lbp.ravel(),256,[0,256])
tc111_lbp_hist,bins = np.histogram(tc111_lbp.ravel(),256,[0,256])
tc112_lbp_hist,bins = np.histogram(tc112_lbp.ravel(),256,[0,256])
tc113_lbp_hist,bins = np.histogram(tc113_lbp.ravel(),256,[0,256])
tc114_lbp_hist,bins = np.histogram(tc114_lbp.ravel(),256,[0,256])
tc115_lbp_hist,bins = np.histogram(tc115_lbp.ravel(),256,[0,256])
tc116_lbp_hist,bins = np.histogram(tc116_lbp.ravel(),256,[0,256])
tc117_lbp_hist,bins = np.histogram(tc117_lbp.ravel(),256,[0,256])
tc118_lbp_hist,bins = np.histogram(tc118_lbp.ravel(),256,[0,256])
tc119_lbp_hist,bins = np.histogram(tc119_lbp.ravel(),256,[0,256])
tc120_lbp_hist,bins = np.histogram(tc120_lbp.ravel(),256,[0,256])
tc121_lbp_hist,bins = np.histogram(tc121_lbp.ravel(),256,[0,256])
tc122_lbp_hist,bins = np.histogram(tc122_lbp.ravel(),256,[0,256])
tc123_lbp_hist,bins = np.histogram(tc123_lbp.ravel(),256,[0,256])
tc124_lbp_hist,bins = np.histogram(tc124_lbp.ravel(),256,[0,256])
tc125_lbp_hist,bins = np.histogram(tc125_lbp.ravel(),256,[0,256])
tc126_lbp_hist,bins = np.histogram(tc126_lbp.ravel(),256,[0,256])
tc127_lbp_hist,bins = np.histogram(tc127_lbp.ravel(),256,[0,256])
tc128_lbp_hist,bins = np.histogram(tc128_lbp.ravel(),256,[0,256])
tc129_lbp_hist,bins = np.histogram(tc129_lbp.ravel(),256,[0,256])
tc130_lbp_hist,bins = np.histogram(tc130_lbp.ravel(),256,[0,256])

#histogram dari LBP untuk data kelas parang
tp1_lbp_hist,bins = np.histogram(tp1_lbp.ravel(),256,[0,256])
tp2_lbp_hist,bins = np.histogram(tp2_lbp.ravel(),256,[0,256])
tp3_lbp_hist,bins = np.histogram(tp3_lbp.ravel(),256,[0,256])
tp4_lbp_hist,bins = np.histogram(tp4_lbp.ravel(),256,[0,256])
tp5_lbp_hist,bins = np.histogram(tp5_lbp.ravel(),256,[0,256])
tp6_lbp_hist,bins = np.histogram(tp6_lbp.ravel(),256,[0,256])
tp7_lbp_hist,bins = np.histogram(tp7_lbp.ravel(),256,[0,256])
tp8_lbp_hist,bins = np.histogram(tp8_lbp.ravel(),256,[0,256])
tp9_lbp_hist,bins = np.histogram(tp9_lbp.ravel(),256,[0,256])
tp10_lbp_hist,bins = np.histogram(tp10_lbp.ravel(),256,[0,256])
tp11_lbp_hist,bins = np.histogram(tp11_lbp.ravel(),256,[0,256])
tp12_lbp_hist,bins = np.histogram(tp12_lbp.ravel(),256,[0,256])
tp13_lbp_hist,bins = np.histogram(tp13_lbp.ravel(),256,[0,256])
tp14_lbp_hist,bins = np.histogram(tp14_lbp.ravel(),256,[0,256])
tp15_lbp_hist,bins = np.histogram(tp15_lbp.ravel(),256,[0,256])
tp16_lbp_hist,bins = np.histogram(tp16_lbp.ravel(),256,[0,256])
tp17_lbp_hist,bins = np.histogram(tp17_lbp.ravel(),256,[0,256])
tp18_lbp_hist,bins = np.histogram(tp18_lbp.ravel(),256,[0,256])
tp19_lbp_hist,bins = np.histogram(tp19_lbp.ravel(),256,[0,256])
tp20_lbp_hist,bins = np.histogram(tp20_lbp.ravel(),256,[0,256])
tp21_lbp_hist,bins = np.histogram(tp21_lbp.ravel(),256,[0,256])
tp22_lbp_hist,bins = np.histogram(tp22_lbp.ravel(),256,[0,256])
tp23_lbp_hist,bins = np.histogram(tp23_lbp.ravel(),256,[0,256])
tp24_lbp_hist,bins = np.histogram(tp24_lbp.ravel(),256,[0,256])
tp25_lbp_hist,bins = np.histogram(tp25_lbp.ravel(),256,[0,256])
tp26_lbp_hist,bins = np.histogram(tp26_lbp.ravel(),256,[0,256])
tp27_lbp_hist,bins = np.histogram(tp27_lbp.ravel(),256,[0,256])
tp28_lbp_hist,bins = np.histogram(tp28_lbp.ravel(),256,[0,256])
tp29_lbp_hist,bins = np.histogram(tp29_lbp.ravel(),256,[0,256])
tp30_lbp_hist,bins = np.histogram(tp30_lbp.ravel(),256,[0,256])
tp31_lbp_hist,bins = np.histogram(tp31_lbp.ravel(),256,[0,256])
tp32_lbp_hist,bins = np.histogram(tp32_lbp.ravel(),256,[0,256])
tp33_lbp_hist,bins = np.histogram(tp33_lbp.ravel(),256,[0,256])
tp34_lbp_hist,bins = np.histogram(tp34_lbp.ravel(),256,[0,256])
tp35_lbp_hist,bins = np.histogram(tp35_lbp.ravel(),256,[0,256])
tp36_lbp_hist,bins = np.histogram(tp36_lbp.ravel(),256,[0,256])
tp37_lbp_hist,bins = np.histogram(tp37_lbp.ravel(),256,[0,256])
tp38_lbp_hist,bins = np.histogram(tp38_lbp.ravel(),256,[0,256])
tp39_lbp_hist,bins = np.histogram(tp39_lbp.ravel(),256,[0,256])
tp40_lbp_hist,bins = np.histogram(tp40_lbp.ravel(),256,[0,256])
tp41_lbp_hist,bins = np.histogram(tp41_lbp.ravel(),256,[0,256])
tp42_lbp_hist,bins = np.histogram(tp42_lbp.ravel(),256,[0,256])
tp43_lbp_hist,bins = np.histogram(tp43_lbp.ravel(),256,[0,256])
tp44_lbp_hist,bins = np.histogram(tp44_lbp.ravel(),256,[0,256])
tp45_lbp_hist,bins = np.histogram(tp45_lbp.ravel(),256,[0,256])
tp46_lbp_hist,bins = np.histogram(tp46_lbp.ravel(),256,[0,256])
tp47_lbp_hist,bins = np.histogram(tp47_lbp.ravel(),256,[0,256])
tp48_lbp_hist,bins = np.histogram(tp48_lbp.ravel(),256,[0,256])
tp49_lbp_hist,bins = np.histogram(tp49_lbp.ravel(),256,[0,256])
tp50_lbp_hist,bins = np.histogram(tp50_lbp.ravel(),256,[0,256])
tp51_lbp_hist,bins = np.histogram(tp51_lbp.ravel(),256,[0,256])
tp52_lbp_hist,bins = np.histogram(tp52_lbp.ravel(),256,[0,256])
tp53_lbp_hist,bins = np.histogram(tp53_lbp.ravel(),256,[0,256])
tp54_lbp_hist,bins = np.histogram(tp54_lbp.ravel(),256,[0,256])
tp55_lbp_hist,bins = np.histogram(tp55_lbp.ravel(),256,[0,256])
tp56_lbp_hist,bins = np.histogram(tp56_lbp.ravel(),256,[0,256])
tp57_lbp_hist,bins = np.histogram(tp57_lbp.ravel(),256,[0,256])
tp58_lbp_hist,bins = np.histogram(tp58_lbp.ravel(),256,[0,256])
tp59_lbp_hist,bins = np.histogram(tp59_lbp.ravel(),256,[0,256])
tp60_lbp_hist,bins = np.histogram(tp60_lbp.ravel(),256,[0,256])
tp61_lbp_hist,bins = np.histogram(tp61_lbp.ravel(),256,[0,256])
tp62_lbp_hist,bins = np.histogram(tp62_lbp.ravel(),256,[0,256])
tp63_lbp_hist,bins = np.histogram(tp63_lbp.ravel(),256,[0,256])
tp64_lbp_hist,bins = np.histogram(tp64_lbp.ravel(),256,[0,256])
tp65_lbp_hist,bins = np.histogram(tp65_lbp.ravel(),256,[0,256])
tp66_lbp_hist,bins = np.histogram(tp66_lbp.ravel(),256,[0,256])
tp67_lbp_hist,bins = np.histogram(tp67_lbp.ravel(),256,[0,256])
tp68_lbp_hist,bins = np.histogram(tp68_lbp.ravel(),256,[0,256])
tp69_lbp_hist,bins = np.histogram(tp69_lbp.ravel(),256,[0,256])
tp70_lbp_hist,bins = np.histogram(tp70_lbp.ravel(),256,[0,256])
tp71_lbp_hist,bins = np.histogram(tp71_lbp.ravel(),256,[0,256])
tp72_lbp_hist,bins = np.histogram(tp72_lbp.ravel(),256,[0,256])
tp73_lbp_hist,bins = np.histogram(tp73_lbp.ravel(),256,[0,256])
tp74_lbp_hist,bins = np.histogram(tp74_lbp.ravel(),256,[0,256])
tp75_lbp_hist,bins = np.histogram(tp75_lbp.ravel(),256,[0,256])
tp76_lbp_hist,bins = np.histogram(tp76_lbp.ravel(),256,[0,256])
tp77_lbp_hist,bins = np.histogram(tp77_lbp.ravel(),256,[0,256])
tp78_lbp_hist,bins = np.histogram(tp78_lbp.ravel(),256,[0,256])
tp79_lbp_hist,bins = np.histogram(tp79_lbp.ravel(),256,[0,256])
tp80_lbp_hist,bins = np.histogram(tp80_lbp.ravel(),256,[0,256])
tp81_lbp_hist,bins = np.histogram(tp81_lbp.ravel(),256,[0,256])
tp82_lbp_hist,bins = np.histogram(tp82_lbp.ravel(),256,[0,256])
tp83_lbp_hist,bins = np.histogram(tp83_lbp.ravel(),256,[0,256])
tp84_lbp_hist,bins = np.histogram(tp84_lbp.ravel(),256,[0,256])
tp85_lbp_hist,bins = np.histogram(tp85_lbp.ravel(),256,[0,256])
tp86_lbp_hist,bins = np.histogram(tp86_lbp.ravel(),256,[0,256])
tp87_lbp_hist,bins = np.histogram(tp87_lbp.ravel(),256,[0,256])
tp88_lbp_hist,bins = np.histogram(tp88_lbp.ravel(),256,[0,256])
tp89_lbp_hist,bins = np.histogram(tp89_lbp.ravel(),256,[0,256])
tp90_lbp_hist,bins = np.histogram(tp90_lbp.ravel(),256,[0,256])
tp91_lbp_hist,bins = np.histogram(tp91_lbp.ravel(),256,[0,256])
tp92_lbp_hist,bins = np.histogram(tp92_lbp.ravel(),256,[0,256])
tp93_lbp_hist,bins = np.histogram(tp93_lbp.ravel(),256,[0,256])
tp94_lbp_hist,bins = np.histogram(tp94_lbp.ravel(),256,[0,256])
tp95_lbp_hist,bins = np.histogram(tp95_lbp.ravel(),256,[0,256])
tp96_lbp_hist,bins = np.histogram(tp96_lbp.ravel(),256,[0,256])
tp97_lbp_hist,bins = np.histogram(tp97_lbp.ravel(),256,[0,256])
tp98_lbp_hist,bins = np.histogram(tp98_lbp.ravel(),256,[0,256])
tp99_lbp_hist,bins = np.histogram(tp99_lbp.ravel(),256,[0,256])
tp100_lbp_hist,bins = np.histogram(tp100_lbp.ravel(),256,[0,256])
tp101_lbp_hist,bins = np.histogram(tp101_lbp.ravel(),256,[0,256])
tp102_lbp_hist,bins = np.histogram(tp102_lbp.ravel(),256,[0,256])
tp103_lbp_hist,bins = np.histogram(tp103_lbp.ravel(),256,[0,256])
tp104_lbp_hist,bins = np.histogram(tp104_lbp.ravel(),256,[0,256])
tp105_lbp_hist,bins = np.histogram(tp105_lbp.ravel(),256,[0,256])
tp106_lbp_hist,bins = np.histogram(tp106_lbp.ravel(),256,[0,256])
tp107_lbp_hist,bins = np.histogram(tp107_lbp.ravel(),256,[0,256])
tp108_lbp_hist,bins = np.histogram(tp108_lbp.ravel(),256,[0,256])
tp109_lbp_hist,bins = np.histogram(tp109_lbp.ravel(),256,[0,256])
tp110_lbp_hist,bins = np.histogram(tp110_lbp.ravel(),256,[0,256])
tp111_lbp_hist,bins = np.histogram(tp111_lbp.ravel(),256,[0,256])
tp112_lbp_hist,bins = np.histogram(tp112_lbp.ravel(),256,[0,256])
tp113_lbp_hist,bins = np.histogram(tp113_lbp.ravel(),256,[0,256])
tp114_lbp_hist,bins = np.histogram(tp114_lbp.ravel(),256,[0,256])
tp115_lbp_hist,bins = np.histogram(tp115_lbp.ravel(),256,[0,256])
tp116_lbp_hist,bins = np.histogram(tp116_lbp.ravel(),256,[0,256])
tp117_lbp_hist,bins = np.histogram(tp117_lbp.ravel(),256,[0,256])
tp118_lbp_hist,bins = np.histogram(tp118_lbp.ravel(),256,[0,256])
tp119_lbp_hist,bins = np.histogram(tp119_lbp.ravel(),256,[0,256])
tp120_lbp_hist,bins = np.histogram(tp120_lbp.ravel(),256,[0,256])
tp121_lbp_hist,bins = np.histogram(tp121_lbp.ravel(),256,[0,256])
tp122_lbp_hist,bins = np.histogram(tp122_lbp.ravel(),256,[0,256])
tp123_lbp_hist,bins = np.histogram(tp123_lbp.ravel(),256,[0,256])
tp124_lbp_hist,bins = np.histogram(tp124_lbp.ravel(),256,[0,256])
tp125_lbp_hist,bins = np.histogram(tp125_lbp.ravel(),256,[0,256])
tp126_lbp_hist,bins = np.histogram(tp126_lbp.ravel(),256,[0,256])
tp127_lbp_hist,bins = np.histogram(tp127_lbp.ravel(),256,[0,256])
tp128_lbp_hist,bins = np.histogram(tp128_lbp.ravel(),256,[0,256])
tp129_lbp_hist,bins = np.histogram(tp129_lbp.ravel(),256,[0,256])
tp130_lbp_hist,bins = np.histogram(tp130_lbp.ravel(),256,[0,256])

#transpose dari LBP untuk data kelas ceplok
tc1_lbp_hist=np.transpose(tc1_lbp_hist[0:18,np.newaxis])
tc2_lbp_hist=np.transpose(tc2_lbp_hist[0:18,np.newaxis])
tc3_lbp_hist=np.transpose(tc3_lbp_hist[0:18,np.newaxis])
tc4_lbp_hist=np.transpose(tc4_lbp_hist[0:18,np.newaxis])
tc5_lbp_hist=np.transpose(tc5_lbp_hist[0:18,np.newaxis])
tc6_lbp_hist=np.transpose(tc6_lbp_hist[0:18,np.newaxis])
tc7_lbp_hist=np.transpose(tc7_lbp_hist[0:18,np.newaxis])
tc8_lbp_hist=np.transpose(tc8_lbp_hist[0:18,np.newaxis])
tc9_lbp_hist=np.transpose(tc9_lbp_hist[0:18,np.newaxis])
tc10_lbp_hist=np.transpose(tc10_lbp_hist[0:18,np.newaxis])
tc11_lbp_hist=np.transpose(tc11_lbp_hist[0:18,np.newaxis])
tc12_lbp_hist=np.transpose(tc12_lbp_hist[0:18,np.newaxis])
tc13_lbp_hist=np.transpose(tc13_lbp_hist[0:18,np.newaxis])
tc14_lbp_hist=np.transpose(tc14_lbp_hist[0:18,np.newaxis])
tc15_lbp_hist=np.transpose(tc15_lbp_hist[0:18,np.newaxis])
tc16_lbp_hist=np.transpose(tc16_lbp_hist[0:18,np.newaxis])
tc17_lbp_hist=np.transpose(tc17_lbp_hist[0:18,np.newaxis])
tc18_lbp_hist=np.transpose(tc18_lbp_hist[0:18,np.newaxis])
tc19_lbp_hist=np.transpose(tc19_lbp_hist[0:18,np.newaxis])
tc20_lbp_hist=np.transpose(tc20_lbp_hist[0:18,np.newaxis])
tc21_lbp_hist=np.transpose(tc21_lbp_hist[0:18,np.newaxis])
tc22_lbp_hist=np.transpose(tc22_lbp_hist[0:18,np.newaxis])
tc23_lbp_hist=np.transpose(tc23_lbp_hist[0:18,np.newaxis])
tc24_lbp_hist=np.transpose(tc24_lbp_hist[0:18,np.newaxis])
tc25_lbp_hist=np.transpose(tc25_lbp_hist[0:18,np.newaxis])
tc26_lbp_hist=np.transpose(tc26_lbp_hist[0:18,np.newaxis])
tc27_lbp_hist=np.transpose(tc27_lbp_hist[0:18,np.newaxis])
tc28_lbp_hist=np.transpose(tc28_lbp_hist[0:18,np.newaxis])
tc29_lbp_hist=np.transpose(tc29_lbp_hist[0:18,np.newaxis])
tc30_lbp_hist=np.transpose(tc30_lbp_hist[0:18,np.newaxis])
tc31_lbp_hist=np.transpose(tc31_lbp_hist[0:18,np.newaxis])
tc32_lbp_hist=np.transpose(tc32_lbp_hist[0:18,np.newaxis])
tc33_lbp_hist=np.transpose(tc33_lbp_hist[0:18,np.newaxis])
tc34_lbp_hist=np.transpose(tc34_lbp_hist[0:18,np.newaxis])
tc35_lbp_hist=np.transpose(tc35_lbp_hist[0:18,np.newaxis])
tc36_lbp_hist=np.transpose(tc36_lbp_hist[0:18,np.newaxis])
tc37_lbp_hist=np.transpose(tc37_lbp_hist[0:18,np.newaxis])
tc38_lbp_hist=np.transpose(tc38_lbp_hist[0:18,np.newaxis])
tc39_lbp_hist=np.transpose(tc39_lbp_hist[0:18,np.newaxis])
tc40_lbp_hist=np.transpose(tc40_lbp_hist[0:18,np.newaxis])
tc41_lbp_hist=np.transpose(tc41_lbp_hist[0:18,np.newaxis])
tc42_lbp_hist=np.transpose(tc42_lbp_hist[0:18,np.newaxis])
tc43_lbp_hist=np.transpose(tc43_lbp_hist[0:18,np.newaxis])
tc44_lbp_hist=np.transpose(tc44_lbp_hist[0:18,np.newaxis])
tc45_lbp_hist=np.transpose(tc45_lbp_hist[0:18,np.newaxis])
tc46_lbp_hist=np.transpose(tc46_lbp_hist[0:18,np.newaxis])
tc47_lbp_hist=np.transpose(tc47_lbp_hist[0:18,np.newaxis])
tc48_lbp_hist=np.transpose(tc48_lbp_hist[0:18,np.newaxis])
tc49_lbp_hist=np.transpose(tc49_lbp_hist[0:18,np.newaxis])
tc50_lbp_hist=np.transpose(tc50_lbp_hist[0:18,np.newaxis])
tc51_lbp_hist=np.transpose(tc51_lbp_hist[0:18,np.newaxis])
tc52_lbp_hist=np.transpose(tc52_lbp_hist[0:18,np.newaxis])
tc53_lbp_hist=np.transpose(tc53_lbp_hist[0:18,np.newaxis])
tc54_lbp_hist=np.transpose(tc54_lbp_hist[0:18,np.newaxis])
tc55_lbp_hist=np.transpose(tc55_lbp_hist[0:18,np.newaxis])
tc56_lbp_hist=np.transpose(tc56_lbp_hist[0:18,np.newaxis])
tc57_lbp_hist=np.transpose(tc57_lbp_hist[0:18,np.newaxis])
tc58_lbp_hist=np.transpose(tc58_lbp_hist[0:18,np.newaxis])
tc59_lbp_hist=np.transpose(tc59_lbp_hist[0:18,np.newaxis])
tc60_lbp_hist=np.transpose(tc60_lbp_hist[0:18,np.newaxis])
tc61_lbp_hist=np.transpose(tc61_lbp_hist[0:18,np.newaxis])
tc62_lbp_hist=np.transpose(tc62_lbp_hist[0:18,np.newaxis])
tc63_lbp_hist=np.transpose(tc63_lbp_hist[0:18,np.newaxis])
tc64_lbp_hist=np.transpose(tc64_lbp_hist[0:18,np.newaxis])
tc65_lbp_hist=np.transpose(tc65_lbp_hist[0:18,np.newaxis])
tc66_lbp_hist=np.transpose(tc66_lbp_hist[0:18,np.newaxis])
tc67_lbp_hist=np.transpose(tc67_lbp_hist[0:18,np.newaxis])
tc68_lbp_hist=np.transpose(tc68_lbp_hist[0:18,np.newaxis])
tc69_lbp_hist=np.transpose(tc69_lbp_hist[0:18,np.newaxis])
tc70_lbp_hist=np.transpose(tc70_lbp_hist[0:18,np.newaxis])
tc71_lbp_hist=np.transpose(tc71_lbp_hist[0:18,np.newaxis])
tc72_lbp_hist=np.transpose(tc72_lbp_hist[0:18,np.newaxis])
tc73_lbp_hist=np.transpose(tc73_lbp_hist[0:18,np.newaxis])
tc74_lbp_hist=np.transpose(tc74_lbp_hist[0:18,np.newaxis])
tc75_lbp_hist=np.transpose(tc75_lbp_hist[0:18,np.newaxis])
tc76_lbp_hist=np.transpose(tc76_lbp_hist[0:18,np.newaxis])
tc77_lbp_hist=np.transpose(tc77_lbp_hist[0:18,np.newaxis])
tc78_lbp_hist=np.transpose(tc78_lbp_hist[0:18,np.newaxis])
tc79_lbp_hist=np.transpose(tc79_lbp_hist[0:18,np.newaxis])
tc80_lbp_hist=np.transpose(tc80_lbp_hist[0:18,np.newaxis])
tc81_lbp_hist=np.transpose(tc81_lbp_hist[0:18,np.newaxis])
tc82_lbp_hist=np.transpose(tc82_lbp_hist[0:18,np.newaxis])
tc83_lbp_hist=np.transpose(tc83_lbp_hist[0:18,np.newaxis])
tc84_lbp_hist=np.transpose(tc84_lbp_hist[0:18,np.newaxis])
tc85_lbp_hist=np.transpose(tc85_lbp_hist[0:18,np.newaxis])
tc86_lbp_hist=np.transpose(tc86_lbp_hist[0:18,np.newaxis])
tc87_lbp_hist=np.transpose(tc87_lbp_hist[0:18,np.newaxis])
tc88_lbp_hist=np.transpose(tc88_lbp_hist[0:18,np.newaxis])
tc89_lbp_hist=np.transpose(tc89_lbp_hist[0:18,np.newaxis])
tc90_lbp_hist=np.transpose(tc90_lbp_hist[0:18,np.newaxis])
tc91_lbp_hist=np.transpose(tc91_lbp_hist[0:18,np.newaxis])
tc92_lbp_hist=np.transpose(tc92_lbp_hist[0:18,np.newaxis])
tc93_lbp_hist=np.transpose(tc93_lbp_hist[0:18,np.newaxis])
tc94_lbp_hist=np.transpose(tc94_lbp_hist[0:18,np.newaxis])
tc95_lbp_hist=np.transpose(tc95_lbp_hist[0:18,np.newaxis])
tc96_lbp_hist=np.transpose(tc96_lbp_hist[0:18,np.newaxis])
tc97_lbp_hist=np.transpose(tc97_lbp_hist[0:18,np.newaxis])
tc98_lbp_hist=np.transpose(tc98_lbp_hist[0:18,np.newaxis])
tc99_lbp_hist=np.transpose(tc99_lbp_hist[0:18,np.newaxis])
tc100_lbp_hist=np.transpose(tc100_lbp_hist[0:18,np.newaxis])
tc101_lbp_hist=np.transpose(tc101_lbp_hist[0:18,np.newaxis])
tc102_lbp_hist=np.transpose(tc102_lbp_hist[0:18,np.newaxis])
tc103_lbp_hist=np.transpose(tc103_lbp_hist[0:18,np.newaxis])
tc104_lbp_hist=np.transpose(tc104_lbp_hist[0:18,np.newaxis])
tc105_lbp_hist=np.transpose(tc105_lbp_hist[0:18,np.newaxis])
tc106_lbp_hist=np.transpose(tc106_lbp_hist[0:18,np.newaxis])
tc107_lbp_hist=np.transpose(tc107_lbp_hist[0:18,np.newaxis])
tc108_lbp_hist=np.transpose(tc108_lbp_hist[0:18,np.newaxis])
tc109_lbp_hist=np.transpose(tc109_lbp_hist[0:18,np.newaxis])
tc110_lbp_hist=np.transpose(tc110_lbp_hist[0:18,np.newaxis])
tc111_lbp_hist=np.transpose(tc111_lbp_hist[0:18,np.newaxis])
tc112_lbp_hist=np.transpose(tc112_lbp_hist[0:18,np.newaxis])
tc113_lbp_hist=np.transpose(tc113_lbp_hist[0:18,np.newaxis])
tc114_lbp_hist=np.transpose(tc114_lbp_hist[0:18,np.newaxis])
tc115_lbp_hist=np.transpose(tc115_lbp_hist[0:18,np.newaxis])
tc116_lbp_hist=np.transpose(tc116_lbp_hist[0:18,np.newaxis])
tc117_lbp_hist=np.transpose(tc117_lbp_hist[0:18,np.newaxis])
tc118_lbp_hist=np.transpose(tc118_lbp_hist[0:18,np.newaxis])
tc119_lbp_hist=np.transpose(tc119_lbp_hist[0:18,np.newaxis])
tc120_lbp_hist=np.transpose(tc120_lbp_hist[0:18,np.newaxis])
tc121_lbp_hist=np.transpose(tc121_lbp_hist[0:18,np.newaxis])
tc122_lbp_hist=np.transpose(tc122_lbp_hist[0:18,np.newaxis])
tc123_lbp_hist=np.transpose(tc123_lbp_hist[0:18,np.newaxis])
tc124_lbp_hist=np.transpose(tc124_lbp_hist[0:18,np.newaxis])
tc125_lbp_hist=np.transpose(tc125_lbp_hist[0:18,np.newaxis])
tc126_lbp_hist=np.transpose(tc126_lbp_hist[0:18,np.newaxis])
tc127_lbp_hist=np.transpose(tc127_lbp_hist[0:18,np.newaxis])
tc128_lbp_hist=np.transpose(tc128_lbp_hist[0:18,np.newaxis])
tc129_lbp_hist=np.transpose(tc129_lbp_hist[0:18,np.newaxis])
tc130_lbp_hist=np.transpose(tc130_lbp_hist[0:18,np.newaxis])

#transpose dari LBP untuk data kelas parang
tp1_lbp_hist=np.transpose(tp1_lbp_hist[0:18,np.newaxis])
tp2_lbp_hist=np.transpose(tp2_lbp_hist[0:18,np.newaxis])
tp3_lbp_hist=np.transpose(tp3_lbp_hist[0:18,np.newaxis])
tp4_lbp_hist=np.transpose(tp4_lbp_hist[0:18,np.newaxis])
tp5_lbp_hist=np.transpose(tp5_lbp_hist[0:18,np.newaxis])
tp6_lbp_hist=np.transpose(tp6_lbp_hist[0:18,np.newaxis])
tp7_lbp_hist=np.transpose(tp7_lbp_hist[0:18,np.newaxis])
tp8_lbp_hist=np.transpose(tp8_lbp_hist[0:18,np.newaxis])
tp9_lbp_hist=np.transpose(tp9_lbp_hist[0:18,np.newaxis])
tp10_lbp_hist=np.transpose(tp10_lbp_hist[0:18,np.newaxis])
tp11_lbp_hist=np.transpose(tp11_lbp_hist[0:18,np.newaxis])
tp12_lbp_hist=np.transpose(tp12_lbp_hist[0:18,np.newaxis])
tp13_lbp_hist=np.transpose(tp13_lbp_hist[0:18,np.newaxis])
tp14_lbp_hist=np.transpose(tp14_lbp_hist[0:18,np.newaxis])
tp15_lbp_hist=np.transpose(tp15_lbp_hist[0:18,np.newaxis])
tp16_lbp_hist=np.transpose(tp16_lbp_hist[0:18,np.newaxis])
tp17_lbp_hist=np.transpose(tp17_lbp_hist[0:18,np.newaxis])
tp18_lbp_hist=np.transpose(tp18_lbp_hist[0:18,np.newaxis])
tp19_lbp_hist=np.transpose(tp19_lbp_hist[0:18,np.newaxis])
tp20_lbp_hist=np.transpose(tp20_lbp_hist[0:18,np.newaxis])
tp21_lbp_hist=np.transpose(tp21_lbp_hist[0:18,np.newaxis])
tp22_lbp_hist=np.transpose(tp22_lbp_hist[0:18,np.newaxis])
tp23_lbp_hist=np.transpose(tp23_lbp_hist[0:18,np.newaxis])
tp24_lbp_hist=np.transpose(tp24_lbp_hist[0:18,np.newaxis])
tp25_lbp_hist=np.transpose(tp25_lbp_hist[0:18,np.newaxis])
tp26_lbp_hist=np.transpose(tp26_lbp_hist[0:18,np.newaxis])
tp27_lbp_hist=np.transpose(tp27_lbp_hist[0:18,np.newaxis])
tp28_lbp_hist=np.transpose(tp28_lbp_hist[0:18,np.newaxis])
tp29_lbp_hist=np.transpose(tp29_lbp_hist[0:18,np.newaxis])
tp30_lbp_hist=np.transpose(tp30_lbp_hist[0:18,np.newaxis])
tp31_lbp_hist=np.transpose(tp31_lbp_hist[0:18,np.newaxis])
tp32_lbp_hist=np.transpose(tp32_lbp_hist[0:18,np.newaxis])
tp33_lbp_hist=np.transpose(tp33_lbp_hist[0:18,np.newaxis])
tp34_lbp_hist=np.transpose(tp34_lbp_hist[0:18,np.newaxis])
tp35_lbp_hist=np.transpose(tp35_lbp_hist[0:18,np.newaxis])
tp36_lbp_hist=np.transpose(tp36_lbp_hist[0:18,np.newaxis])
tp37_lbp_hist=np.transpose(tp37_lbp_hist[0:18,np.newaxis])
tp38_lbp_hist=np.transpose(tp38_lbp_hist[0:18,np.newaxis])
tp39_lbp_hist=np.transpose(tp39_lbp_hist[0:18,np.newaxis])
tp40_lbp_hist=np.transpose(tp40_lbp_hist[0:18,np.newaxis])
tp41_lbp_hist=np.transpose(tp41_lbp_hist[0:18,np.newaxis])
tp42_lbp_hist=np.transpose(tp42_lbp_hist[0:18,np.newaxis])
tp43_lbp_hist=np.transpose(tp43_lbp_hist[0:18,np.newaxis])
tp44_lbp_hist=np.transpose(tp44_lbp_hist[0:18,np.newaxis])
tp45_lbp_hist=np.transpose(tp45_lbp_hist[0:18,np.newaxis])
tp46_lbp_hist=np.transpose(tp46_lbp_hist[0:18,np.newaxis])
tp47_lbp_hist=np.transpose(tp47_lbp_hist[0:18,np.newaxis])
tp48_lbp_hist=np.transpose(tp48_lbp_hist[0:18,np.newaxis])
tp49_lbp_hist=np.transpose(tp49_lbp_hist[0:18,np.newaxis])
tp50_lbp_hist=np.transpose(tp50_lbp_hist[0:18,np.newaxis])
tp51_lbp_hist=np.transpose(tp51_lbp_hist[0:18,np.newaxis])
tp52_lbp_hist=np.transpose(tp52_lbp_hist[0:18,np.newaxis])
tp53_lbp_hist=np.transpose(tp53_lbp_hist[0:18,np.newaxis])
tp54_lbp_hist=np.transpose(tp54_lbp_hist[0:18,np.newaxis])
tp55_lbp_hist=np.transpose(tp55_lbp_hist[0:18,np.newaxis])
tp56_lbp_hist=np.transpose(tp56_lbp_hist[0:18,np.newaxis])
tp57_lbp_hist=np.transpose(tp57_lbp_hist[0:18,np.newaxis])
tp58_lbp_hist=np.transpose(tp58_lbp_hist[0:18,np.newaxis])
tp59_lbp_hist=np.transpose(tp59_lbp_hist[0:18,np.newaxis])
tp60_lbp_hist=np.transpose(tp60_lbp_hist[0:18,np.newaxis])
tp61_lbp_hist=np.transpose(tp61_lbp_hist[0:18,np.newaxis])
tp62_lbp_hist=np.transpose(tp62_lbp_hist[0:18,np.newaxis])
tp63_lbp_hist=np.transpose(tp63_lbp_hist[0:18,np.newaxis])
tp64_lbp_hist=np.transpose(tp64_lbp_hist[0:18,np.newaxis])
tp65_lbp_hist=np.transpose(tp65_lbp_hist[0:18,np.newaxis])
tp66_lbp_hist=np.transpose(tp66_lbp_hist[0:18,np.newaxis])
tp67_lbp_hist=np.transpose(tp67_lbp_hist[0:18,np.newaxis])
tp68_lbp_hist=np.transpose(tp68_lbp_hist[0:18,np.newaxis])
tp69_lbp_hist=np.transpose(tp69_lbp_hist[0:18,np.newaxis])
tp70_lbp_hist=np.transpose(tp70_lbp_hist[0:18,np.newaxis])
tp71_lbp_hist=np.transpose(tp71_lbp_hist[0:18,np.newaxis])
tp72_lbp_hist=np.transpose(tp72_lbp_hist[0:18,np.newaxis])
tp73_lbp_hist=np.transpose(tp73_lbp_hist[0:18,np.newaxis])
tp74_lbp_hist=np.transpose(tp74_lbp_hist[0:18,np.newaxis])
tp75_lbp_hist=np.transpose(tp75_lbp_hist[0:18,np.newaxis])
tp76_lbp_hist=np.transpose(tp76_lbp_hist[0:18,np.newaxis])
tp77_lbp_hist=np.transpose(tp77_lbp_hist[0:18,np.newaxis])
tp78_lbp_hist=np.transpose(tp78_lbp_hist[0:18,np.newaxis])
tp79_lbp_hist=np.transpose(tp79_lbp_hist[0:18,np.newaxis])
tp80_lbp_hist=np.transpose(tp80_lbp_hist[0:18,np.newaxis])
tp81_lbp_hist=np.transpose(tp81_lbp_hist[0:18,np.newaxis])
tp82_lbp_hist=np.transpose(tp82_lbp_hist[0:18,np.newaxis])
tp83_lbp_hist=np.transpose(tp83_lbp_hist[0:18,np.newaxis])
tp84_lbp_hist=np.transpose(tp84_lbp_hist[0:18,np.newaxis])
tp85_lbp_hist=np.transpose(tp85_lbp_hist[0:18,np.newaxis])
tp86_lbp_hist=np.transpose(tp86_lbp_hist[0:18,np.newaxis])
tp87_lbp_hist=np.transpose(tp87_lbp_hist[0:18,np.newaxis])
tp88_lbp_hist=np.transpose(tp88_lbp_hist[0:18,np.newaxis])
tp89_lbp_hist=np.transpose(tp89_lbp_hist[0:18,np.newaxis])
tp90_lbp_hist=np.transpose(tp90_lbp_hist[0:18,np.newaxis])
tp91_lbp_hist=np.transpose(tp91_lbp_hist[0:18,np.newaxis])
tp92_lbp_hist=np.transpose(tp92_lbp_hist[0:18,np.newaxis])
tp93_lbp_hist=np.transpose(tp93_lbp_hist[0:18,np.newaxis])
tp94_lbp_hist=np.transpose(tp94_lbp_hist[0:18,np.newaxis])
tp95_lbp_hist=np.transpose(tp95_lbp_hist[0:18,np.newaxis])
tp96_lbp_hist=np.transpose(tp96_lbp_hist[0:18,np.newaxis])
tp97_lbp_hist=np.transpose(tp97_lbp_hist[0:18,np.newaxis])
tp98_lbp_hist=np.transpose(tp98_lbp_hist[0:18,np.newaxis])
tp99_lbp_hist=np.transpose(tp99_lbp_hist[0:18,np.newaxis])
tp100_lbp_hist=np.transpose(tp100_lbp_hist[0:18,np.newaxis])
tp101_lbp_hist=np.transpose(tp101_lbp_hist[0:18,np.newaxis])
tp102_lbp_hist=np.transpose(tp102_lbp_hist[0:18,np.newaxis])
tp103_lbp_hist=np.transpose(tp103_lbp_hist[0:18,np.newaxis])
tp104_lbp_hist=np.transpose(tp104_lbp_hist[0:18,np.newaxis])
tp105_lbp_hist=np.transpose(tp105_lbp_hist[0:18,np.newaxis])
tp106_lbp_hist=np.transpose(tp106_lbp_hist[0:18,np.newaxis])
tp107_lbp_hist=np.transpose(tp107_lbp_hist[0:18,np.newaxis])
tp108_lbp_hist=np.transpose(tp108_lbp_hist[0:18,np.newaxis])
tp109_lbp_hist=np.transpose(tp109_lbp_hist[0:18,np.newaxis])
tp110_lbp_hist=np.transpose(tp110_lbp_hist[0:18,np.newaxis])
tp111_lbp_hist=np.transpose(tp111_lbp_hist[0:18,np.newaxis])
tp112_lbp_hist=np.transpose(tp112_lbp_hist[0:18,np.newaxis])
tp113_lbp_hist=np.transpose(tp113_lbp_hist[0:18,np.newaxis])
tp114_lbp_hist=np.transpose(tp114_lbp_hist[0:18,np.newaxis])
tp115_lbp_hist=np.transpose(tp115_lbp_hist[0:18,np.newaxis])
tp116_lbp_hist=np.transpose(tp116_lbp_hist[0:18,np.newaxis])
tp117_lbp_hist=np.transpose(tp117_lbp_hist[0:18,np.newaxis])
tp118_lbp_hist=np.transpose(tp118_lbp_hist[0:18,np.newaxis])
tp119_lbp_hist=np.transpose(tp119_lbp_hist[0:18,np.newaxis])
tp120_lbp_hist=np.transpose(tp120_lbp_hist[0:18,np.newaxis])
tp121_lbp_hist=np.transpose(tp121_lbp_hist[0:18,np.newaxis])
tp122_lbp_hist=np.transpose(tp122_lbp_hist[0:18,np.newaxis])
tp123_lbp_hist=np.transpose(tp123_lbp_hist[0:18,np.newaxis])
tp124_lbp_hist=np.transpose(tp124_lbp_hist[0:18,np.newaxis])
tp125_lbp_hist=np.transpose(tp125_lbp_hist[0:18,np.newaxis])
tp126_lbp_hist=np.transpose(tp126_lbp_hist[0:18,np.newaxis])
tp127_lbp_hist=np.transpose(tp127_lbp_hist[0:18,np.newaxis])
tp128_lbp_hist=np.transpose(tp128_lbp_hist[0:18,np.newaxis])
tp129_lbp_hist=np.transpose(tp129_lbp_hist[0:18,np.newaxis])
tp130_lbp_hist=np.transpose(tp130_lbp_hist[0:18,np.newaxis])



#gabungkan data citra menjadi satu matriks data training
trainData = np.concatenate((tc1_lbp_hist,
                            tc2_lbp_hist,
                            tc3_lbp_hist,
                            tc4_lbp_hist,
                            tc5_lbp_hist,
                            tc6_lbp_hist,
                            tc7_lbp_hist,
                            tc8_lbp_hist,
                            tc9_lbp_hist,
                            tc10_lbp_hist,
                            tc11_lbp_hist,
                            tc12_lbp_hist,
                            tc13_lbp_hist,
                            tc14_lbp_hist,
                            tc15_lbp_hist,
                            tc16_lbp_hist,
                            tc17_lbp_hist,
                            tc18_lbp_hist,
                            tc19_lbp_hist,
                            tc20_lbp_hist,
                            tc21_lbp_hist,
                            tc22_lbp_hist,
                            tc23_lbp_hist,
                            tc24_lbp_hist,
                            tc25_lbp_hist,
                            tc26_lbp_hist,
                            tc27_lbp_hist,
                            tc28_lbp_hist,
                            tc29_lbp_hist,
                            tc30_lbp_hist,
                            tc31_lbp_hist,
                            tc32_lbp_hist,
                            tc33_lbp_hist,
                            tc34_lbp_hist,
                            tc35_lbp_hist,
                            tc36_lbp_hist,
                            tc37_lbp_hist,
                            tc38_lbp_hist,
                            tc39_lbp_hist,
                            tc40_lbp_hist,
                            tc41_lbp_hist,
                            tc42_lbp_hist,
                            tc43_lbp_hist,
                            tc44_lbp_hist,
                            tc45_lbp_hist,
                            tc46_lbp_hist,
                            tc47_lbp_hist,
                            tc48_lbp_hist,
                            tc49_lbp_hist,
                            tc50_lbp_hist,
                            tc51_lbp_hist,
                            tc52_lbp_hist,
                            tc53_lbp_hist,
                            tc54_lbp_hist,
                            tc55_lbp_hist,
                            tc56_lbp_hist,
                            tc57_lbp_hist,
                            tc58_lbp_hist,
                            tc59_lbp_hist,
                            tc60_lbp_hist,
                            tc61_lbp_hist,
                            tc62_lbp_hist,
                            tc63_lbp_hist,
                            tc64_lbp_hist,
                            tc65_lbp_hist,
                            tc66_lbp_hist,
                            tc67_lbp_hist,
                            tc68_lbp_hist,
                            tc69_lbp_hist,
                            tc70_lbp_hist,
                            tc71_lbp_hist,
                            tc72_lbp_hist,
                            tc73_lbp_hist,
                            tc74_lbp_hist,
                            tc75_lbp_hist,
                            tc76_lbp_hist,
                            tc77_lbp_hist,
                            tc78_lbp_hist,
                            tc79_lbp_hist,
                            tc80_lbp_hist,
                            tc81_lbp_hist,
                            tc82_lbp_hist,
                            tc83_lbp_hist,
                            tc84_lbp_hist,
                            tc85_lbp_hist,
                            tc86_lbp_hist,
                            tc87_lbp_hist,
                            tc88_lbp_hist,
                            tc89_lbp_hist,
                            tc90_lbp_hist,
                            tc91_lbp_hist,
                            tc92_lbp_hist,
                            tc93_lbp_hist,
                            tc94_lbp_hist,
                            tc95_lbp_hist,
                            tc96_lbp_hist,
                            tc97_lbp_hist,
                            tc98_lbp_hist,
                            tc99_lbp_hist,
                            tc100_lbp_hist,
                            tc101_lbp_hist,
                            tc102_lbp_hist,
                            tc103_lbp_hist,
                            tc104_lbp_hist,
                            tc105_lbp_hist,
                            tc106_lbp_hist,
                            tc107_lbp_hist,
                            tc108_lbp_hist,
                            tc109_lbp_hist,
                            tc110_lbp_hist,
                            tc111_lbp_hist,
                            tc112_lbp_hist,
                            tc113_lbp_hist,
                            tc114_lbp_hist,
                            tc115_lbp_hist,
                            tc116_lbp_hist,
                            tc117_lbp_hist,
                            tc118_lbp_hist,
                            tc119_lbp_hist,
                            tc120_lbp_hist,
                            tc121_lbp_hist,
                            tc122_lbp_hist,
                            tc123_lbp_hist,
                            tc124_lbp_hist,
                            tc125_lbp_hist,
                            tc126_lbp_hist,
                            tc127_lbp_hist,
                            tc128_lbp_hist,
                            tc129_lbp_hist,
                            tc130_lbp_hist,
                            tp1_lbp_hist,
                            tp2_lbp_hist,
                            tp3_lbp_hist,
                            tp4_lbp_hist,
                            tp5_lbp_hist,
                            tp6_lbp_hist,
                            tp7_lbp_hist,
                            tp8_lbp_hist,
                            tp9_lbp_hist,
                            tp10_lbp_hist,
                            tp11_lbp_hist,
                            tp12_lbp_hist,
                            tp13_lbp_hist,
                            tp14_lbp_hist,
                            tp15_lbp_hist,
                            tp16_lbp_hist,
                            tp17_lbp_hist,
                            tp18_lbp_hist,
                            tp19_lbp_hist,
                            tp20_lbp_hist,
                            tp21_lbp_hist,
                            tp22_lbp_hist,
                            tp23_lbp_hist,
                            tp24_lbp_hist,
                            tp25_lbp_hist,
                            tp26_lbp_hist,
                            tp27_lbp_hist,
                            tp28_lbp_hist,
                            tp29_lbp_hist,
                            tp30_lbp_hist,
                            tp31_lbp_hist,
                            tp32_lbp_hist,
                            tp33_lbp_hist,
                            tp34_lbp_hist,
                            tp35_lbp_hist,
                            tp36_lbp_hist,
                            tp37_lbp_hist,
                            tp38_lbp_hist,
                            tp39_lbp_hist,
                            tp40_lbp_hist,
                            tp41_lbp_hist,
                            tp42_lbp_hist,
                            tp43_lbp_hist,
                            tp44_lbp_hist,
                            tp45_lbp_hist,
                            tp46_lbp_hist,
                            tp47_lbp_hist,
                            tp48_lbp_hist,
                            tp49_lbp_hist,
                            tp50_lbp_hist,
                            tp51_lbp_hist,
                            tp52_lbp_hist,
                            tp53_lbp_hist,
                            tp54_lbp_hist,
                            tp55_lbp_hist,
                            tp56_lbp_hist,
                            tp57_lbp_hist,
                            tp58_lbp_hist,
                            tp59_lbp_hist,
                            tp60_lbp_hist,
                            tp61_lbp_hist,
                            tp62_lbp_hist,
                            tp63_lbp_hist,
                            tp64_lbp_hist,
                            tp65_lbp_hist,
                            tp66_lbp_hist,
                            tp67_lbp_hist,
                            tp68_lbp_hist,
                            tp69_lbp_hist,
                            tp70_lbp_hist,
                            tp71_lbp_hist,
                            tp72_lbp_hist,
                            tp73_lbp_hist,
                            tp74_lbp_hist,
                            tp75_lbp_hist,
                            tp76_lbp_hist,
                            tp77_lbp_hist,
                            tp78_lbp_hist,
                            tp79_lbp_hist,
                            tp80_lbp_hist,
                            tp81_lbp_hist,
                            tp82_lbp_hist,
                            tp83_lbp_hist,
                            tp84_lbp_hist,
                            tp85_lbp_hist,
                            tp86_lbp_hist,
                            tp87_lbp_hist,
                            tp88_lbp_hist,
                            tp89_lbp_hist,
                            tp90_lbp_hist,
                            tp91_lbp_hist,
                            tp92_lbp_hist,
                            tp93_lbp_hist,
                            tp94_lbp_hist,
                            tp95_lbp_hist,
                            tp96_lbp_hist,
                            tp97_lbp_hist,
                            tp98_lbp_hist,
                            tp99_lbp_hist,
                            tp100_lbp_hist,
                            tp101_lbp_hist,
                            tp102_lbp_hist,
                            tp103_lbp_hist,
                            tp104_lbp_hist,
                            tp105_lbp_hist,
                            tp106_lbp_hist,
                            tp107_lbp_hist,
                            tp108_lbp_hist,
                            tp109_lbp_hist,
                            tp110_lbp_hist,
                            tp111_lbp_hist,
                            tp112_lbp_hist,
                            tp113_lbp_hist,
                            tp114_lbp_hist,
                            tp115_lbp_hist,
                            tp116_lbp_hist,
                            tp117_lbp_hist,
                            tp118_lbp_hist,
                            tp119_lbp_hist,
                            tp120_lbp_hist,
                            tp121_lbp_hist,
                            tp122_lbp_hist,
                            tp123_lbp_hist,
                            tp124_lbp_hist,
                            tp125_lbp_hist,
                            tp126_lbp_hist,
                            tp127_lbp_hist,
                            tp128_lbp_hist,
                            tp129_lbp_hist,
                            tp130_lbp_hist
                            ), axis=0).astype(np.float32)


#Kelasnya
kelas = ["Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",	
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok","Batik Ceplok",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang",
	"Batik Parang","Batik Parang","Batik Parang","Batik Parang","Batik Parang"]

from sklearn.model_selection import cross_val_score
#Buat model KNN
for i in range (2,25):
    knn=KNeighborsClassifier(n_neighbors=i)
    akurasiknn = cross_val_score(knn, trainData, kelas, cv=5, scoring='accuracy')
    print("Nilai akurasi dengan K-Nearest Neighbor K =",i," : ", akurasiknn.mean())

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(trainData,kelas)

#Buat model NB
nb=nb = MultinomialNB()
akurasinb = cross_val_score(nb, trainData, kelas, cv=5, scoring='accuracy')
print("Nilai akurasi dengan Naive Bayes : ", akurasinb.mean())
nb.fit(trainData,kelas)






#Identifikasi motif batik. Apakah termasuk motif parang atau motif ceplok ?
#PILIH GAMBAR YANG INGIN DIIDENTIFIKASI
gambar = cv.imread('Dataset_projek/Testing/Testing parang/Yogya-Parang-Parang-Sogan (15).jpg', cv.IMREAD_GRAYSCALE)
x = getdataTest(gambar)
prediksi_dgn_knn = knn.predict(x)
prediksi_dgn_nb = nb.predict(x)
print()
print("Gambar ini diidentifikasi dengan KNN K=18 hasilnya : ", prediksi_dgn_knn)
print("Gambar ini diidentifikasi dengan Naive Bayes hasilnya : ", prediksi_dgn_knn)

cv.namedWindow('Gambar yang diidentifikasi', cv.WINDOW_NORMAL)
cv.imshow('Gambar yang diidentifikasi' , gambar)
cv.waitKey(0)
cv.destroyAllWindows()
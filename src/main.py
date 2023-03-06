# -*- coding: utf-8 -*-

# ===============================================================================
# 
#                               CV Lab 2
#
#       Author: Rzhevskiy S.S. ITMO University
# 
# ===============================================================================

import cv2
import numpy as np
import time


#
#   Основная логика работы
#

#Вариант без встроенной функции OpenCV - на базе SQDIFF хороший результат. но ОЧЕНЬ медленно
def template_matching_my(src, temp):

    pt = template_matching_my_calc(src, temp)

    #Рисуем прямоугольник
    cv2.rectangle(src, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 2)

    #Отображаем результат
    cv2.imshow("Image",src)
    cv2.waitKey(0)
    return

def template_matching_my_calc(src, temp):

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Получаем размеры исходного изображения
    h = gray.shape[0]
    w = gray.shape[1]

    # Получаем размеры темплата
    ht = temp.shape[0]
    wt = temp.shape[1]

    # Массив для хранения показателя метрики
    score = np.empty((h - ht, w - wt))

    # Слайдим по картинке
    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            # Алгоритм SQDIFF
            #https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dab65c042ed62c9e9e095a1e7e41fe2773
            diff = np.power((gray[dy:dy + ht, dx:dx + wt] - temp), 2)
            score[dy, dx] = diff.sum()

    pt = np.unravel_index(score.argmin(), score.shape)

    return(pt[1], pt[0])



#Вариант реализации средствами OpenCV
def template_matching_ocv(src, temp):

    # Получаем параметры картинок
    h, w = temp.shape

    res = cv2.matchTemplate(gray,temp,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(src,top_left, bottom_right, (0, 0, 200), 2)

    #Отображаем результат
    cv2.imshow("Image",src)
    cv2.waitKey(0)
    return

#Вариант реализации feature-matching ORB
def feature_matching_orb(src, temp):
    
    #Переводим в ЧБ
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Создаем экземпляр ORB-детектора на 1000 кейпоинтов
    orb = cv2.ORB_create(1000)

    # Определяем кейпоинты на оригинальном изображении
    (kp1, des1) = orb.detectAndCompute(gray, None)

    # Определяем кейпоинты на темплате
    (kp2, des2) = orb.detectAndCompute(temp, None)

    # Создаем BFMatche
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Сравниваем
    matches = bf.match(des1,des2)
    # Сортируем по расстоянию(векторов)
    matches = sorted(matches, key = lambda x:x.distance)

    # Рисуем первые 10 матчей
    #img3 = cv2.drawMatches(img,kp1,temp,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print("Matches: {0}".format(len(matches)))

    #Получаем координаты совпадений https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python

    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
    #list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

    pt = list_kp1[0]

    x = int(pt[0] - w / 2)
    y = int(pt[1] - h / 2)

    cv2.rectangle(src,(x, y), (x + w, y + h), (0, 0, 200), 2)

    cv2.imshow("Image",src)
    cv2.waitKey(0)
    return


if __name__ == "__main__":
    # Загружаем картинки
    img = cv2.imread('example.jpg')
    temp = cv2.imread('temp_original.jpg')

    # Переводим в чб
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # Получаем параметры картинок
    h, w = temp.shape

    # Рассчитываем координаты объекта(с контролем времени)
    # start_time = time.time()

    template_matching_ocv(img, temp)
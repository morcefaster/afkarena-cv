import os
import numpy as np
import cv2
from enum import Enum
from plot_helper import plot_helper as plt

DISTANCE_THRESHOLD_HERO = 0.75
DISTANCE_THRESHOLD_FURNITURE = 0.95
MIN_GOOD_THRESHOLD_HERO = 15
MIN_GOOD_THRESHOLD_FURNITURE = 5
IMAGES_DIRECTORY_HERO = '_images_hero'
IMAGES_DIRECTORY_FURNITURE = '_images_furniture_si'
SHOW_PLOTS = True

class Ascension(Enum):
    Ascended=2
    Epic=1
    Not_Specified=0

    @staticmethod
    def from_str(label):
        if label == 'Ascended':
            return Ascension.Ascended
        elif label == 'Epic':
            return Ascension.Epic
        else:
            return Ascension.Not_Specified
            #raise NotImplementedError

class SignatureItem(Enum):
    Zero=0
    One=1
    Ten=10
    Twenty=20
    Thirty=30

    @staticmethod
    def from_str(label):
        if label == '0':
            return SignatureItem.Zero
        else:
            return Ascension.Not_Specified
            #raise NotImplementedError

class Hero:
    def __init__(
        self,
        score: int = 0,
        name: str = "",
        ascension: Ascension = Ascension.Not_Specified,
        signature_item: SignatureItem = SignatureItem.Zero,
        furniture = 0):
        self.score = score
        self.name = name
        self.ascension = ascension
        self.signature_item = signature_item
        self.furniture = furniture
    
    @staticmethod
    def find_heroes(
            image_filename):
        heroes = {}
        for filename in os.listdir(IMAGES_DIRECTORY_HERO):
            img1 = cv2.imread(os.path.join(IMAGES_DIRECTORY_HERO, filename)) # queryImage
            img2 = cv2.imread(image_filename) # trainImage

            # Initiate SIFT detector
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < DISTANCE_THRESHOLD_HERO*n.distance:
                    #print("m.distance=%s, n.distance=%s" % (m.distance, DISTANCE_THRESHOLD_HERO*n.distance))
                    good.append([m])
            print("Hero: good=%s" % str(len(good)))

            if len(good)>=MIN_GOOD_THRESHOLD_HERO:
                # cv2.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                # Match descriptors.
                matches = bf.match(des1, des2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)
                good_matches = matches[:10]
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = img1.shape[:2]
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)

                img2_rect = cv2.boundingRect(dst) # get img2 rect
                dst += (w, 0)  # adding offset
                img3 = cv2.polylines(img3, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                hero_portrait = img2[img2_rect[1]:img2_rect[1]+img2_rect[3], img2_rect[0]:img2_rect[0]+img2_rect[2]]
                #cv2.imwrite('roi.png', img2[img2_rect[1]:img2_rect[1]+img2_rect[3], img2_rect[0]:img2_rect[0]+img2_rect[2]])

                if SHOW_PLOTS:
                    plt.show_bgr(img3)
                    #plt.imshow(sub_img), plt.show()

                furniture = Hero.find_furniture(hero_portrait)

                delims = os.path.splitext(filename)[0].split('-')
                hero_key = delims[0]
                ascension = Ascension.from_str(delims[1])
                #signature_item = delims[2]
                signature_item = SignatureItem.Zero

                if hero_key in heroes:
                    if len(good) > heroes[hero_key].score or ascension.value > heroes[hero_key].ascension.value:
                        heroes[hero_key] = Hero(len(good), hero_key, ascension, int(signature_item), int(furniture))
                else:
                    heroes[hero_key] = Hero(len(good), hero_key, ascension, signature_item, int(furniture))

        return heroes

    @staticmethod
    def find_furniture(
            hero_portrait) -> int:
        if SHOW_PLOTS:
            plt.show_bgr(hero_portrait)
        ret = 0
        for filename in sorted(os.listdir(IMAGES_DIRECTORY_FURNITURE)):
            img1 = cv2.imread(os.path.join(IMAGES_DIRECTORY_FURNITURE, filename)) # queryImage
            img2 = hero_portrait # trainImage

            # Initiate SIFT detector
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.show_bgr(img3)
                if m.distance < DISTANCE_THRESHOLD_FURNITURE*n.distance:
                    print("m.distance=%s, n.distance=%s" % (m.distance, DISTANCE_THRESHOLD_FURNITURE*n.distance))
                    good.append([m])                    

            print("Furniture: good=%s" % str(len(good)))

            top_score = 0
            if len(good)>=MIN_GOOD_THRESHOLD_FURNITURE:
                # cv2.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.show_bgr(img3)

                # Match descriptors.
                matches = bf.match(des1, des2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)
                good_matches = matches[:10]
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = img1.shape[:2]
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)

                img2_rect = cv2.boundingRect(dst) # get img2 rect
                dst += (w, 0)  # adding offset
                img3 = cv2.polylines(img3, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                #sub_img = img2[img2_rect[1]:img2_rect[1]+img2_rect[3], img2_rect[0]:img2_rect[0]+img2_rect[2]]
                #cv2.imwrite('roi.png', img2[img2_rect[1]:img2_rect[1]+img2_rect[3], img2_rect[0]:img2_rect[0]+img2_rect[2]])

                if SHOW_PLOTS:
                    plt.show_bgr(img3)
                    #plt.imshow(sub_img), plt.show()

                if len(good) > top_score:
                    ret = os.path.splitext(filename)[0].split('-')
        return ret

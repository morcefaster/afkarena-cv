import img_helper
import plot_helper
import cv2 as cv
import os

PORTRAIT_BOUNDS = [[-30,-20],[10,20]]
#PORTRAIT_BOUNDS = [[0,0],[0,0]]
ZERO_BOUNDS = [[0,0],[0,0]]
FURNITURE_BOUNDS = [[0, 20], [30, 40]]
FURNITURE_DIRECTORY = "_images_furniture"

def main():
    (_,_,_,_,r,match)= img_helper.img_helper.find_scaled_path("_images_hero/Tasi-Skin.jpg","resources/joko.png", PORTRAIT_BOUNDS, show_progress=False)
    print(r)
    plot_helper.plot_helper.show_bgr(match)
    # furniture = match[FURNITURE_BOUNDS[0][1]:FURNITURE_BOUNDS[1][1], FURNITURE_BOUNDS[0][0]:FURNITURE_BOUNDS[1][0]]
    # plot_helper.plot_helper.show_bgr(furniture)
    #print(img_helper.img_helper.find_best_match(FURNITURE_DIRECTORY, furniture))   
    for f_p in sorted(os.listdir(FURNITURE_DIRECTORY)):
        print(f_p)
        furn = cv.imread(os.path.join(FURNITURE_DIRECTORY, f_p), cv.IMREAD_COLOR)
        img_helper.img_helper.match_with_mask(furn, match)


if __name__ == "__main__":
    main()
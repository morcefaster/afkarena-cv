import matplotlib.pyplot as plt
import cv2

class plot_helper:
    @staticmethod
    def show_bgr(img):
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_img)
        plt.show()
import cv2
import numpy as np

_CLASS_COLOR = {
    "1": "red",
    "2": "yellow",
    "3": "blue",
    "4": "green",
    "5": "black",
    "6": "light_blue",
}



#大津法+二值化
def ostu_threshold(RAW_image):
    Gray_image = cv2.cvtColor(RAW_image, cv2.COLOR_BGR2GRAY)
    thre, Ostu_image = cv2.threshold(Gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("thre_image", Ostu_image)
    return Ostu_image

#物块颜色处理
# def block_preprocessing(gongxun)

if __name__ == "__main__":
    pass
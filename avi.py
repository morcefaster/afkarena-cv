import os
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont

screenshot_directory = 'C:/Users/Dirtyore/Nox_share/ImageShare/Screenshots/'
output = 'C:/Users/Dirtyore/Desktop/output.avi'

images = [img for img in os.listdir(screenshot_directory) if img.endswith(".png")]
frame = cv2.imread(os.path.join(screenshot_directory, images[0]))

height, width, layers = frame.shape
left = 1369
top = 3185
right = width-260
bottom = height-930
crop = frame[top:bottom, left:right, :]

font = ImageFont.truetype("arial.ttf", 36)
org = (10, 10)
font_scale = 1
color = (255, 255, 255)
thickness = 2

img_pil = Image.fromarray(crop)
draw = ImageDraw.Draw(img_pil)
draw.text(org,  "some text .... blah", font = font, fill = color)
crop_with_text = numpy.array(img_pil)
cv2.imshow('Image', crop_with_text)
cv2.waitKey(0)

video = cv2.VideoWriter(output, 0, 1, (right-left, bottom-top))

frame_count = 0
skip_frames = 2
for screenshot_filename in images:
    frame_count += 1
    if frame_count % skip_frames == 0:
        continue
    image = cv2.imread(os.path.join(screenshot_directory, screenshot_filename))
    crop = image[top:bottom, left:right, :]

    filename_tokens = screenshot_filename.split("-")
    txt_label = "NNZ - %s/%s/%s %s:%s:%s" % \
        (
            filename_tokens[1],
            filename_tokens[2],
            filename_tokens[0].split("_")[1],
            filename_tokens[3],
            filename_tokens[4],
            filename_tokens[5].split(".")[0]
        )

    img_pil = Image.fromarray(crop)
    draw = ImageDraw.Draw(img_pil)
    draw.text(org,  txt_label, font = font, fill = color)
    crop_with_text = numpy.array(img_pil)
    video.write(crop_with_text)

cv2.destroyAllWindows()
video.release()
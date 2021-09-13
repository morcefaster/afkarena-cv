import glob
from PIL import Image, ImageDraw, ImageFont

screenshot_directory = "C:/Users/Dirtyore/Nox_share/ImageShare/Screenshots/*.png"
output = "C:/Users/Dirtyore/Desktop/output.webp"

imgs = []

for screenshot_filename in sorted(glob.glob(screenshot_directory)):
    img = Image.open(screenshot_filename)
    width, height = img.size
    # left = 520
    # top = 1190
    # right = width-110
    # bottom = height-340
    left = 1369
    top = 3185
    right = width-260
    bottom = height-930
    img = img.crop((left, top, right, bottom))
    imgs.append(img)

    filename_tokens = screenshot_filename.split("-")
    txt_label = "NNZ - %s/%s/%s %s:%s:%s" % \
        (
            filename_tokens[1],
            filename_tokens[2],
            filename_tokens[0].split("_")[2],
            filename_tokens[3],
            filename_tokens[4],
            filename_tokens[5].split(".")[0]
        )

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        "arial.ttf",
        36)
    draw.text(
        (10, 10),
        txt_label,
        fill=(255, 255, 255),
        font=font)
    print(txt_label)

img.save(
    fp=output,
    format='WEBP',
    append_images=imgs,
    save_all=True,
    duration=400,
    loop=0,
    quality=85,
    subsampling=0)

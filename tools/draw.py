from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# define decorator
def init_parameters(fun, **init_dict):
    """
    help you to set the parameters in one's habits
    """

    def job(*args, **option):
        option.update(init_dict)
        return fun(*args, **option)

    return job


def cv2_img_add_text(
    img,
    text,
    left_corner: Tuple[int, int],
    text_rgb_color=(255, 0, 0),
    text_size=24,
    font="mingliu.ttc",
    **option
):
    """
    USAGE:
        cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
    """
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(
        font=font, size=text_size, encoding=option.get("encoding", "utf-8")
    )
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get("replace"):
        img[:] = cv2_img[:]
        return None
    return cv2_img


def main():
    np_img = np.ones(IMG_SHAPE, dtype=np.uint8) * 255  # background with white color

    np_img = cv2_img_add_text(
        np_img, "Hello\nWorld", (0, 0), text_rgb_color=(255, 0, 0), text_size=TEXT_SIZE
    )
    np_img = cv2_img_add_text(
        np_img,
        "中文",
        (0, LINE_HEIGHT * 2),
        text_rgb_color=(0, 255, 0),
        text_size=TEXT_SIZE,
    )

    cur_y = LINE_HEIGHT * 3
    draw_text = init_parameters(
        cv2_img_add_text,
        text_size=TEXT_SIZE,
        text_rgb_color=(0, 128, 255),
        font="kaiu.ttf",
        replace=True,
    )
    for msg in ("笑傲江湖", "滄海一聲笑"):
        draw_text(np_img, msg, (0, cur_y))
        cur_y += LINE_HEIGHT + 1
    draw_text(
        np_img,
        """123
            456
            789
            """,
                    (0, cur_y),
                )
    cv2.imshow("demo", np_img), cv2.waitKey(0)


if __name__ == "__main__":
    IMG_HEIGHT, IMG_WIDTH, CHANNEL = IMG_SHAPE = (250, 160, 3)
    TEXT_SIZE = LINE_HEIGHT = 32
    main()

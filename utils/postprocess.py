import cv2 as cv
import numpy as np
import random
import colorsys
import hashlib
from shapely.geometry import Polygon

#======== upsize the mask to orginal size ====================================
def up_mask(mask, H, W):
    asp_ratio = W / H
    size = mask.shape[0]

    if asp_ratio > 1:  # horizontal image
        cr_w = size
        cr_h = int(size / asp_ratio)

        offset = int(cr_w - cr_h)
        if offset % 2 != 0:  # odd offset
            top = int(offset // 2 + 1)
        else:
            top = int(offset // 2)

        cr_mask = mask[top:top + cr_h, :]

    elif asp_ratio < 1:  # vertical image
        cr_w = int(size / asp_ratio)
        cr_h = size

        offset = int(cr_h - cr_w)
        if offset % 2 != 0:  # odd offset
            left = int(offset // 2 + 1)
        else:
            left = int(offset // 2)

        cr_mask = mask[:, left:left + cr_w]

    dim = (W, H)

    up_mask = cv.resize(cr_mask, dim, interpolation=cv.INTER_NEAREST)

    return up_mask


#========= combine individual masks into one ================================
def create_mask(masks_path, H, W):
    n = len(masks_path)
    full_mask = np.zeros(shape=(H, W, n))
    for i in range(len(masks_path)):
        mask = cv.imread(masks_path[i], 0)  # grayscale
        mask[mask == 255] = 1
        full_mask[:, :, i] = mask

    return full_mask


#==========================================================================
def crop_pts(mask):
    """
    return coordinate points from predicted mask
    """
    # mask = mask.numpy().astype(np.float32)
    # mask = np.squeeze(mask, axis=2)

    coord = np.where(mask == [1])
    y_min = min(coord[0]) - 60  # offsets for safety
    y_max = max(coord[0]) + 60
    x_max = max(coord[1]) - 600
    x_min = min(coord[1])
    if x_min > 60:
        x_min = min(coord[1]) - 60

    pts = [x_min, y_min, x_max, y_max]
    pts = np.asarray(pts).astype(int)

    return pts


#=========== Return the up sized (original size) mask =============================
def cr_up(mask, pts, H, W):
    top = pts[1]
    bottom = H - pts[3]
    left = pts[0]
    right = W - pts[2]

    pad_mask = cv.copyMakeBorder(mask, top, bottom, left, right, cv.BORDER_CONSTANT, None, value=0)

    return pad_mask


#========Finding the first two largest elements from a list of arrays=========
def finding_largest(arr_list):
    largest = [a for a in arr_list if len(a) == max([len(a) for a in arr_list])]
    largest = np.squeeze(np.array(largest))
    largest_idx = np.argmax([len(a) for a in arr_list])
    arr_list.pop(largest_idx)
    second_largest = [a for a in arr_list if len(a) == max([len(a) for a in arr_list])]
    second_largest = np.squeeze(np.array(second_largest))
    return [largest, second_largest]


#================= Extract the head bone structures from ventral view images as polygons ===================
str_names = ['p', ['op2', 'op1'], ['oc2', 'oc1'], 'n', ['m2', 'm1'], ['hm2', 'hm1'], ['en2', 'en1'], ['d2', 'd1'], ['cl2', 'cl1'], 
            ['ch2','ch1'], ['cb2', 'cb1'], ['br2b', 'br2a'], ['br1b', 'br1a']]
def v_extract_polygons(mask, n,H,W):
    poly_dict = {}  # to store polygons in a dictionary

    for j in range(n):
        pred_channel = mask[:, :, j].astype(np.uint8)
        ret, thresh = cv.threshold(pred_channel, 1, 255, cv.THRESH_OTSU)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours = [c for c in contours if len(c) > 25] if len(contours) >= 2 else list(contours)

        if contours:
            if (str_names[j] == 'p') or (str_names[j] == 'n'):
                if len(contours) > 1:
                    contours = [c for c in contours if len(c) == max([len(a) for a in contours])]
                poly_dict[str_names[j]] = Polygon(np.squeeze(contours[0]))
            else:
                if len(contours) == 1:
                    poly = Polygon(np.squeeze(contours[0]))
                    min_coords, max_coords = np.squeeze(contours[0])[:, 1].min(), np.squeeze(contours[0])[:, 1].max()
                    if min_coords <= H // 2 and max_coords <= H // 2:
                        poly_dict[str_names[j][1]] = poly
                        poly_dict[str_names[j][0]] = None
                    else:
                        poly_dict[str_names[j][0]] = poly
                        poly_dict[str_names[j][1]] = None
                elif len(contours) == 2:
                    for k in range(2):
                        poly_dict[str_names[j][k]] = Polygon(np.squeeze(contours[k]))
                else:
                    largest_contours = finding_largest(contours)
                    for k in range(len(largest_contours)):
                        pred_str = np.squeeze(contours[k])
                        poly = Polygon(pred_str)
                        min_coords = pred_str[:,1].min()   # to check if the present struc is upper or lower one
                        max_coords = pred_str[:,1].max()
                        if min_coords <= H // 2 and max_coords <= H // 2:
                            poly_dict[str_names[j][1]] = poly
                        else:
                            poly_dict[str_names[j][0]] = poly
        else:
            if isinstance(str_names[j], list):
                poly_dict[str_names[j][0]] = None
                poly_dict[str_names[j][1]] = None
            else:
                poly_dict[str_names[j]] = None

    return poly_dict


#======= Extract vertebral bone structures from lateral view as polygons ==========================
def l_extract_polygons(mask):
    blur = cv.blur(mask, (9, 9))
    ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU)
    pred_structures, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    pred_structures = list(pred_structures)
    pred_structures = [a for a in pred_structures if len(a) > 25]

    return pred_structures


#========= generate random colors for drawing contours on images ======================
def color_from_name(name, bright=True):
    """
    Generate a consistent BGR color for a given object name.
    """
    brightness = 1.0 if bright else 0.7

    # Hash name → stable hue
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0

    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, brightness)

    # Convert to OpenCV BGR uint8
    return (int(b * 255), int(g * 255), int(r * 255))



#================ Drawing countours (bone structures) on images ===========               
def draw_polygons_old(image, contours):  # contours (polygons) are dictionary objects
    colors = random_colors(24)
    contours_list = []
    for key, value in contours.items():
        contours_list.append(value)
    for i in range(len(contours_list)):
        if contours_list[i] is not None:
            poly = contours_list[i]
            coords = list(poly.exterior.coords)
            coords = np.array(coords).astype(np.int32)
            coords = coords.reshape((-1, 1, 2))
            cv.drawContours(image, coords, -1, colors[i], 3, cv.LINE_AA)
        else:
            continue
    return image

    import cv2 as cv
#======================================================================

def draw_polygons(image, contours, thickness=2, font_scale=0.6):
    """
    Draw and label shapely polygons on an image.

    contours: dict
        key   → object name (label)
        value → Polygon or list/tuple of Polygons
    """
    for name, polys in contours.items():

        if polys is None:
            continue

        # Normalize to list
        if not isinstance(polys, (list, tuple)):
            polys = [polys]

        color = color_from_name(name)

        for idx, poly in enumerate(polys):
            if poly is None or poly.is_empty:
                continue

            # Convert shapely polygon → OpenCV contour
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            coords = coords.reshape((-1, 1, 2))

            cv.drawContours(
                image,
                [coords],
                -1,
                color,
                thickness,
                cv.LINE_AA
            )

            # Label only once per object (on first polygon)
            if idx == 0:
                x, y = coords[0][0]
                cv.putText(
                    image,
                    name,
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    2,
                    cv.LINE_AA
                )

    return image

 
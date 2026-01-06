

import cv2 as cv


def rescale_pad(image, mask, desired_size):
    
    h, w = image.shape[:2]
    
    aspect = w / h
    
    if aspect > 1 : #horizontal image
        new_w = desired_size
        new_h = int(desired_size * h / w)
        offset = int(new_w - new_h)
        if offset %  2 != 0: #odd offset
            top = offset // 2 + 1
            bottom = offset // 2
        else:
            top = bottom = offset // 2
        
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_NEAREST)
        pad_img = cv.copyMakeBorder(re_img, top, bottom, 0, 0, cv.BORDER_REPLICATE)
        if mask is not None:
            re_mask = cv.resize(mask, dim, interpolation = cv.INTER_NEAREST)
            pad_mask = cv.copyMakeBorder(re_mask, top, bottom, 0, 0, cv.BORDER_REPLICATE)
            
        else:
            pad_mask = None

            
    elif aspect < 1:  #vertical image
        new_h = desired_size
        new_w = int(desired_size * w / h)
        offset = int(new_h - new_w)
        if offset %  2 != 0: #odd offset
            left = offset //2 + 1
            right = offset // 2
        else:
            left = right = offset // 2
        
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_NEAREST)
        pad_img = cv.copyMakeBorder(re_img, 0, 0, left, right, cv.BORDER_REPLICATE)
        if mask is not None:
            re_mask = cv.resize(mask, dim, interpolation = cv.INTER_NEAREST)
            pad_mask = cv.copyMakeBorder(re_mask, 0, 0, left, right, cv.BORDER_REPLICATE)
        else:
            pad_mask = None

    
    
    return pad_img, pad_mask
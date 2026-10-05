import numpy as np

def expand_bbox(obbox,imgsize,borders = 5, border_type = 'percentage') :
    ih = imgsize[0]
    iw = imgsize[1]

    x1,y1,x2,y2 = np.array(obbox)
    w = int((x2 - x1))
    h = int((y2 - y1))

    if border_type == 'percentage':
        yborder = int(round((y2 - y1) * borders / 100))
        xborder = int(round((x2 - x1) * borders / 100))
        x1,y1,x2,y2 = int(round(x1)),int(round(y1)),int(round(x1))+w,int(round(y1))+h
        nx1, nx2 = x1 - xborder, x2 + xborder
        ny1, ny2 = y1 - yborder, y2 + yborder
    else : #  border_type == 'center_fixed'
        yborder =(borders - h) // 2
        xborder =(borders - w) // 2

        nx1, nx2 = int(round(x1) - xborder), int(round(x2) + xborder)
        ny1, ny2 = int(round(y1) - yborder), int(round(y2) + yborder)

    nx1, nx2 = max(0,nx1), min(iw, nx2)
    ny1, ny2 = max(0,ny1) , min(ih, ny2)
    bbox = [nx1,ny1,nx2,ny2]
    dbbox = [abs(nx1-x1),abs(ny1-y1),abs(nx2-x2),abs(ny2-y2)]

    return bbox,dbbox



def get_corners(msk, label=True, borders=5, border_type='percentage') :
    white_pixels = np.array(np.where(msk == label))

    if white_pixels is None :
        return None,None,None,None
    y1 = min(white_pixels[0, :])
    y2 = max(white_pixels[0, :])
    x1 = min(white_pixels[1, :])
    x2 = max(white_pixels[1, :])

    bbox = expand_bbox([x1, y1, x2, y2], msk.shape, borders=borders, border_type=border_type)

    return bbox


def crop_bbox(img,bbox) :
    x1,y1,x2,y2 = np.array(bbox)
    h = int(y2-y1)
    w = int(x2 - x1)
    if img.ndim == 3 :
        imgc = img[int(y1):int(y1)+h,int(x1):int(x1)+w,:]
    elif img.ndim == 2:
        imgc = img[int(y1):int(y1) + h, int(x1):int(x1) + w]

    return imgc
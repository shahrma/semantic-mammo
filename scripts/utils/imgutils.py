import os

import numpy as np
import pydicom
import json

#from common.mammo_preprocessing import fullMammoPreprocess,maskPreprocess,sumMasks,norm_16b
#import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')

from common.bboxutils import get_corners, crop_bbox


def load_image(src_img, resize = 2):
    simg = pydicom.dcmread(src_img)
    simg = simg.pixel_array

    (h,w) = simg.shape
    simg = cv2.resize(simg, (round(w/resize), round(h/resize)))

    return  simg



def get_meta_by_idx(meta,id) :
    for item in meta :
        if item['abnormality id'] == id :
            return item
    return []


def get_data(srcdir,item,outsize = None) :
    mask_path  = [f.path.replace('\\', '/').split('/')[-1] for f in os.scandir(os.path.join(srcdir,item)) if f.is_dir()]

    with open(os.path.join(srcdir,item,'meta.json')) as f:
        meta = json.load(f)

    scan_img = os.path.join(srcdir,item,f'{item}.dcm')
    simg = pydicom.dcmread(scan_img)
    simg = simg.pixel_array
    if outsize is not None :
        simg = cv2.resize(simg, (outsize, outsize))  # cv2 converts dims order

    labels = []
    masks = []
    for mp_dir in mask_path:
        cmeta = get_meta_by_idx(meta, int(mp_dir))
        mp = os.path.join(srcdir,item,mp_dir,f'{item}_MASK.dcm')
        # Read mask(s) .dcm file(s).
        dimg = pydicom.dcmread(mp)
        mimg = dimg.pixel_array
        mimg = cv2.resize(mimg, (simg.shape[1],simg.shape[0])) #cv2 converts dims order
        masks.append(mimg)

        breast_density = cmeta['breast_density'] if 'breast_density' in cmeta.keys() else cmeta['breast density']
        labels.append({ 'pathology': cmeta['pathology'] ,
                        'type' :  cmeta['abnormality type'] ,
                        'abnormality_id' : cmeta['abnormality id'] ,
                        'patient_id' : cmeta['patient_id'],
                        'density' : breast_density,
                        'side' :  cmeta['left or right breast'],
                        'view' :  cmeta['image view']})

    return simg,masks,labels

import matplotlib.pyplot as plt
import numpy as np
import os

def compose_trans_mask(simg, mask,mode = 'green'):
    mask = mask.astype(float)/255.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    simg = clahe.apply((simg*255).astype(np.uint16))

    cimg = (simg - simg.min()) / (simg.max() - simg.min())
    cimg = np.stack([cimg] * 3, axis=-1)

    cimg0 = 255 * (cimg[:, :, 0])
    cimg1 = 255 * (cimg[:, :, 1])
    cimg2 = 255 * (cimg[:, :, 2])
    cimg2[mask>0] = (cimg2[mask>0]*0.25 + 255*0.75)
    #cimg2 = 255 * (cimg[:, :, 2] * (1 - mask)*0.05 + mask*0.95)

    outimg = np.concatenate(( cimg0[:, :, None],cimg1[:, :, None], cimg2[:, :, None]), axis=2)
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)
    return outimg

def compose_mask(simg, mask,mode = 'green'):
    mask = mask.astype(float)/255.0
    if mode == 'green':
        cmap = plt.cm.get_cmap('YlGn')
        cmap = cmap(np.linspace(0, 1, 256))
        cimg = cmap[simg.astype(np.uint8)]
    else:
        cimg = (simg - simg.min()) / (simg.max() - simg.min())
        cimg = np.stack([cimg] * 3, axis=-1)

    cimg0 = 255 * (cimg[:, :, 0] * (1 - mask))
    cimg1 = 255 * (cimg[:, :, 1] * (1 - mask))
    cimg2 = 255*cimg[:, :, 2] + mask * 255
    cimg0[cimg0 > 255] = 255
    outimg = np.concatenate(( cimg0[:, :, None],cimg1[:, :, None], cimg2[:, :, None]), axis=2)
    outimg = outimg.astype(np.uint8)
    return outimg

def create_edged_mask (mimg):
    edge = cv2.Canny(mimg, 20, 30)
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)
    mask = edge/np.max(edge)
    mask = (255*mask).astype(np.uint8)
    return mask

def get_marked_image(src_img,msk_img,mode = 'green'):
    simg = pydicom.dcmread(src_img)
    mima = 0
    for m in msk_img:
        mimg = pydicom.dcmread(m)
        mimg = mimg.pixel_array
        mima = mima + mimg.astype(float)
    simg = simg.pixel_array/256
    mima[mima > 255] = 255
    mimg = mima.astype(np.uint8)

    (h,w) = simg.shape
    simg = cv2.resize(simg, (round(w/4), round(h/4)))
    mimg = cv2.resize(mimg, (round(w/4), round(h/4)))

    if mode == 'trans':
        mask = mimg
        outimg = compose_trans_mask(simg, mask, mode=mode)
    else:
        mask = create_edged_mask(mimg)
        outimg = compose_mask(simg, mask,mode = mode)

    return outimg, simg, mask

def get_croped_mask(src_img,msk_img,borders=5, border_type='percentage', dsize=None):
    simg = pydicom.dcmread(src_img)
    simg = simg.pixel_array

    smask = pydicom.dcmread(msk_img)
    smask = smask.pixel_array
    bbox = get_corners(smask,label=255,borders=borders, border_type=border_type)
    cmask = crop_bbox(smask, bbox[0])
    cimg = crop_bbox(simg, bbox[0])

    if dsize is not None:
        cimg = cv2.resize(cimg, (dsize, dsize))
        cmask = cv2.resize(cmask, (dsize, dsize))
        simg = cv2.resize(simg, (dsize, dsize))

    return cimg, cmask, simg, bbox


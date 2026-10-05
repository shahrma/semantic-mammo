import argparse
import copy
import csv
import os.path
import tqdm
import pydicom
import pandas as pd
import cv2
from scripts.utils.imgutils import load_image

import matplotlib
matplotlib.use('Agg')
import pickle as pkl

import copy
import yaml


# Harmonise equivalent descriptor spellings in the CBIS-DDSM case descriptions
def fix_values(df):
    # Calcification
    key =  'calc morphology'
    if key in df.keys():
        df[key] = df[key].replace("LUCENT_CENTERED" ,"LUCENT_CENTER")
        df[key] = df[key].replace("PLEOMORPHIC-AMORPHOUS" ,"AMORPHOUS-PLEOMORPHIC")
        df[key] = df[key].replace("PLEOMORPHIC-PLEOMORPHIC" ,"PLEOMORPHIC")
        df[key] = df[key].replace("AMORPHOUS-ROUND_AND_REGULAR" ,"ROUND_AND_REGULAR-AMORPHOUS")
        df[key] = df[key].replace("PUNCTATE-LUCENT_CENTER" ,"LUCENT_CENTER-PUNCTATE")
        df[key] = df[key].replace("ROUND_AND_REGULAR-LUCENT_CENTERED" ,"ROUND_AND_REGULAR-LUCENT_CENTER")
        df[key] = df[key].replace("PUNCTATE-ROUND_AND_REGULAR" ,"ROUND_AND_REGULAR-PUNCTATE")
        df[key] = df[key].replace("COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED" ,"COARSE-ROUND_AND_REGULAR-LUCENT_CENTER")

    # Mass
    if 'mass shape' in df.keys():
        df["mass shape"] = df["mass shape"].replace("LOBULATED-OVAL", "OVAL-LOBULATED")
    if 'mass margins' in df.keys():
        df["mass margins"] = df["mass margins"].replace("OBSCURED-CIRCUMSCRIBED", "CIRCUMSCRIBED-OBSCURED")

    return df

from scripts.utils.imgutils import get_marked_image,get_croped_mask,load_image


def print_verbose(args, **kwargs):
    print(args, **kwargs)

def get_filepath_from_metadata(df,val) :
    SubjectID, StudyUID, SeriesUID, _ = val.split('/')
    curr = df.loc[(df['Subject ID'] == SubjectID) & (df['Study UID'] == StudyUID) & (df['Series UID'] == SeriesUID)]
    filepath = curr['File Location'].item()
    filepath = '/'.join(filepath.replace('\\','/').split('/')[1:])
    return filepath

#
# import numpy as np
# def classify_mass_shape(label):
#     shape_terms = ['ROUND', 'OVAL', 'LOBULATED', 'IRREGULAR']
#
#     parts = label.split('-')
#     shape_parts = [p for p in parts if p in shape_terms]
#     other_parts = [p for p in parts if p not in shape_terms]
#
#     if len(parts) == len(shape_parts):  # all are shapes
#         combined_shape = '-'.join(shape_parts)
#         return [combined_shape, None]
#     else:
#         shape = shape_parts[0] if shape_parts else None
#         adjacent = other_parts[0] if other_parts else None
#         return [shape, adjacent]




def get_subjects(metafile,csvfiles,basedir) :
    subjects = {}

    ASSESSMENT_MAP = {'BENIGN_WITHOUT_CALLBACK': 'benign',
                      'BENIGN': 'benign', 'MALIGNANT': 'malignant'}
    ASSESSMENTS = ('benign', 'malignant')
    SIDE_MAP = {'LEFT': 'left', 'RIGHT': 'right'}
    ABNORMALITY_MAP = {'calcification': 'calcification', 'mass': 'mass'}
    VIEW_MAP = {'CC': 'cc', 'MLO': 'mlo'}
    dfMeta = pd.read_csv(metafile)

    for filename in csvfiles:
    #   print_verbose(filename)
        dfDesc = pd.read_csv(filename)
        dfDesc = fix_values(dfDesc)
        for idx, line in tqdm.tqdm(dfDesc.iterrows(), desc='preload'):
            scan_key = (line['patient_id'], line['left or right breast'],
                        line['image view'], line['abnormality type'])
            side = SIDE_MAP[line['left or right breast']],
            view = VIEW_MAP[line['image view']],

            image_file_path = line['image file path'].strip()
            image_file_path = get_filepath_from_metadata(dfMeta, image_file_path)
            ROI_file_path1 = line['ROI mask file path'].strip()
            ROI_file_path1 = get_filepath_from_metadata(dfMeta, ROI_file_path1)
            ROI_file_path2 = line['cropped image file path'].strip()
            ROI_file_path2 = get_filepath_from_metadata(dfMeta, ROI_file_path2)

            #cropped_image_file_path = line['cropped image file path'].strip()
            #cropped_image_file_path = get_filepath_from_metadata(df, cropped_image_file_path)
            image_file_path = os.path.join(basedir, image_file_path , '1-1.dcm')
            if ROI_file_path1 == ROI_file_path2 :
                ROI_file_path1 = os.path.join(basedir, ROI_file_path1,'1-1.dcm')
                ROI_file_path2 = os.path.join(basedir, ROI_file_path2,'1-2.dcm')
            else :
                ROI_file_path1 = os.path.join(basedir, ROI_file_path1,'1-1.dcm')
                ROI_file_path2 = os.path.join(basedir, ROI_file_path2,'1-1.dcm')

            if os.path.getsize(ROI_file_path1) >  os.path.getsize(ROI_file_path2) :
                crop_file_path = ROI_file_path2
                mask_file_path = ROI_file_path1
            else :
                crop_file_path = ROI_file_path1
                mask_file_path = ROI_file_path2

            if not os.path.exists(image_file_path):
                print_verbose('File not found: %s' % image_file_path)
            elif not os.path.exists(crop_file_path):
                print_verbose('File not found: %s' % crop_file_path)
            elif not os.path.exists(mask_file_path):
                print_verbose('File not found: %s' % mask_file_path)
            else:
                with pydicom.filereader.dcmread(image_file_path, stop_before_pixels=True) as d:
                    img_size = (d['Rows'].value, d['Columns'].value)
                with pydicom.filereader.dcmread(crop_file_path, stop_before_pixels=True) as d:
                    crop_size = (d['Rows'].value, d['Columns'].value)
                with pydicom.filereader.dcmread(mask_file_path, stop_before_pixels=True) as d:
                    mask_size = (d['Rows'].value, d['Columns'].value)

                assert(not (crop_size[0] == mask_size[0]))


                if crop_size[0] > mask_size[0]:
                    crop_file_path, mask_file_path = mask_file_path, crop_file_path
                    crop_size, mask_size = mask_size, crop_size

                if scan_key not in subjects:
                    subjects[scan_key] = {
                        'patient_id':       line['patient_id'],
                        'side':             SIDE_MAP[line['left or right breast']],
                        'view':             VIEW_MAP[line['image view']],
                        'breast density':    int(line['breast density'] if 'breast density' in line else line['breast_density']),
                        'image_file':       image_file_path,
                        'modality' : 'MG',
                        'acquisition type': 'digitized',
                        'abnormality' : {},
                    }

                abnormality_id = line['abnormality id']
                abnormality_data = {
                    'abnormality type': ABNORMALITY_MAP[line['abnormality type']],
                    'assessment_score': int(line['assessment']),
                    'assessment_label': ASSESSMENT_MAP[line['pathology']],
                    'mask_file': mask_file_path,
                }

                for feature in ['mass shape', 'mass margins','subtlety','calc type','calc distribution'] :
                    if feature in line.keys() :
                        abnormality_data[feature] = line[feature]
                    else :
                        abnormality_data[feature] = None

                subjects[scan_key]['abnormality'][abnormality_id] = abnormality_data

    return subjects

def prepare_marked_workset(dataroot,outroot,mode = 'green',image_list=None,outext='.jpg'):
    lists = (('mass_case_description_train_set.csv', 'ddsm-WS00-mass-train.h5'),
             ('mass_case_description_test_set.csv', 'ddsm-WS00-mass-test.h5'),
             ('calc_case_description_train_set.csv', 'ddsm-WS00-calc-train.h5'),
             ('calc_case_description_test_set.csv', 'ddsm-WS00-calc-test.h5'))
    ''' 
    lists = (
             ('calc_case_description_train_set.csv', 'ddsm-WS00-calc-train.h5'),
             ('calc_case_description_test_set.csv', 'ddsm-WS00-calc-test.h5'))
    '''
    os.makedirs(outroot,exist_ok=True)
    csvroot = os.path.join(dataroot,'lists')
    basedir = os.path.join(dataroot,'data')
    metafile = os.path.join(dataroot,'data','metadata.csv')
    subjects_file = os.path.join(outroot,'subjects_file.pkl')
    os.makedirs(os.path.join(outroot,'marked'),exist_ok=True)
    os.makedirs(os.path.join(outroot, 'resized'), exist_ok=True)
    os.makedirs(os.path.join(outroot, 'masks'), exist_ok=True)
    csvfiles = [os.path.join(csvroot, list[0]) for list in lists]

    import pickle as pkl

    if not os.path.exists(subjects_file) :
        subjects = get_subjects(metafile,csvfiles,basedir)
        with open(subjects_file, 'wb') as f:
            pkl.dump(subjects, f)
        print('Created subjects_file')

    with open(subjects_file, "rb") as f:
        subjects = pkl.load(f)
        print('Loaded subjects_file')

    imglist = {}
    for k,v in subjects.items():
        outfile_ = '_'.join(list(k))
        if not outfile_ in image_list:
            continue
        outfile = os.path.join(outroot,'marked',f'{outfile_}{outext}')
        simgfile = os.path.join(outroot,'resized', f'{outfile_}.jpg')
        maskfile = os.path.join(outroot,'masks', f'{outfile_}.jpg')

        imglist[k] = {'marked':outfile , 'resized':simgfile, 'masks':maskfile}
        if not os.path.exists(outfile):
            image_file = v['image_file']
            mask_files = [a['mask_file'] for _,a in v['abnormality'].items()]

            try:
                outimg, simg, mask = get_marked_image(image_file, mask_files,mode=mode)
                cv2.imwrite(outfile, outimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(maskfile, mask)
                cv2.imwrite(simgfile, simg)
            except:
                print(f'{outfile} could not be created')

    return subjects, imglist


def adjust_to_meta_standard(meta):
    peta = {}

    peta['patient_id'] = meta['patient_id']
    peta['view'] = meta['view'].lower()
    peta['side'] = meta['side']
    peta['bbox'] = meta['bbox']

    peta['num abnormalities'] = meta['ab_num']
    peta['abnormality idx'] = meta['ab_idx']

    peta['abnormality type'] = meta['abnormality type']
    peta['pathology'] = meta['assessment_label'].lower()
    peta['assessment score'] = meta['assessment_score']

    if 'mass shape' in meta.keys() and meta['mass shape'] is not None:
        # [mass_shape, mass_shape_adjacent] = classify_mass_shape(meta['mass shape'])
        peta['mass shape'] = meta['mass shape']
        # peta['associated findings'] = mass_shape_adjacent

    if 'mass margins' in meta.keys() and meta['mass margins'] is not None:
        peta['mass margins'] = meta['mass margins']

    if 'calc type' in meta.keys() and meta['calc type'] is not None:
        peta['calc morphology'] = meta['calc type']
    if 'calc distribution' in meta.keys() and meta['calc distribution'] is not None:
        peta['calc distribution'] = meta['calc distribution']

    peta['subtlety'] = meta['subtlety']

    peta['image_file'] = meta['image_file']
    peta['mask_file'] = meta['mask_file']

    peta['breast density'] = str(meta['breast density'])

    peta['modality'] = 'MG'
    peta['acquisition type'] = 'digitized'

    return peta


def prepare_roi_workset(dataroot, outroot, borders=5, border_type='perecent', output_size=384):
    lists = (('mass_case_description_train_set.csv', 'ddsm-WS00-mass-train.h5'),
             ('mass_case_description_test_set.csv', 'ddsm-WS00-mass-test.h5'),
             ('calc_case_description_train_set.csv', 'ddsm-WS00-calc-train.h5'),
             ('calc_case_description_test_set.csv', 'ddsm-WS00-calc-test.h5'))
    # '''
    # lists = (
    #          ('calc_case_description_train_set.csv', 'ddsm-WS00-calc-train.h5'),
    #          ('calc_case_description_test_set.csv', 'ddsm-WS00-calc-test.h5'))
    # '''
    os.makedirs(outroot,exist_ok=True)
    csvroot = os.path.join(dataroot,'lists')
    basedir = os.path.join(dataroot,'data')
    metafile = os.path.join(dataroot,'data','metadata.csv')
    subjects_file = os.path.join(outroot,'subjects_file.pkl')
    os.makedirs(os.path.join(outroot,'rois'),exist_ok=True)
    os.makedirs(os.path.join(outroot,'full'),exist_ok=True)
    os.makedirs(os.path.join(outroot, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(outroot, 'meta'), exist_ok=True)
    csvfiles = [os.path.join(csvroot, list[0]) for list in lists]
    subjects_csv_file = os.path.join(outroot, 'subjects_file.csv')

    import pickle as pkl

    if not os.path.exists(subjects_file) :
        subjects = get_subjects(metafile,csvfiles,basedir)
        with open(subjects_file, 'wb') as f:
            pkl.dump(subjects, f)
        print('Created subjects_file')

    with open(subjects_file, "rb") as f:
        subjects = pkl.load(f)
        print('Loaded subjects_file')

    # df = subjects2table(subjects)
    # df.to_csv(subjects_csv_file)

    for k,v in subjects.items():
        outfile_ = '_'.join(list(k))
        vabns = v['abnormality']
        vbase = copy.deepcopy(v)
        vbase.pop('abnormality')

        for abk,abv in vabns.items():
            merged_dict = {**vbase, **abv}
            merged_dict['ab_idx'] = abk
            merged_dict['ab_num'] = len(vabns)
            outpatch_ = f'{outfile_}_{abk}'

            outimgfile = os.path.join(outroot,'rois',f'{outpatch_}.png')
            fullimgfile = os.path.join(outroot,'full',f'{outpatch_}.png')
            outmaskfile = os.path.join(outroot, 'masks', f'{outpatch_}.png')
            outmetafile = os.path.join(outroot, 'meta', f'{outpatch_}.yaml')

            if not os.path.exists(outimgfile):
                image_file = v['image_file']

                try:
                    cimg, cmask, simg, bbox = get_croped_mask(image_file, abv['mask_file'],
                                                              borders=borders,border_type=border_type, dsize=output_size)
                    merged_dict['bbox'] = bbox[0]
                    print(f'\rSaved : {outimgfile} ', end='', flush=True)

                    cv2.imwrite(fullimgfile, simg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    cv2.imwrite(outimgfile, cimg,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    cv2.imwrite(outmaskfile, cmask)

                    merged_dict['image_file'] = merged_dict['image_file'].replace(dataroot,'.')
                    merged_dict['mask_file'] = merged_dict['mask_file'].replace(dataroot,'.')

                    pmeta = adjust_to_meta_standard(merged_dict)

                    with open(outmetafile, 'w') as yaml_file:
                        yaml.dump(pmeta, yaml_file, default_flow_style=False)
                except:
                    print(f'{outimgfile} could not be created')

def subjects2table(subjects):
    lines = []
    for k, item in subjects.items():
        case_line = {k:v for k,v in item.items() if k not in ['abnormality']}
        case_line['name'] = '-'.join(k)
        case_line['num_abnormality'] = len(item['abnormality'])

        for abn_id, abn in item['abnormality'].items():
            line = copy.deepcopy(case_line)
            line['abnormality_id'] = int(abn_id)
            line.update({k:v for k,v in abn.items()})
            lines.append(line)

    df = pd.DataFrame(lines)
    return df
#
# def prepare_fullbody_workset(dataroot,outroot,mode = 'green',image_list=None,outext='.jpg'):
#     lists = (('mass_case_description_train_set.csv', ),
#              ('mass_case_description_test_set.csv', ),
#              ('calc_case_description_train_set.csv',),
#              ('calc_case_description_test_set.csv', ))
#     '''
#     lists = (
#              ('calc_case_description_train_set.csv', 'ddsm-WS00-calc-train.h5'),
#              ('calc_case_description_test_set.csv', 'ddsm-WS00-calc-test.h5'))
#     '''
#     os.makedirs(outroot,exist_ok=True)
#     csvroot = os.path.join(dataroot,'lists')
#     basedir = os.path.join(dataroot,'data')
#     metafile = os.path.join(dataroot,'data','metadata.csv')
#     subjects_file = os.path.join(outroot,'subjects_file.pkl')
#     subjects_csv_file = os.path.join(outroot, 'subjects_file.csv')
#
#     images_path = os.path.join(outroot,'images')
#     masks_path = os.path.join(outroot, 'masks')
#     os.makedirs(images_path,exist_ok=True)
#     os.makedirs(masks_path, exist_ok=True)
#     csvfiles = [os.path.join(csvroot, list[0]) for list in lists]
#
#     if not os.path.exists(subjects_file) :
#         subjects = get_subjects(metafile,csvfiles,basedir)
#         with open(subjects_file, 'wb') as f:
#             pkl.dump(subjects, f)
#         print('Created subjects_file')
#
#     with open(subjects_file, "rb") as f:
#         subjects = pkl.load(f)
#         print('Loaded subjects_file')
#
#     df = subjects2table(subjects)
#     df.to_csv(subjects_csv_file)
#
#     for index, row in df.iterrows():
#         info = row.to_dict()
#         name = info['name']
#         image_file = info['image_file']
#         mask_file = info['mask_file']
#         abnormality_id = info['abnormality_id']
#         if 'P_00059' in name:
#             print('')
#         outimage_file = os.path.join(images_path,f'{name}.png')
#         if not os.path.exists(outimage_file):
#             simg = load_image(image_file)
#             cv2.imwrite(outimage_file, simg)
#
#         outmask_file = os.path.join(masks_path,f'{name}-M{abnormality_id}.png')
#         mimg = load_image(mask_file)
#
#         if not (mimg.shape == simg.shape):
#             print(f'{name}  -- {mimg.shape[0] / simg.shape[0]} - {mimg.shape[1] / simg.shape[1]}')
#             mimg = cv2.resize(mimg, (simg.shape[1], simg.shape[0]))
#
#         cv2.imwrite(outmask_file, mimg, [cv2.IMWRITE_PNG_COMPRESSION, 1])
#

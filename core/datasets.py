import random

from torchvision.transforms import transforms
import torchvision.datasets as datasets

import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd

import torchvision.transforms as transforms
import numpy as np

from core.augmentations import StretchLevels, CustomTrainAugmentations, CustomValAugmentations,FinalAligning

# def classify_mass_shape(label):
#     shape_terms = ['ROUND', 'OVAL', 'LOBULATED', 'IRREGULAR']
#
#     if pd.isna(label):
#         return pd.Series([np.nan, np.nan])
#
#     parts = label.split('-')
#     shape_parts = [p for p in parts if p in shape_terms]
#     other_parts = [p for p in parts if p not in shape_terms]
#
#     if len(parts) == len(shape_parts):  # all are shapes
#         combined_shape = '-'.join(shape_parts)
#         return pd.Series([combined_shape, np.nan])
#     else:
#         shape = shape_parts[0] if shape_parts else np.nan
#         adjacent = other_parts[0] if other_parts else np.nan
#         return pd.Series([shape, adjacent])


def refine_meta(df):
    # df[['mass shape', 'mass shape adjacent']] = df['mass shape'].apply(classify_mass_shape)
    columns = [c for c in  ['breast density',
                'mass shape', 'mass margins', 'mass density',
                'calc associated', 'calc morphology', 'calc distribution',
                'abnormality type', 'num abnormalities','pathology'] if c in df.columns ]
    for col in columns:
        df[col] = df[col].fillna('N/A')

    df = df.reset_index(drop=True)

    for col in ['breast density','mass density', 'calc associated']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

def prepare_categories(meta, col, spliter=None, add_unknown=False, add_NA=True):
    if col not in meta.keys():
        categories = {'N/A' : 0}
        return categories
    print(f'prepare_categories : {col}')
    categories = sorted(list(meta[col].unique()))
    categories = [f'{name}' for idx, name in enumerate(categories)]

    if spliter is not None:
        categories = sorted(set(margin for item in categories for margin in item.split(spliter)))

    if add_NA:
        categories = [x for x in categories if x != 'N/A']
        categories = ['N/A'] + categories

    if add_unknown:
        categories = categories + ['unknown']

    categories = {name: idx for idx, name in enumerate(categories)}

    return categories

def define_meta_categories(meta,add_unknown):
    categories = {}
    categories['breast density'] = prepare_categories(meta, 'breast density',add_unknown=add_unknown)

    categories['mass shape'] = prepare_categories(meta, 'mass shape',spliter='-',add_unknown=add_unknown)
    categories['mass margins'] = prepare_categories(meta, 'mass margins', spliter='-',add_unknown=add_unknown)

    categories['mass density'] = prepare_categories(meta, 'mass density', spliter='-',add_unknown=add_unknown)

    categories['calc associated'] = prepare_categories(meta, 'calc associated', spliter='-',add_unknown=add_unknown)
    categories['calc morphology'] = prepare_categories(meta, 'calc morphology', spliter='-', add_unknown=add_unknown)
    categories['calc distribution'] = prepare_categories(meta, 'calc distribution', spliter='-', add_unknown=add_unknown)

    categories['abnormality type'] = prepare_categories(meta, 'abnormality type', add_unknown=add_unknown)
    categories['num abnormalities'] = prepare_categories(meta, 'num abnormalities', add_unknown=add_unknown)

    categories['pathology'] = prepare_categories(meta, 'pathology', add_NA=False)


    return categories

def define_label_categories(meta, target_col):
    return prepare_categories(meta, target_col)
#
# class DDSMRoiDataset(Dataset):
#     def __init__(self, data_path, folds, subset='full', transform=None, label_categories=None, channels = None,
#                  target_col=None,meta_categories=None, meta_augment=None,add_unknown=False):
#         self.data_path = data_path
#         self.folds = folds
#         self.subset = subset
#         self.transform = transform
#         self.meta_augment = meta_augment
#         df_meta = []
#         subset_list = ['mass','calcification'] if subset == 'full' else subset
#         self.channels = ['rois'] if channels is None else channels
#
#         for subset in subset_list:
#             for fold in folds:
#                 df_meta.append(pd.read_csv(os.path.join(data_path,'folds',subset,f'{fold}.csv'),index_col=0))
#
#         self.meta = pd.concat(df_meta,ignore_index=True)
#         self.shape_terms = ['ROUND', 'OVAL', 'LOBULATED', 'IRREGULAR']
#         self.meta = refine_meta(self.meta, self.shape_terms)
#
#         self.target_col = ['pathology'] if target_col is None else target_col
#         self.target_col = '-'.join(self.target_col)
#         if len(target_col) > 1:
#             self.meta[self.target_col] = self.meta[target_col].astype(str).agg('-'.join, axis=1)
#
#         self.label_categories = define_label_categories(self.meta, self.target_col) if label_categories is None else label_categories
#         self.meta_categories = define_meta_categories(self.meta,add_unknown) if meta_categories is None else meta_categories
#
#     def get_categories(self):
#         return self.label_categories, self.meta_categories
#
#     def __len__(self):
#         return len( self.meta)
#
#     def __getitem__(self, idx):
#         item = self.meta.iloc[idx]
#         category = item[self.target_col]
#         image_name = item['name']
#
#         channel_array = []
#         for channel in self.channels:
#             image_path = os.path.join(self.data_path, channel, f'{image_name}.png')
#             channel_array.append(np.array(Image.open(image_path)))
#
#         image = np.stack(channel_array, axis=0)
#
#         if self.transform:
#             image = self.transform(image)
#
#         label = self.label_categories[category]
#
#         def set_meta_heatmap(dict_cat_, category_):
#             heatmap = np.zeros([len(dict_cat_)])
#             for v in category_.split('-'): # The split is for cases that there are multi hits in the meta category
#                 heatmap[dict_cat_[v]] = float(1.0)
#             return heatmap
#
#         meta = {}
#         for k, dict_cat in self.meta_categories.items():
#             if self.meta_augment is None:
#                 meta[k] = set_meta_heatmap(dict_cat,item[k])
#             else:
#                 if 'random_unknown' in self.meta_augment:
#                     prob = self.meta_augment['random_unknown']['prob']
#                     if random.uniform(0, 1) > prob:
#                         meta[k] = set_meta_heatmap(dict_cat, 'unknown')
#                     else:
#                         meta[k] = set_meta_heatmap(dict_cat,item[k])
#
#         return image, label, meta, idx

def get_categories(dconfig, add_unknown=True):
    df_meta = []
    for dataname, data_config in dconfig.items():
        data_path = data_config['data_path']
        subset = data_config['subset']
        folds = data_config['folds_train']

        subset_list = ['mass', 'calcification'] if subset == 'full' else subset

        for subset in subset_list:
            for fold in folds:
                df_meta.append(pd.read_csv(os.path.join(data_path, 'folds', subset, f'{fold}.csv'), index_col=0))
    meta = pd.concat(df_meta, ignore_index=True)
    meta = refine_meta(meta)

    meta_categories = define_meta_categories(meta, add_unknown)

    return meta_categories

class RoiDataset(Dataset):
    def __init__(self, data_path, folds, subset='full', transform=None, channels = None,
                 target_col=None, meta_augment=None, categories=None,ablation_mode=None, expansion_method=None):
        self.data_path = data_path
        self.folds = folds
        self.subset = subset
        self.transform = transform
        self.meta_augment = meta_augment
        self.ablation_mode = [] if ablation_mode is None else ablation_mode
        df_meta = []
        subset_list = ['mass','calcification'] if subset == 'full' else subset
        self.channels = ['rois'] if channels is None else channels
        # self.expansion_method = 'duplicate' if expansion_method is None else expansion_method

        self.categories = categories
        if 'calc associated' in self.categories.keys():
            self.categories.pop('calc associated')

        for subset in subset_list:
            for fold in folds:
                df_meta.append(pd.read_csv(os.path.join(data_path,'folds',subset,f'{fold}.csv'),index_col=0))

        self.meta = pd.concat(df_meta,ignore_index=True)
        # self.shape_terms = ['ROUND', 'OVAL', 'LOBULATED', 'IRREGULAR']
        self.meta = refine_meta(self.meta)

        self.target_col = target_col
        # self.target_col = ['pathology'] if target_col is None else target_col
        # self.target_col = '-'.join(self.target_col)
        # if len(target_col) > 1:
        #     self.meta[self.target_col] = self.meta[target_col].astype(str).agg('-'.join, axis=1)

        # self.label_categories = define_label_categories(self.meta, self.target_col) if label_categories is None else label_categories
        # self.meta_categories = define_meta_categories(self.meta,add_unknown) if meta_categories is None else meta_categories

    def get_categories(self):
        return self.categories

    def __len__(self):
        return len( self.meta)

    def __getitem__(self, idx):
        item = self.meta.iloc[idx]
        category = item[self.target_col]
        image_name = item['name']
        from scipy.ndimage import gaussian_filter

        channel_array = []
        for channel in self.channels:
            image_path = os.path.join(self.data_path, channel, f'{image_name}.png')
            img = np.array(Image.open(image_path))
            if 'smoothing_3' in self.ablation_mode:
                img = gaussian_filter(img, sigma=3)
            channel_array.append(img)

        image = np.stack(channel_array, axis=0)

        if self.transform:
            image = self.transform(image)

        label = self.categories[self.target_col][category]

        def set_meta_heatmap(dict_cat_, category_):
            heatmap = np.zeros([len(dict_cat_)])
            if not isinstance(category_,str):
                pass
            for v in category_.split('-'): # The split is for cases that there are multi hits in the meta category
                try:
                    heatmap[dict_cat_[v]] = float(1.0)
                except:
                    pass
            return heatmap

        # NOTE: `self.categories` also contains the target column (main.py builds it as
        # [target_col] + descriptor groups), so `meta` includes a one-hot of the label
        # itself (e.g. meta['pathology']). This does not leak the label into the model:
        #   * WrappedConvNeXt: each MetaLayerNorm reads only meta[<its descriptor group>]
        #     (the replacing_list keys); the Base model (no replacing_list) ignores meta.
        #   * Loss: engine.calculate_multiple_loss only uses the groups listed in
        #     training.criteria; the configs list only 'main', which is the label returned
        #     separately by this method.
        # Do not add the target column to a model's descriptor groups or to the criteria,
        # or the label would become a model input.
        meta = {}
        for k, dict_cat in self.categories.items():
            try :
                if k in item.keys():
                    meta[k] = set_meta_heatmap(dict_cat,f'{item[k]}')
                else:
                    meta[k] = set_meta_heatmap(dict_cat, f'N/A')
            except:
                raise f'{k} is not in {item.keys()}'


            # if self.meta_augment is None:
            #     try :
            #         if k in item.keys():
            #             meta[k] = set_meta_heatmap(dict_cat,f'{item[k]}')
            #         else:
            #             meta[k] = set_meta_heatmap(dict_cat, f'N/A')
            #     except:
            #         pass
            # else:
            #     if 'random_unknown' in self.meta_augment:
            #         prob = self.meta_augment['random_unknown']['prob']
            #         if random.uniform(0, 1) > prob:
            #             meta[k] = set_meta_heatmap(dict_cat, 'unknown')
            #         else:
            #             meta[k] = set_meta_heatmap(dict_cat,item[k])

        return image, label, meta, idx



def get_ROI_dataset(dconfig, pconfig, categories):
    target_col = dconfig.get('target_col', 'pathology')
    channels = pconfig.get('channels', None)
    meta_augment = pconfig.get('meta_augment', None)

    # Global parameters
    output_size = pconfig.get('output_size', 256)
    depth_duplication = pconfig.get('depth_duplication', 1)
    augmentation_steps = pconfig.get('augmentation_steps', ["rotate", "crop", "jitter", "flip"])
    ablation_mode = pconfig.get('ablation_mode', None)
    train_datasets = []
    val_datasets = []

    for dataname, data_config in dconfig.items():
        stretch_type = data_config['preprocess'].get('stretch_type', 'min_max')
        stretch_args = data_config['preprocess'].get('stretch_args', None)

        transform_train = transforms.Compose([
            StretchLevels(stretch_type, args=stretch_args),
            CustomTrainAugmentations(output_size=output_size, params=pconfig.get('augmentation', dict()), steps=augmentation_steps),
            FinalAligning(output_size=output_size, depth_duplication=depth_duplication)
        ])

        transform_val = transforms.Compose([
            StretchLevels(stretch_type, args=stretch_args),
            # Should be the same of TrainAugmentation for compatible resizing (These class does not do random operation)
            CustomValAugmentations(output_size=output_size, params=pconfig.get('augmentation', dict()), steps=augmentation_steps),
            FinalAligning(output_size=output_size, depth_duplication=depth_duplication)
        ])



        train_dataset = RoiDataset(data_path=data_config['data_path'], subset=data_config['subset'], folds=data_config['folds_train'],
                                   transform=transform_train, channels=channels, target_col=target_col, meta_augment=meta_augment,
                                   categories=categories, expansion_method=None,ablation_mode=ablation_mode)
        train_datasets.append(train_dataset)

        val_dataset = RoiDataset(data_path=data_config['data_path'], subset=data_config['subset'], folds=data_config['folds_valid'],
                                transform=transform_val, channels=channels, target_col=target_col,
                                categories=categories, expansion_method=None,ablation_mode=ablation_mode)
        val_datasets.append(val_dataset)

    # Compose datasets using ConcatDataset
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    num_classes = len(categories[target_col])  # replace with the number of classes in your dataset
    return train_dataset, val_dataset, num_classes

#
#
# def get_dataset_CBIS_DDSM(dconfig):
#     output_size = dconfig.get('output_size', 256)
#     augmentation_steps = dconfig.get('augmentation_steps', ["rotate", "crop", "jitter", "flip"])
#     # max_rotation_deg = dconfig.get('max_rotation_deg', 5)
#     stretch_type = dconfig.get('stretch_type', 'min_max')
#     stretch_args = dconfig.get('stretch_args', None)
#     depth_duplication = dconfig.get('depth_duplication', 1)
#     workset_source = dconfig.get('source', 'manual')
#     channels = dconfig.get('channels', None)
#     target_col = dconfig.get('target_col', ['pathology'])
#     meta_augment = dconfig.get('meta_augment', None)
#     add_unknown = dconfig.get('add_unknown', False)
#
#     transform_train = transforms.Compose([
#         StretchLevels(stretch_type, args=stretch_args),
#         CustomTrainAugmentations(output_size=output_size, params=dconfig.get('augmentation', dict()), steps= augmentation_steps),
#         FinalAligning(output_size=output_size,depth_duplication=depth_duplication)
#     ])
#
#     transform_val = transforms.Compose([
#         StretchLevels(stretch_type, args=stretch_args),
#         # Should be the same of TrainAugmentation for compatible resizing (These class does not do random operation)
#         CustomValAugmentations(output_size=output_size, params=dconfig.get('augmentation', dict()), steps= augmentation_steps),
#         FinalAligning(output_size=output_size,depth_duplication=depth_duplication)
#     ])
#
#     if workset_source == 'kaggle':
#         from core.dataset_kpatches import DDSMKRoiDataset
#         train_dataset = DDSMKRoiDataset(data_directory=dconfig['data_path'], subset = dconfig['subset'], prefix='train', transform=transform_train)
#         label_categories, meta_categories = train_dataset.get_categories()
#         val_dataset = DDSMKRoiDataset(data_directory=dconfig['data_path'], subset = dconfig['subset'], prefix='test', transform=transform_val)
#     else:
#         train_dataset = DDSMRoiDataset(data_path=dconfig['data_path'], subset = dconfig['subset'], folds=dconfig['folds_train'],
#                                        transform=transform_train, channels=channels, target_col=target_col,meta_augment=meta_augment,add_unknown=add_unknown)
#         label_categories, meta_categories = train_dataset.get_categories()
#         val_dataset = DDSMRoiDataset(data_path=dconfig['data_path'], subset = dconfig['subset'], folds=dconfig['folds_valid'],
#                                      transform=transform_val, channels=channels, target_col=target_col,
#                                      label_categories=label_categories, meta_categories=meta_categories,add_unknown=add_unknown)
#
#     num_classes = len(label_categories)  # replace with the number of classes in your dataset
#     return train_dataset, val_dataset, num_classes


if __name__ == '__main__':
    # data path is different in number of backword direcotry levels
    dconfig = {'data_path' : '../../../worksets/DDSM/rois/data_250702_cfixed_resized_384',
               'type': 'roi', 'source': 'manual', 'subset': ['calcification'],
               'folds_train': ['train_0', 'train_1', 'train_2', 'train_3', 'train_4'],
               'folds_valid': ['test'],
               'folds_test': ['test'],
               'stretch_type': 'norm',
               'stretch_args': {'max_value': 65535, 'mean': 0.5, 'std': 0.25},
               'output_size': 224, 'depth_duplication': 3,
               'augmentation_steps': ['rotate', 'crop', 'jitter', 'flip', 'noise', 'erase'],
               'augmentation': {'rotate': {'max_rotation_deg': 15},
                                'noise': {'noise_a': 0.01, 'noise_b': 0.05},
                                'jitter': {'brightness_jitter': 0.15, 'contrast_jitter': 0.15},
                                'crop': {'scale_a': 0.9, 'scale_b': 1.1, 'max_shift': 20},
                                'erase': {'sl': 0.02, 'sh': 0.1, 'r1': 0.3}
                                }
               }

    output_size = dconfig.get('output_size', 256)
    augmentation_steps = dconfig.get('augmentation_steps', ["rotate", "crop", "jitter", "flip"])
    # max_rotation_deg = dconfig.get('max_rotation_deg', 5)
    stretch_type = dconfig.get('stretch_type', 'min_max')
    stretch_args = dconfig.get('stretch_args', None)
    depth_duplication = dconfig.get('depth_duplication', 1)
    workset_source = dconfig.get('source', 'manual')
    channels = dconfig.get('channels', None)
    target_col = dconfig.get('target_col', ['pathology'])

    transform_train = transforms.Compose([
        StretchLevels(stretch_type, args=stretch_args),
        CustomTrainAugmentations(output_size=output_size, params=dconfig.get('augmentation', dict()), steps= augmentation_steps),
        FinalAligning(output_size=output_size,depth_duplication=depth_duplication)
    ])

    train_dataset = RoiDataset(data_path=dconfig['data_path'], subset=dconfig['subset'], folds=dconfig['folds_train'], transform=transform_train,
                                   channels=channels, target_col=target_col)

    output = train_dataset.__getitem__(10)
    pass
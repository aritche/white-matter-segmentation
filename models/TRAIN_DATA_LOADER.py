import numpy as np
import random
from glob import glob
import torch
import cv2

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def displace(x, d_x, d_y, border_value=0):
    h, w, d = x.shape
    result = np.ones((h,w,d)) * border_value
    if d_x >= 0 and d_y >= 0:
        result[d_y:,d_x:,:] = x[:h-d_y,:w-d_x,:]
    elif d_x >= 0 and d_y < 0:
        result[:h+d_y,d_x:,:] = x[-d_y:,:w-d_x,:]
    elif d_x < 0 and d_y >= 0:
        result[d_y:,:w+d_x,:] = x[:h-d_y,-d_x:,:]
    else:
        result[:h+d_y,:w+d_x,:] = x[-d_y:,-d_x:,:]
    return result

def norm(data):
    data_normalized = np.zeros(data.shape, dtype=data.dtype)
    mean = data.mean()
    std = data.std() + 1e-7
    data_normalized = (data - mean) / std
    new_zero = (0 - mean) / std
    return data_normalized, new_zero

def augment_numpy(a, b, inplace=False, border_value=0):
    if not inplace:
        a = a.copy() # copy so that original matrices are not altered
        b = b.copy() # copy so that original matrices are not altered

    rotation_angle = np.random.uniform(-45,45)
    elastic_alpha, elastic_sigma = np.random.uniform(90,120), np.random.uniform(9,11)
    d_x, d_y = int(np.random.uniform(-10,10)), int(np.random.uniform(-10,10))
    zooming_factor = np.random.uniform(0.9,1.5)
    resampling_factor = np.random.uniform(0.5,1)
    noise_mean, noise_variance = 0, np.random.uniform(0,0.05)

    h, w, d = a.shape

    # 1. Elastic deformation (code from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a)
    # Some aspect sof code also from https://github.com/osuossu8/myEDAs/blob/dbae8bf2fd9542ed08af21610c223762d67e4ca5/forImage/elastic.py 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if np.random.uniform() < 0.2:
        dx = gaussian_filter((np.random.rand(h,w) * 2 - 1), elastic_sigma, mode="constant", cval=0) * elastic_alpha
        dy = gaussian_filter((np.random.rand(h,w) * 2 - 1), elastic_sigma, mode="constant", cval=0) * elastic_alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        xsum, ysum = np.float32(x+dx), np.float32(y+dy)

        for i in range(9):
            a[:,:,i] = cv2.remap(a[:,:,i], xsum, ysum, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

        for i in range(72):
            b[:,:,i] = cv2.remap(b[:,:,i], xsum, ysum, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 2. Rotation and Zooming
    if np.random.uniform() >= 0.2:
        rotation_angle = 0
    if np.random.uniform() >= 0.2:
        zooming_factor = 1

    if rotation_angle != 0 or zooming_factor != 1:
        mat = cv2.getRotationMatrix2D((w//2, h//2), rotation_angle, zooming_factor) 
        for i in range(9):
            a[:,:,i] = cv2.warpAffine(a[:,:,i], mat, (144,144), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        b = cv2.warpAffine(b, mat, (144,144), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. Displacement
    if np.random.uniform() < 0.2:
        a = displace(a, d_x, d_y, border_value)
        b = displace(b, d_x, d_y)

    # 4. Resampling
    if np.random.uniform() < 0.2:
        a = cv2.resize(a, (int(resampling_factor*w), int(resampling_factor*h)), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, (w,h), interpolation=cv2.INTER_NEAREST)

    # 5. Gaussain noise
    if np.random.uniform() < 0.2:
        a = a + np.random.normal(loc=noise_mean, scale=noise_variance, size=a.shape)

    return a,b

def get_data(peaks_fn, segs_fn, means, sdevs, do_augment):
    # Load data
    peaks = np.load(peaks_fn)
    segs = np.load(segs_fn)

    # Normalise sample to zero mean and unit variance
    peaks, new_zero = norm(peaks)

    # Augment the peaks
    if do_augment:
        peaks, segs = augment_numpy(peaks, np.float32(segs), new_zero)

    # Convert to torch tensors
    peaks = torch.from_numpy(np.float32(peaks))
    peaks = peaks.permute(2,0,1) # channels first for pytorch
    segs = torch.from_numpy(np.float32(segs))
    segs = segs.permute(2,0,1) # channels first for pytorch

    return [peaks, segs]

def get_folds(folds_dir):
    peaks, segs = [], []
    for fold_num in ['1', '2', '3', '4', '5']:
        peak_fns = sorted(list(set(['_'.join(item.split('_')[:-2]) for item in glob(folds_dir + '/fold' + fold_num +'/peaks/*.npy')])))
        seg_fns = sorted(list(set(['_'.join(item.split('_')[:-2]) for item in glob(folds_dir + '/fold' + fold_num + '/segmentations/*.npy')])))

        peaks.append(peak_fns)
        segs.append(seg_fns)

    return peaks, segs

def breakdown_folds(peak_folds, seg_folds, eval_fold, valid_fold):
    # Adjust folds to range [0,4]
    eval_fold, valid_fold = eval_fold-1, valid_fold-1

    test_peaks, test_segs = peak_folds[eval_fold], seg_folds[eval_fold]
    valid_peaks, valid_segs = peak_folds[valid_fold], seg_folds[valid_fold]

    train_peaks, train_segs = [], []
    for i in range(len(peak_folds)):
        if i != eval_fold and i != valid_fold:
            train_peaks += peak_folds[i]
            train_segs += seg_folds[i]

    return [train_peaks, train_segs], [valid_peaks, valid_segs], [test_peaks, test_segs]


def get_stats(extensions, peaks_fn):
    means = np.float32(np.zeros(9))
    sdevs = np.float32(np.zeros(9))

    print("Calculating mean for the dataset...")
    num_means = 0
    for base_fn in peaks_fn:
        for extension in extensions: 
            for slice_num in range(144):
                fn = base_fn + '_' + extension + '_' + str(slice_num) + '.npy'
                if num_means % 1000 == 0:
                    print(num_means/(len(peaks_fn)*len(extensions)*144)*100)
                data = np.load(fn)
                means += np.mean(data.reshape((-1,9)), axis=0)
                num_means += 1
    means = means/num_means

    print("Calculating sdev for the dataset...")
    squared_diffs_sum = np.float32(np.zeros(9))
    num_sdevs = 0
    for base_fn in peaks_fn:
        for extension in extensions:
            for slice_num in range(144):
                fn = base_fn + '_' + extension + '_' + str(slice_num) + '.npy'
                if num_sdevs % 1000 == 0:
                    print(num_sdevs/(len(peaks_fn)*len(extensions)*144)*100)
                data = np.load(fn)
                pixels = data.reshape((-1,9))
                diff = (pixels - means)**2
                mean_diff = np.mean(diff,axis=0)
                squared_diffs_sum += mean_diff
                num_sdevs += 1
    sdevs = (squared_diffs_sum / num_sdevs) ** (1/2)

    return means, sdevs

def paired_shuffle(a,b):
    merged = list(zip(a,b))
    random.shuffle(merged)
    new_a,new_b = zip(*merged)

    return new_a, new_b

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folds_dir, eval_fold, valid_fold, is_validation=False, do_augment=True, means=np.array([]), sdevs=np.array([]), fake_limit=None):
        self.is_validation = is_validation
        self.do_augment = do_augment
        self.extensions = ['sagittal', 'axial', 'coronal']

        # Get the root file names for each data fold (e.g. './xxx/folds/fold1/peaks/992774')
        peak_folds, seg_folds = get_folds(folds_dir)
        f1_peaks, f2_peaks, f3_peaks, f4_peaks, f5_peaks = peak_folds
        f1_segs, f2_segs, f3_segs, f4_segs, f5_segs = seg_folds

        # Merge/split folds into training, validation, and test
        train, valid, test = breakdown_folds(peak_folds, seg_folds, eval_fold, valid_fold)

        train_peaks, train_segs = train
        valid_peaks, valid_segs = valid
        test_peaks, test_segs = test

        # Compute mean/sdev of non-test set
        if len(means) == 0:
            self.means, self.sdevs = get_stats(self.extensions, train_peaks+valid_peaks)
        else:
            self.means, self.sdevs = means, sdevs
        self.means, self.sdevs = np.float32(self.means), np.float32(self.sdevs)
        print(self.means)
        print(self.sdevs)

        if fake_limit != None:
            #train_peaks, train_segs = paired_shuffle(train_peaks, train_segs)
            train_peaks = train_peaks[:fake_limit]
            train_segs = train_segs[:fake_limit]

        if self.is_validation:
            self.peak_fns = valid_peaks
            self.seg_fns = valid_segs
        else:
            self.peak_fns = train_peaks
            self.seg_fns = train_segs

        # Adjust the final dataset to include all 144 slices
        peak_slices, seg_slices = [], []
        for item in self.peak_fns:
            for i in range(144):
                peak_slices.append(item + '_' + str(i) + '.npy')
        for item in self.seg_fns:
            for i in range(144):
                seg_slices.append(item + '_' + str(i) + '.npy')
        self.peak_fns = peak_slices
        self.seg_fns = seg_slices

        print(str(len(self.peak_fns)) + ' peaks have been loaded')
        print(str(len(self.seg_fns)) + ' segs have been loaded')

    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):
        # Randomly choose an orientation for the current slice
        r = np.random.randint(0,3)
        peak_fn = '_'.join(self.peak_fns[idx].split('_')[:-1]) + '_' + self.extensions[r] + '_' + self.peak_fns[idx].split('_')[-1]
        seg_fn = '_'.join(self.seg_fns[idx].split('_')[:-1]) + '_' + self.extensions[r] + '_' + self.seg_fns[idx].split('_')[-1]
        return get_data(peak_fn, seg_fn, self.means, self.sdevs, self.do_augment)

    def __len__(self):
        return len(self.peak_fns)

    def get_std_params(self):
        return [self.means, self.sdevs]

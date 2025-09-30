import numpy as np
import os
import glob
import joblib
import argparse
from tqdm import tqdm

from nilearn.input_data import NiftiMasker
import nibabel as nib

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir, standardize

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='en',
                    help='language: en, fr or cn')
args = parser.parse_args()
lang = args.lang.lower()

assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'

fmri_data_resampled = os.path.join(lbl.home_folder, 'lpp_{}_resampled'.format(lang))
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, 'sub-{}*'.format(lang.upper()))))

fmri_data_avg_subject = os.path.join(lbl.home_folder, 'lpp_{}_average_subject'.format(lang))
make_dir(fmri_data_avg_subject)

n_runs = lbl.n_runs
t_r = lbl.t_r

fmri_subs_runs = []
for sub_id in tqdm(subject_list):
    sub_id_basename = os.path.basename(sub_id)
    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, '*.nii.gz')))
    fmri_runs = [] # n_runs x n_timesteps x n_voxels
    for fmri_img in fmri_imgs_sub:
        nifti_masker = NiftiMasker(mask_img='mask_lpp_{}.nii.gz'.format(lang), detrend=True, standardize=True,
                                   high_pass=1/128, t_r=t_r)
        fmri_runs.append(nifti_masker.fit_transform(fmri_img))
    fmri_subs_runs.append(fmri_runs)

for run in range(n_runs):
    fmri_mean_sub = np.mean([fmri_sub_runs[run] for fmri_sub_runs in fmri_subs_runs], axis=0)
    fmri_mean_sub = standardize(fmri_mean_sub, axis=0)
    filename = os.path.join(fmri_data_avg_subject, 'average_subject_run-{}.gz'.format(run))
    joblib.dump(fmri_mean_sub, filename, compress=4)

# now compute reliable voxels
from sklearn.linear_model import Ridge
import time

np.random.seed(1234)

n_subjects = len(subject_list)
n_voxels = nifti_masker.n_elements_

alphas = np.logspace(2,7,16)

n_trials = 10

corr_split = []
for i_trial in range(n_trials):
    print('='*80)
    
    idx_random = np.arange(n_subjects)
    np.random.shuffle(idx_random)
    
    idx_group_1 = idx_random[:n_subjects//2]
    idx_group_2 = idx_random[n_subjects//2:]
    
    regressors_runs = [np.mean([fmri_subs_runs[idx_sub][run][10:-10] for idx_sub in idx_group_1], axis=0)
                                   for run in range(n_runs)]
    fmri_runs = [np.mean([fmri_subs_runs[idx_sub][run][10:-10] for idx_sub in idx_group_2], axis=0)
                                   for run in range(n_runs)]

    corr_runs = []
    for run_test in range(n_runs):
        tic = time.time()
        
        runs_train = np.setdiff1d(np.arange(n_runs), run_test)
        x_train = np.vstack([regressors_runs[run_train] for run_train in runs_train])
        x_test = regressors_runs[run_test]
        y_train = np.vstack([fmri_runs[run_train] for run_train in runs_train])
        y_test = fmri_runs[run_test]
        
        ############ start nested CV 
        #leave another run apart as a validation test
        run_val = runs_train[0]
        runs_train_val = np.setdiff1d(runs_train, run_val)
        x_train_val = np.vstack([regressors_runs[run_train_val] for run_train_val in runs_train_val])
        x_val = regressors_runs[run_val]
        y_train_val = np.vstack([fmri_runs[run_train] for run_train in runs_train_val])
        y_val = fmri_runs[run_val]

        corr_val = []
        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(x_train_val, y_train_val)
            y_pred = model.predict(x_val)
            corr_tmp = [np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]
            corr_val.append(corr_tmp)    

        idx_best_alpha = np.argmax(np.mean(corr_val, axis=1))
        alpha = alphas[idx_best_alpha]
        ############ end nested CV 
        
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        corr_tmp = [np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]

        corr_runs.append(corr_tmp)
        
        toc = time.time()
        
        print('run ', run_test, '\t', 'mean = {:.03f}'.format(np.mean(corr_tmp)), '\t',
            'max = {:.03f}'.format(np.max(corr_tmp)), '\t',
            'time elapsed = {:.03f}'.format(toc-tic))

    corr_split.append(np.mean(corr_runs, axis=0))

filename = 'isc_{}trials_{}.gz'.format(n_trials, lang)
joblib.dump(np.array(corr_split), filename, compress=4)
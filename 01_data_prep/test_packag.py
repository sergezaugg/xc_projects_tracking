

# --------------
# Author : Serge Zaugg
# Description : Minimalistic demo example of xco usage in practice
# For real projects we recommend using a dir outside if this repo
# --------------

# install xco package from github
# pip install https://github.com/sergezaugg/xeno_canto_organizer/releases/download/vx.x.x/xeno_canto_organizer-x.x.x-py3-none-any.whl

import os
import xeno_canto_organizer.xco as xco
# import src.xeno_canto_organizer.xco as xco

# make a projects dir, if it does not already exist
if not os.path.isdir('./temp_xc_project'):
    os.makedirs('./temp_xc_project')

# Make an instance of the XCO class and define the start path 
xc = xco.XCO(start_path = './temp_xc_project')

# A few examples (on first use xc api key must be provided)
xc.download_summary(gen = "Corvus", sp = "corone", cnt = "France",verbose=True)
xc.download_summary(gen = "Corvus",cnt = "Germany",q = "B",  len_min = 10,verbose=True)
xc.download_summary(gen = "Parus",sp = "major",cnt = "switzerland",  q = ">C",len_max = 10 , verbose=True)
xc.download_summary(cnt = "Spain",q = "A",len_min = 5,len_max = 7,verbose=True)
xc.download_summary(gen = "Parus",sp = "major", cnt = "Spain",q = "A",len_min = 8,len_max = 14,verbose=True)



xc.reload_local_summary()

xc.df_recs.shape
print(xc.df_recs['gen'].value_counts())
print(xc.df_recs['q'].value_counts())
print(xc.df_recs['lic'].value_counts())

# Download the files 
xc.download_audio_files(verbose=True)
# Convert mp3s to wav with a specific sampling rate (requires ffmpeg to be installed)
xc.mp3_to_wav(conversion_fs = 24000)
# Extract spectrograms from segments and store as PNG
xc.extract_spectrograms(
    fs_tag = 24000, 
    segm_duration = 3.0, 
    segm_step = 0.5, 
    win_siz = 512, 
    win_olap = 192, 
    max_segm_per_file = 12, 
    specsub = True, 
    colormap='viridis',
    verbose=True
    )


#---------------------------------------
# Open a new session
# The pre-downloaded mp3 files can be reprocessed with different parameters 
# Point XCO to the dir with pre-downloaded mp33
import xeno_canto_organizer.xco as xco

xc = xco.XCO(start_path = './temp_xc_project')

# Make wavs with fs = 20000 and then short spectrogram 
xc.mp3_to_wav(conversion_fs = 20000)
xc.extract_spectrograms(fs_tag = 20000, segm_duration = 0.202, segm_step = 0.5, win_siz = 256, win_olap = 220.5, max_segm_per_file = 20, 
                        specsub = True, colormap='gray')

# Make  Make wavs with fs = 16000 and then long spectrogram 
xc.mp3_to_wav(conversion_fs = 16000)
xc.extract_spectrograms(fs_tag = 16000, segm_duration = 1.738, segm_step = 0.95, win_siz = 256, win_olap = 220.00, max_segm_per_file = 20, 
                        specsub = False, colormap='viridis')








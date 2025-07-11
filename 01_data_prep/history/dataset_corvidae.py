#----------------------
# Author : Serge Zaugg
# Description : Example of xco usage in practice 
#----------------------

import xeno_canto_organizer.xco as xco

# Make an instance of the XCO class and define the start path 
xc = xco.XCO(start_path = 'd:/xc_real_projects/xc_corvidae_01')
# Create a template json parameter file (to be edited)
xc.make_param(filename = 'download_corvidae.json', template = "corvidae")
# Get information of what will be downloaded
xc.download_summary(params_json = 'download_corvidae.json')
# make summary tables 
print(xc.df_recs.shape)
# keep only if fs large enough 
sel = xc.df_recs['smp'].astype(int)>= 24000
xc.df_recs = xc.df_recs[sel]
# Download the files 
xc.download_audio_files()
# Convert mp3s to wav with a specific sampling rate (requires ffmpeg to be installed)
xc.mp3_to_wav(conversion_fs = 24000)
# Make rectangular spectrogram with size = 128 freq x 256 time 
xc.extract_spectrograms(fs_tag = 24000, segm_duration = 0.394 , segm_step = 0.80, win_siz = 256, win_olap = 220.5, 
                        max_segm_per_file = 16, equalize = True, colormap='gray')



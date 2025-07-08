# --------------
# Author : Serge Zaugg
# Description : 
# --------------

import xeno_canto_organizer.xco as xco

xco.api_key_to_env()

xc2 = xco.XCO(start_path = "D:/xc_real_projects/xc_data_01_train")

xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 10, len_max = 10, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 12, len_max = 12, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 14, len_max = 14, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 16, len_max = 16, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 18, len_max = 18, verbose=True)

# len(xc2.recs_pool)
xc2.compile_df_and_save(verbose = True)

xc2.reload_local_summary()
xc2.df_recs.shape
print(xc2.df_recs['length'].value_counts())
print(xc2.df_recs['full_spec_name'].value_counts())
print(xc2.df_recs['smp'].value_counts())
print(xc2.df_recs['gen'].value_counts())
print(xc2.df_recs['lic'].value_counts())

# successfully performed on 20250707-22:40
xc2.download_audio_files(verbose=True)

# Done! successfully converted: 10023 files, failed: 30
xc2.mp3_to_wav(conversion_fs = 24000)

# 
xc2.extract_spectrograms(
    fs_tag = 24000, 
    # segm_duration = 0.394, segm_step = 0.50, win_siz = 256, win_olap = 220.5, max_segm_per_file = 20, # rectangular spectrogram 128 freq x 256 time 
    segm_duration = 1.738, segm_step = 0.95, win_siz = 256, win_olap = 220.5, max_segm_per_file = 20, # long spectrogram 128 freq x 1152 time (1024+128= 1152)
    specsub = True, log_f_min = 0.02, colormap='viridis', verbose = True)


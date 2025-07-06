

# --------------
# Author : Serge Zaugg
# Description : Minimalistic demo example of xco usage in practice
# For real projects we recommend using a dir outside if this repo
# --------------



import xeno_canto_organizer.xco as xco
# Store API key in env variable of current session - needed for download_summary()
xco.api_key_to_env()
#---------------------------------
# (Example 2) search can be based on family and larger areas + sampling rate limits 
xc2 = xco.XCO(start_path = './temp_xc_project_02')
xc2.download_summary(fam = "Corvidae", area = "Europe", smp_min = 16000, smp_max = 16000,  len_min = 1, len_max = 10, verbose=True)
xc2.download_summary(fam = "Paridae", area = "Europe", smp_min = 16000, smp_max = 16000,  len_min = 1, len_max = 20, verbose=True)

xc2.compile_df_and_save(verbose = True)
xc2.download_audio_files(verbose=True)
xc2.mp3_to_wav(conversion_fs = 8000)
xc2.extract_spectrograms(fs_tag = 8000, segm_duration = 1.0, segm_step = 0.5, win_siz = 256, win_olap = 220.5, 
                        max_segm_per_file = 20, specsub = True, log_f_min = None, colormap='gray')


# --------------
# Author : Serge Zaugg
# Description : Minimalistic demo example of xco usage in practice
# For real projects we recommend using a dir outside if this repo
# --------------

import xeno_canto_organizer.xco as xco

xco.api_key_to_env()



xc2 = xco.XCO(start_path = "D:/xc_real_projects/xc_dataset_train_01")

xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 10, len_max = 10, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 12, len_max = 12, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 14, len_max = 14, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 16, len_max = 16, verbose=True)
xc2.download_summary(area = "Europe", smp_min = 24000, smp_max = 48000, q = "A", len_min = 18, len_max = 18, verbose=True)

# len(xc2.recs_pool)
xc2.compile_df_and_save(verbose = True)
# xc2.df_recs.columns

xc2.df_recs.shape
print(xc2.df_recs['length'].value_counts())
print(xc2.df_recs['full_spec_name'].value_counts())
print(xc2.df_recs['smp'].value_counts())
print(xc2.df_recs['gen'].value_counts())
print(xc2.df_recs['lic'].value_counts())


xc2.download_audio_files(verbose=True)


xc2.mp3_to_wav(conversion_fs = 24000)



xc2.extract_spectrograms(
    fs_tag = 24000, 
    segm_duration = 1.0, 
    segm_step = 0.5, 
    win_siz = 256, 
    win_olap = 220.5, 
    max_segm_per_file = 20, 
    specsub = True, 
    log_f_min = None, 
    colormap='gray'
    )






# xc2.download_summary(fam = "Corvidae", area = "Europe", smp_min = 16000, smp_max = 16000,  len_min = 1, len_max = 10, verbose=True)
# xc1.download_summary(gen = "Corvus", sp = "corax", cnt = "switzerland", q = "A", len_max = 200, smp_min = 44100, verbose=True)
# xc1.download_summary(gen = "Pyrrhocorax",          cnt = "France",      q = "B", len_max = 5,   smp_min = 44100, verbose=True)
# xc1.download_summary(gen = "Coloeus",              cnt = "Belgium",     q = "C", len_max = 50 , smp_min = 44100, verbose=True)




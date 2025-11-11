from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = 'D:\\study\\level 4\\Image_processing-1\\Project\\Assignment_4\\lasot'
    settings.network_path = 'D:\\study\\level 4\\Image_processing-1\\Project\\Assignment_4\\test\\networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = 'D:\\study\\level 4\\Image_processing-1\\Project\\Assignment_4\\SeqTrack'
    settings.result_plot_path = 'D:\\study\\level 4\\Image_processing-1\\Project\\Assignment_4\\test\\result_plots'
    settings.results_path = 'D:\\study\\level 4\\Image_processing-1\\Project\\Assignment_4\\test\\tracking_results'    # Where to store tracking results
    settings.save_dir = 'D:\\study\\level 4\\Image_processing-1\\Project\\Assignment_4'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings


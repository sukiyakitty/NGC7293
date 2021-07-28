NGC7293: JetBrains PyCharm Python program Main forder

Cardinal.py: main entrance for real time analysis

Scheduling_x.py sequential execution of auto processing and analysis in real time 

Run.py: entrance for off-line analysis

Lib_Class.py: the class library

Lib_Function.py: functions
image enhancement
cell density
all kinds of stitching

Lib_Features.py: Features analysis functions
research_stitched_image_elastic_bat


Make_a_Call.py : GSM Modular can make a telephone call


For examples, the following is an example of the entire processing:

run Cardinal.py, using: python Cardinal.py
Cardinal.py will call Scheduling_x.py for image getting and Scheduling follow ZEN software auto exporting.
Scheduling_x.py call stitching_CZI_AutoBestZ()(in Lib_Function.py) for image stitching.
Scheduling_x.py call get_AE_density()(in Lib_Function.py) for density getting.
Scheduling_x.py call RT_PGC_Features()(in Lib_Features.py) for image enhancing and features generating.
Scheduling_x.py call call_analysis() for other analysis.
RT_PGC_Features() call core_features_one_image() for "SIFT\SURF\ORB" features generating.
The Features are saving in "Features" folder under the main folder(main_path).
In the "Features" folder, there are "S1.csv","S2.csv","S3.csv"... "S" means well.
In one "S1.csv" file: row is time point; col is features vector sets.

After we got "S1.csv" file, next we reducing the feature dimensions(also called Data Visualization or called Manifold analysis).
Noticing: you can using merge_all_well_features() concatenate "S1.csv","S2.csv","S3.csv"... to one file: 'All_FEATURES.csv'
Using tools in Lib_Manifold.py, we can do the Manifold analysis in batches.
For examples, using do_manifold(), we can reducing features in "features.csv" to an "PCA.csv" file(setting to Keep only the first 3 dimensions).
Noticing, do_manifold() can also outputing: "Isomap.CSV", "LLE.csv", "MDS.csv", "Modified_LLE.csv", "PCA.csv", "SpectralEmbedding.csv", "tSNE.csv"
We can draw Visualization figures using those manifold analysis outputs.

exp:
    main_path=r'D:\Image_Processing\CD13'
    features_csv=r'All_FEATURES.csv'
    features_cols = range(0, 255)
    output_folder = r'test20210727'
    do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)

Next, draw using draw_mainfold() function.

exp:
    main_path = r'D:\Image_Processing\CD13'
    mainfold_path = r'test20210727'
    exp_file = r'Experiment_Plan.csv'
    draw_mainfold(main_path, mainfold_path, exp_file)



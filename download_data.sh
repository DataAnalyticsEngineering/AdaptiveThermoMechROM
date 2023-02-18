OPTIONS="-q --show-progress"
DATASET=https://darus.uni-stuttgart.de/api/access/datafile/:persistentId\?persistentId\=doi:10.18419/darus-2822
wget $OPTIONS $DATASET/9 -O input/striped_normal_4x4x4.h5  # microstructures[0]
wget $OPTIONS $DATASET/8 -O input/sphere_normal_16x16x16_10samples.h5  # microstructures[1]
wget $OPTIONS $DATASET/3 -O input/sphere_normal_32x32x32_10samples.h5  # microstructures[2]
wget $OPTIONS $DATASET/10 -O input/sphere_combo_16x16x16_10samples.h5  # microstructures[3]
wget $OPTIONS $DATASET/4 -O input/octahedron_normal_16x16x16_10samples.h5  # microstructures[4]
wget $OPTIONS $DATASET/6 -O input/octahedron_combo_16x16x16_10samples.h5  # microstructures[5]
wget $OPTIONS $DATASET/11 -O input/octahedron_combo_32x32x32.h5  # microstructures[6]
wget $OPTIONS $DATASET/2 -O input/random_rve_vol20.h5  # microstructures[7]
wget $OPTIONS $DATASET/5 -O input/random_rve_vol40.h5  # microstructures[8]
wget $OPTIONS $DATASET/1 -O input/random_rve_vol60.h5  # microstructures[9]

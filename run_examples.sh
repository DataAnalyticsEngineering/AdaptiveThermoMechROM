mkdir output
# https://github.com/remykarem/python2jupyter
# pip install p2j
p2j eg5_FFANN.py
# reverse: p2j p2j -r eg5_FFANN.ipynb
/usr/bin/time -v -o output/eg2.time python -u eg2_compare_approximations.py > output/eg2.out
/usr/bin/time -v -o output/eg3.time python -u eg3_hierarchical_sampling.py > output/eg3.out

# try:
#     import pyminirom
# except ImportError as e:
#     !rm -rf pyminirom_repo
#     !rm -rf data_lfs
#     !git clone --depth 1 https://gitlab+deploy-token-629071:4DEL-qAps7zKAyMGXEQD@gitlab.com/shadialameddin/pyminirom.git pyminirom_repo
#     !git clone --depth 1 https://gitlab.com/shadialameddin/data_lfs.git data_lfs && cd data_lfs && git lfs pull
#     !cd pyminirom_repo && pip install -e .
#     # pip install numpy --upgrade # in case of error when importing spams
#     import os
#     os.kill(os.getpid(), 9) # used to kill Google colab runtime in order to use the newly installed packages
# 
#     # avoid restarting the colab notebook after `pip install -e`
#     import site
#     site.main() -->

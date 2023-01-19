from plot_utils import visual_trend

# absolute response time change
visual_trend('../dataset/timecare_block.csv','resptime_abs')

# relative response time change
visual_trend('../dataset/timecare_block.csv','resptime_rela')

# absolute accuracy change
visual_trend('../dataset/timecare_block.csv','accuracy_abs')

# relative accuracy change
visual_trend('../dataset/timecare_block.csv','accuracy_rela')


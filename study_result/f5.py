from plot_utils import visual_result

# boxplot for absolute accuracy change between two groups
visual_result('../dataset/timecare_spss.csv','accuracy_abs','box')

# boxplot for absolute response time change between two groups
visual_result('../dataset/timecare_spss.csv','resptime_abs','box')

# boxplot for absolute attention change between two groups
visual_result('../dataset/timecare_spss.csv','attention_abs','box')

# boxplot for absolute anxiety change between two groups
visual_result('../dataset/timecare_spss.csv','anxiety_abs','box')

# histplot for absolute accuracy change between two groups
visual_result('../dataset/timecare_spss.csv','accuracy_abs','hist')

# histplot for absolute response time change between two groups
visual_result('../dataset/timecare_spss.csv','resptime_abs','hist')

# histplot for absolute attention change between two groups
visual_result('../dataset/timecare_spss.csv','attention_abs','hist')

# histplot for absolute anxiety change between two groups
visual_result('../dataset/timecare_spss.csv','anxiety_abs','hist')



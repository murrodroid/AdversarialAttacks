import pingouin as pg
from scipy.stats import shapiro

'''
Image_quality - changes depending om method of IQA used
Image_id - adversarial image path (ID)
'''

#Load dataset
df = ...


#-------------------------------------------------------------
#CHECK FOR ANOVA ASSUMPTIONS
#Check for Sphericity assumption
pg.sphericity(df, dv='image_quality', subject='image_id', within=['attack_type'])
pg.sphericity(df, dv='image_quality', subject='image_id', within=['target_label'])
pg.sphericity(df, dv='image_quality', subject='image_id', within=['attack_type', 'target_label'])


#Check for normality assumption 
'''We assume the unexplained variation/"noise" is normally distributed. 
Needs to be done after fitting ANOVA'''

df['group_mean'] = df.groupby(['image_id', 'attack_type', 'target_label'])['image_quality'].transform('mean')
df['residual'] = df['image_quality'] - df['group_mean']

shapiro(df['residual'])

#-------------------------------------------------------------


#Documentation: https://pingouin-stats.org/build/html/generated/pingouin.rm_anova.html
#Perform Two-way Repeated Measures ANOVA
aov = pg.rm_anova(
    dv='image_quality',              # The outcome/dependent variable
    within=['attack_type', 'target_label'],  # The within-subject factors
    subject='image_id',              # Identifier for repeated measures (the unit)
    data=df,                         # Your full dataframe
    detailed=True                    # Gives additional output like effect size (η²p)
)

#Results example 
'''
| Source                       | SS    | DF  | MS     | F    | p-unc | np2  |
| ---------------------------- | ----- | --- | ------ | ---- | ----- | ---- |
| attack\_type                 | 0.052 | 3   | 0.0173 | 6.45 | 0.001 | 0.21 |
| target\_label                | 0.030 | 9   | 0.0033 | 3.02 | 0.005 | 0.18 |
| attack\_type × target\_label | 0.090 | 27  | 0.0033 | 2.55 | 0.012 | 0.25 |
| Error                        | 0.150 | 240 | ...    |      |       |      |
#-------------------------------------

SS = Sum of Squares
DF = Degrees of Freedom
MS = Mean Squares
F = F-statistic
p-unc = uncorrected p-value
np2 = partial eta-squared (effect size)
'''

#Perform Pairwise T-tests with Bonferroni correction
pg.pairwise_tests(
    dv='image_quality',
    within='attack_type',       # within-subject factor
    subject='image_id',         # ID for repeated measures
    data=df,
    padjust='bonf',             # Bonferroni correction
    effsize='cohen',            # Cohen's d
    alternative='two-sided'     # Two-sided test
)
'''Need Bonferroni to correct for multiple comparisons'''
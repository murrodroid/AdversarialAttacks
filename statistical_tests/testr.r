# 1) Install and load emmeans (if not installed)
install.packages("emmeans", repos="https://cloud.r-project.org")
library(emmeans)

# 2) Change to the folder containing the CSV or use an absolute path
setwd("c:/Users/canic/OneDrive/DTU/Fagprojekt/AdversarialAttacks/results")

# Import data
df <- read.csv("generation_metadata.csv")

# Now fit your model
anova_model <- aov(psnr_score ~ attack, data = df)
emm <- emmeans(anova_model, specs = ~ attack)
tukey_result <- pairs(emm, adjust = "tukey")
print(tukey_result)

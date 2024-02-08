library(dplyr)
library(lme4)
library(texreg)

# Save to use in next analysis
path_data_files = "/Users/garci061/pCloud Drive/projects/2018_rumor/info_transmission_networks/data/data_processing/"

### SIMILARITY WITH THE ORIGINAL STORY
## Read data
results <- read_delim(paste0(path_data_files,"analysis_similarity_transformers.csv"), show_col_types = FALSE)  |> 
  select(layer_n, rep, condition, sim_original_story) 


# No RE (AIC -1627)
formula3 = sim_original_story ~  condition:layer_n + layer_n
mod3 <- glm(formula3, data = results)
mod3$aic
#conditionNetwork:layer_n  0.017329   0.004387   3.951 8.45e-05 ***
confint(mod3, method="Wald") #0.01260623  0.02014244

# Random intercepts (AIC -1812)
formula2 = sim_original_story ~  condition:layer_n + layer_n + (1 + rep)
mod2 <- glm(formula2, data = results)
mod2$aic
confint(mod2, method="Wald") #0.009915929  0.0247422450


### SIMILARITY WITH THE ORIGINAL STORY
results <- read_delim(paste0(path_data_files,"analysis_similarity_ind_transformers.csv"), show_col_types = FALSE)  |>
  select(layer_n, k, condition, story_merged) |> 
  rename(similarity_within_story = story_merged)


# No RE (AIC -18552)
formula3b = similarity_within_story ~  condition:layer_n + layer_n
mod3b <- glm(formula3b, data = results)
mod3b$aic
#conditionNetwork:layer_n  0.017329   0.004387   3.951 8.45e-05 ***
ci <- confint(mod3b, method="Wald") #0.02926061  0.03184255
ci["conditionNetwork:layer_n", ]

# Robustneess (shouldn't be differents (AIC -18563)
formula2b = similarity_within_story ~  condition:layer_n + layer_n + (1 + k)
mod2b <- glm(formula2b, data = results)
mod2b$aic
ci <- confint(mod2b, method="Wald") #0.02789901 0.03078316 
ci["conditionNetwork:layer_n", ]


texreg(list(mod3, mod2, mod3b), digits = 3)

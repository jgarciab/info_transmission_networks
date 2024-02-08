library(dplyr)
library(lme4)
library(texreg)

# Save to use in next analysis
path_data_files = "/Users/garci061/pCloud Drive/projects/2018_rumor/info_transmission_networks/data/"
path_figures = ".../results"



results <- read_delim(paste0(path_data_files,"data_processing/transmissions_word_level.csv"), delim = ",", show_col_types = FALSE) 

## Models of transmission

# No RE (AIC 39750.21)
formula3 = transmitted ~  number_stories_observed  + condition + log(number_words_read) 
mod3 <- glm(formula3, data = results, family="binomial")
mod3$aic

# Random intercepts (AIC 33739.82)
formula2 = transmitted ~  number_stories_observed  + condition + log(number_words_read)  + (1 | word)  
mod2 <- glmer(formula2, data = results, family="binomial")
summary(mod2)$AICtab


# RAndom slopes and intercepts (AIC = 33592.32)
formula1 = transmitted ~  number_stories_observed   + condition + log(number_words_read)  + (1 + number_stories_observed|word)  
mod1 <- glmer(formula1, data = results, family="binomial")
summary(mod1)
summary(mod1)$AICtab
confint(mod1, method = "Wald")


# Create a LaTeX table for the model
texreg(list(mod1, mod2, mod3))



# Calculate baseline probabilities
predict_d <- results |> 
  mutate(pred = predict(mod1, type="response"))
  select(word) |> 
  distinct() |> 
  mutate(number_stories_observed = 1,
         number_words_read = 265,
         condition = "Chain",
         number_observed = 1)

predict_d <- predict_d |> 
  mutate(baseline_prob_3_word_265 = predict(mod1, newdata=predict_d, type="response"),
         baseline_prob_3_word_100 = predict(mod1, newdata=mutate(predict_d, number_words_read=100), type="response"),
         baseline_prob_3_word_30 = predict(mod1, newdata=mutate(predict_d, number_words_read=30), type="response"),
         baseline_prob_2_word_100 = predict(mod1, newdata=mutate(predict_d, number_stories_observed=2/3, number_words_read=3*100, condition="Network"), type="response"),
         baseline_prob_1_word_100 = predict(mod1, newdata=mutate(predict_d, number_stories_observed=1/3, number_words_read=3*100, condition="Network"), type="response"),
         baseline_prob_2_word_30 = predict(mod1, newdata=mutate(predict_d, number_stories_observed=2/3, number_words_read=3*30, condition="Network"), type="response"),
         baseline_prob_1_word_30 = predict(mod1, newdata=mutate(predict_d, number_stories_observed=1/3, number_words_read=3*30, condition="Network"), type="response")) |>   
  arrange(desc(baseline_prob_3_word_265))
  

write_delim(rownames_to_column(ranef(mod1)[[1]], var = "word"), paste0(path_data_files,"/data_final/word_random_effects.csv"), delim="\t")
write_delim(predict_d, paste0(path_data_files,"data_final/word_baseline_probs.csv"), delim="\t")



## Plot the model
#library(easystats)
#library(ggplot2)
#model_dashboard(mod1)
#plot_predictions(mod1, condition = c("number_stories_observed"), type="response", rug=TRUE) 


library(dplyr)
library(lme4)
library(texreg)
library(readr)

# Save to use in next analysis
path_data_files = "/Users/garci061/pCloud Drive/projects/2018_rumor/info_transmission_networks/data/"
path_figures = ".../results"

fear_words <- read_delim(paste0(path_data_files,"data_processing/NRC_emotion_fear_words.csv"), delim = ",", show_col_types = FALSE, col_names = FALSE)#, skip = 2) 
names(fear_words) <- c("word", "count")
fear_words$fear <- 1
fear_words

results <- read_delim(paste0(path_data_files,"data_processing/transmissions_word_level.csv"), delim = ",", show_col_types = FALSE) 
results$number_stories_observed <- as.factor(results$number_stories_observed)
results <- left_join(results, fear_words)
results$fear <- tidyr::replace_na(results$fear, 0)
## Models of transmission


# No RE (AIC 39750.21)
formula3 = transmitted ~  number_stories_observed  + condition + log(number_words_read) 
mod3 <- glm(formula3, data = results, family="binomial")
mod3$aic

# Random intercepts 
formula2_c = transmitted ~   log(number_words_read)  + (1 | word) 
mod2_c <- glmer(formula2_c, data = results |> filter(condition == "Chain", layer_n > 1), family="binomial")
summary(mod2_c)

# Random intercepts 
formula2_n = transmitted ~  I(number_stories_observed) + log(number_words_read)  + (1 | word) - 1
mod2_n <- glmer(formula2_n, data = results |> filter(condition == "Network", layer_n > 1), family="binomial")
summary(mod2_n)

# Random intercepts 
formula2_b = transmitted ~ number_stories_observed*condition + log(number_words_read)  + (1 | word) - 1
mod2_b <- glmer(formula2_b, data = results |> filter(layer_n > 1), family="binomial")
summary(mod2_b)

exp(fixef(mod2_b))

# Random intercepts 
formula2_b_fear = transmitted ~  fear + I(number_stories_observed)*condition + log(number_words_read)*condition  + (1 | word) - 1
mod2_b_fear <- glmer(formula2_b_fear, data = results |> filter(layer_n > 1), family="binomial", control = glmerControl(optimizer = "Nelder_Mead"))
summary(mod2_b_fear)
summary(mod2_b_fear)$AICtab
summary(mod2_b)$AICtab

# Random intercepts 
formula2_b_fear = transmitted ~  fear + log(number_words_read)  + (1 | word) 
mod2_b_fear <- glmer(formula2_b_fear, data = results |> filter(layer_n == 1), family="binomial", control = glmerControl(optimizer = "Nelder_Mead"))
summary(mod2_b_fear)

# Random intercepts (AIC 33739.82)
formula2 = transmitted ~  number_stories_observed  + condition + log(number_words_read)  + (1 | word)  
mod2 <- glmer(formula2, data = results, family="binomial")
summary(mod2)$AICtab

exp(fixef(mod2_b))

results |> group_by(condition, layer_n) |> summarize(mean(number_words_read))

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


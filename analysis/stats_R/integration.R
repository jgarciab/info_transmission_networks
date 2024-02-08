library(tidyverse)
library(lme4)
library(texreg)
#library(marginaleffects)
#library(stargazer)
#library(broom)

folder_figures = "~/Dropbox/Apps/Overleaf/2022_rumor_final/Figures/"


results <- read_delim("all_persons.csv", delim = "\t", show_col_types = FALSE) 

# %>% 
#   mutate(log_nwr = log10(number_words_read)) %>% 
#   filter(layer_n > 1)


# No RE
formula3 = transmitted ~  number_stories_observed  + condition + log(number_words_read) 
mod3 <- glm(formula3, data = results, family="binomial")
summary(mod3)$AIC

# Random intercepts (AIC 34174.66)
formula2 = transmitted ~  number_stories_observed  + condition + log(number_words_read)  + (1 | word)  
mod2 <- glmer(formula2, data = results, family="binomial")
summary(mod2)$AICtab


# RAndom slopes and intercepts (AIC = 34033.09)
formula1 = transmitted ~  number_stories_observed   + condition + log(number_words_read)  + (1 + number_stories_observed|word)  
mod1 <- glmer(formula1, data = results, family="binomial")
summary(mod1)
summary(mod1)$AICtab
confint(mod1, method = "Wald")

# RAndom slopes and intercepts, including number_observed (AIC = 33635.36)
formula1_alt = transmitted ~  number_stories_observed  + log(number_observed) + condition + log(number_words_read)  + (1 + number_stories_observed|word)  
mod1_alt <- glmer(formula1_alt, data = results, family="binomial")
summary(mod1_alt)
summary(mod1_alt)$AICtab



# RAndom slopes and intercepts (AIC = 33705.33)
#formula1 = transmitted ~  number_observed + number_stories_observed   + condition + log(number_words_read)  + (1 + number_stories_observed|word)  
#mod1 <- glmer(formula1, data = results, family="binomial")
#summary(mod1)
#summary(mod1)$AICtab




results <- results |> mutate(pred = predict(mod1, type="response"))





# Create a LaTeX table for the model

texreg(list(mod1, mod2, mod3))

predict_d <- results |> 
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
  

write_delim(rownames_to_column(ranef(mod1)[[1]], var = "word"), "word_random_effects.csv", delim="\t")
write_delim(predict_d, "word_baseline_probs.csv", delim="\t")





re <- tibble(x = rownames(getME(mod1, "Zt")), y = as.vector(getME(mod1, "b"))) %>% 
  arrange(y)


library(easystats)
library(ggplot2)
model_dashboard(mod1)
plot_predictions(mod, condition = c("number_stories_observed"), type="response", rug=TRUE) 

#renv::init()

#install.packages(c("tidyverse","survival","ranger","ggfortify","tidytext","tm","SnowballC", "modelsummary))
#renv::snapshot()

#Tutorials
##https://www.emilyzabor.com/tutorials/survival_analysis_in_r_tutorial.html
##https://rviews.rstudio.com/2017/09/25/survival-analysis-with-r/

#renv::restore()

library(tidyverse)
library(survival)
library(ranger)
library(ggfortify)
library(tidytext)
library(tm)
library(SnowballC)
#library(modelsummary)

folder_figures = "~/Dropbox/Apps/Overleaf/2022_rumor_final/Figures/"
path_data_files = "../../cleaned_data/"



chain <- 
  read_delim(paste0(path_data_files, "cleaned_chain_exp.csv"),
             delim="\t", 
             col_select = c("layer_n", "rep","story_merged")) %>% 
  mutate(condition = "chain")

network <- 
  read_delim(paste0(path_data_files, "cleaned_network_exp.csv"),
             delim="\t", 
             col_select = c("layer_n", "rep","story_merged")) %>% 
  mutate(condition = "network")

# MErge
data <- bind_rows(network,chain)


stopWords <- stopwords("en")

original_story = "Through history, most people didn't die of cancer or heart disease, the lifestyle diseases that are common in the Western world today. This is mostly because people didn't live long enough to develop them. They died of injuries -- being gored by an ox, shot on a battlefield, crushed in one of the new factories of the Industrial Revolution -- and most of the time from infection, which followed those injuries.  That changed when antibiotics were discovered. In 1928, Alexander Fleming discovered penicillin, a drug still used today to fight bacterial infections. Suddenly, infections that had been a death sentence became remedied within days. During World War II, this drug treated pneumonia and sepsis, and has been estimated to have saved between 12-15% of Allied Forces lives. We have been living inside the golden epoch of the miracle drugs, and now, we are coming to an end of it.  People are dying of infections again because of a phenomenon called antibiotic resistance, or popularly referred to as “superbugs”. Bacteria compete against each other for resources, for food, by manufacturing lethal compounds that they direct against each other. When we first made antibiotics, we took those compounds into the lab and made our own versions of them, and bacteria responded to our attack the way they always had.  For 70 years, we played a game of leapfrog -- our drug and their resistance, and then another drug, and then resistance again -- and now the game is ending. Bacteria develop resistance so quickly that pharmaceutical companies have decided making antibiotics is not in their best interest, so there are infections moving across the world for which, out of the more than 100 antibiotics available on the market, two drugs might work with side effects, or one drug, or none.  It would be natural to hope that these infections are extraordinary cases, but in fact, in the United States and Europe, 50 thousand people a year die of infections which no drugs can help. A project chartered by the British government known as the Review on Antimicrobial Resistance estimates that the worldwide toll right now is 700 thousand deaths a year. Also, if we can't get this under control by 2050, the worldwide toll will be 10 million deaths a year (more than the current population of New York City).  The scale of antibiotic resistance seems overwhelming, but if you've ever bought a fluorescent light bulb because you were concerned about climate change, you already know what it feels like to take a tiny step to address an overwhelming problem. We could take those kinds of steps for antibiotic use too. We could forgo giving an antibiotic for our kids’ ear infection, if we're not sure it's the right one. And we could promise each other to never again to buy chicken or shrimp or fruit raised with routine antibiotic use. If we did those things, we could slow down the arrival of the post-antibiotic world."
original_story <- as_tibble(list(story = original_story)) %>% 
  unnest_tokens(word, story) %>% 
  mutate(word = str_to_lower(word)) %>% 
  distinct(word, .keep_all = TRUE)  %>% 
  filter(!(word %in% stopWords))

chain_original_story <-
  original_story %>% 
  crossing(rep = unique(chain$rep), condition = "chain") 

network_original_story <-
  original_story %>% 
  crossing(rep = unique(network$rep), condition = "network") 

# MErge
data_original_story <- bind_rows(chain_original_story,network_original_story)




#network %>% filter(layer_n == 5, rep == 0) %>% .$story_merged

# Tokens
data_word <- data %>% 
  unnest_tokens(word, story_merged) %>% 
  mutate(word = str_to_lower(word)) %>% 
  filter(!(word %in% stopWords)) %>% 
  right_join(data_original_story, by = c("word","rep","condition")) %>% 
  mutate(word = wordStem(word),
         word = ifelse(word == "bacteria", "bacteri", word),
         word = ifelse(word == "die", "death", word),
         word = ifelse(word == "di", "death", word),
         layer_n = 1+replace_na(layer_n, 0)) %>% 
  distinct(rep, condition, word, .keep_all = TRUE) 

tail(data_word)


# last time
data_word  <- data_word %>% 
  group_by(rep, condition, word) %>% 
  filter(layer_n == max(layer_n)) %>% 
  arrange(desc(layer_n)) %>% 
  mutate(status = ifelse(layer_n == 5, 0, 1), #Add censoring
         important = word %in% c("antibiot","infect","resist","drug","superbug","bacteri","death","diseas"))


a <- data_word %>% 
  group_by(word, condition) %>% 
  summarise(ml = mean(layer_n), .groups = "keep") %>% 
  filter(ml > 1.5) %>%
  pivot_wider(id_cols= "word", names_from = "condition", values_from = "ml") %>% 
  arrange(desc(network))
print(a, n=100)

data_word
# Survival analysis
km <- with(data_word, Surv(layer_n, status))

km_fit <- survfit(Surv(layer_n, status) ~ 1, data=data_word)
summary(km_fit, times = c(0,1,2,3,4,5))
autoplot(km_fit)


# Modeling survival based on condition
km_trt_fit <- survfit(Surv(layer_n, status) ~  condition, data=data_word)
summary <- summary(km_trt_fit, times = c(0,1,2,3,4,5))
summary

#modelsummary(km_trt_fit)

# Plotting survival curve
autoplot(km_trt_fit) +
  scale_y_continuous(trans='log10') +
  xlim(0,5) +
  theme_minimal() +
  labs(x = "Generation", y = "p(Survival)") + 
  theme(aspect.ratio=0.7,
        panel.grid.major.x = element_blank() ,
        panel.grid.minor.x = element_blank() ,
        panel.grid.major.y = element_line( size=.1, color="black" ) ) +
  scale_color_manual(values=c("#e6af2e","#3d348b")) + 
  scale_fill_manual(values=c("#e6af2e","#3d348b"))  

ggsave(paste0(folder_figures, "survival.pdf"))

# Model of survival based on condition
cox <- coxph(Surv(layer_n, status) ~ condition*important, data = data_word)
summary(cox)



cox <- coxph(Surv(layer_n, status) ~ condition + condition:important + word, data = data_word)
a <- summary(cox)
a
b <- data.frame(a$coefficients)
filter(b, `Pr...z..` < 0.05,
       coef < 0)

survdiff(formula = Surv(layer_n, status) ~ condition, data = data_word)

# TODO:
## Re-introduction of words
## Why are important words dropped in the layer 2 but kept in layer 1?


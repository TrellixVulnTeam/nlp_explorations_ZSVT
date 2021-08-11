
library(here)
library(dplyr)
library(tidytext)
library(topicmodels)
library(stopwords)
library(ggplot2)

articles <- readr::read_tsv(here("data/raw/swissdox/210809_request/Angst1.tsv"))

articles

table(articles$medium_name)

range(articles$pubtime)

table(is.na(articles$rubric))

# rubriken
table(articles$rubric)[order(table(articles$rubric), decreasing = TRUE)]

# get zürich subset for the moment (only nzz and ta because no rubric for 20min friday)

articles_zh <- articles %>% dplyr::filter(grepl("Zürich", rubric, ignore.case = TRUE))

# what does an article look like?

articles_zh %>% filter(medium_code == "TA") %>% slice_sample(n = 1) %>% select(content) %>% .$content
articles_zh %>% filter(medium_code == "NZZ") %>% slice_sample(n = 1) %>% select(content) %>% .$content

# some tidy texting -> get tokens

articles_zh %>% filter(medium_code == "TA") %>% 
  slice_sample(n = 1) %>% select(id,content) %>% 
  tidytext::unnest_tokens(tbl = ., output = "tokens", input = "content", )

# let's dive in with a very basic topic model

# to keep things lightweight, look at articles using nachhaltigkeit somehow,
# then tokenize, remove stopwords

articles_zh_tokens <- 
  articles_zh %>% filter(medium_code %in% c("TA","NZZ")) %>% 
  filter(grepl("nachhalt", content)) %>% 
  select(content, "id") %>%
  tidytext::unnest_tokens(tbl = ., output = "word", input = "content") %>% 
  anti_join(get_stopwords(language = "de"), by = c("word" = "word")) %>% 
  filter(!(word %in% c("p","tx","au","zt","lg","ld"))) %>%  # remove html tags
  filter(!(grepl("^[0-9]*$",word))) %>% # remove numbers
  filter(!(word %in% c("dass", "zürich", "mehr"))) # remove some custom words

# frequent words
articles_zh_tokens %>% count(word, sort = TRUE)

length(unique(articles_zh_tokens$id))

# add count of words per article

articles_zh_tokens <-
  articles_zh_tokens %>% count(word,id, sort = TRUE)

# turn into term-document matrix

articles_zh_dtm <-
  articles_zh_tokens %>% cast_dtm(term = "word", document = "id", value = "n")

articles_zh_dtm
head(articles_zh_dtm$dimnames)
tm::inspect(articles_zh_dtm)

# topic model

zh_lda <- LDA(articles_zh_dtm, k = 10, control = list(seed = 1234))

zh_topics <- tidy(zh_lda, matrix = "beta")
zh_topics

zh_top_terms <- zh_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

zh_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

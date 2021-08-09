
library(here)
library(dplyr)
library(tidytext)

articles <- readr::read_tsv(here("data/raw/swissdox/210809_request/Angst1.tsv"))

articles

table(articles$medium_name)

range(articles$pubtime)

table(is.na(articles$rubric))

# rubriken
table(articles$rubric)[order(table(articles$rubric), decreasing = TRUE)]

# get zürich subset for the moment (only nzz and ta because no rubric for 20min friday)

articles_zh <- articles %>% dplyr::filter(grepl("Zürich", rubric, ignore.case = TRUE))
Was
# what does an article look like?

articles_zh %>% filter(medium_code == "TA") %>% slice_sample(n = 1) %>% select(content) %>% .$content
articles_zh %>% filter(medium_code == "NZZ") %>% slice_sample(n = 1) %>% select(content) %>% .$content

# some tidy texting -> get tokens
articles_zh %>% filter(medium_code == "TA") %>% 
  slice_sample(n = 1) %>% select(content) %>% 
  tidytext::unnest_tokens(tbl = ., output = "tokens", input = "content")

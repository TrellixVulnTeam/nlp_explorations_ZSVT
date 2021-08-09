
library(reticulate)

# reticulate::miniconda_update()

conda_create("nlp_test", packages = c("numpy", "stanza"))

use_condaenv("nlp_test")

stanza <- import("stanza")

nlp_ger <- stanza$Pipeline('de')

doc <- nlp_ger("Wie viele fröhliche Murmeltiere leben in den Bergen? 
               Wie viele davon heissen Max oder Moritz?")

doc

doc$text
doc$get("lemma")
doc$get("feats")

df <- data.frame(text = doc$get("text"),
                 token_lemma = doc$get("lemma"),
                 type = doc$get("upos"))

df

library(readr)
library(ggrepel)
library(writexl)
library(tidyverse)

# leggo csv
df <- read_csv("/home/cbernasconi/et/experiments/cross_dataset/results/thesis_csvs/2figer.csv",
                   col_types = cols(...1 = col_skip(), 
                                    precision = col_number(), 
                                    recall = col_number(), 
                                    `f1-score` = col_number()
                                    )
                )
#write_xlsx(df,"../output/few_nerd_spec_hs/spec_fully.xlsx")

filtered_df = df %>% filter(precision>0, recall>0)

# creo grafico
filtered_df %>% ggplot(aes(x=precision, y=recall, color=sourcedataset)) + geom_point() +
  geom_text_repel(label=filtered_df$Type, max.overlaps=60) + theme_light() +
  labs(x="\nPrecison", y="Recall\n") +
  theme(legend.position = 'bottom',
        panel.grid.minor = element_blank())
ggsave('/home/cbernasconi/et/experiments/cross_dataset/results/ova_zs.png', height = 9, width = 14)
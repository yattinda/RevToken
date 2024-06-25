library(tidyverse)
library(ggplot2)
library(gridExtra)
library("ggpubr")
library(patchwork)

raw = read.csv(paste0("motivation.csv"),stringsAsFactors = F)
raw$dataset <- factor(raw$dataset, levels = c('OpenstackNova','OpenstackIronic', 'QtBase'), ordered = TRUE)


initial_feedback_time <- raw %>% select(c("dataset","initial_feedback_time"))
percentage_time_cost <- raw %>% select(c("dataset","percentage_time_cost"))
code_size <- raw %>% select(c("dataset","changed_line"))
time_size_relation <-  raw %>% select(c("dataset","initial_feedback_time","changed_line"))

nova_size <- code_size %>% filter(dataset=="OpenstackNova")
ironic_size <- code_size %>% filter(dataset=="OpenstackIronic")
base_size <- code_size %>% filter(dataset=="QtBase")

nova_per <- percentage_time_cost %>% filter(dataset=="OpenstackNova")
ironic_per <- percentage_time_cost %>% filter(dataset=="OpenstackIronic")
base_per <- percentage_time_cost %>% filter(dataset=="QtBase")

g1 <- percentage_time_cost %>% ggplot() + geom_boxplot(aes(x=dataset, y = percentage_time_cost, fill=dataset),show.legend = FALSE)  + scale_y_continuous(breaks = 0:5*0.2, labels = scales::percent)+ scale_fill_brewer(palette = "Blues", direction=-1) + theme_bw() + labs(x="")+ labs(y="The Proportion of the Waiting Hours") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) 

g2 <- initial_feedback_time %>% ggplot() + geom_boxplot(aes(x=dataset, y = initial_feedback_time, fill=dataset), show.legend = FALSE) + coord_cartesian(ylim= c(0,500))+ scale_fill_brewer(palette = "Blues", direction=-1) + theme_bw() + labs(x="") + labs(y="Waiting Hours to Receive the First Comment") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) 

g3 <- code_size %>% ggplot() + geom_boxplot(aes(x=dataset, y = changed_line, fill=dataset), show.legend = FALSE) + coord_cartesian(ylim= c(0,500))+ scale_fill_brewer(palette = "Blues", direction=-1) + theme_bw() + labs(x="") + labs(y="Patch Size") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) 

nova_rel <- time_size_relation %>% filter(dataset=="OpenstackNova")
ironic_rel <- time_size_relation %>% filter(dataset=="OpenstackIronic")
base_rel <- time_size_relation %>% filter(dataset=="QtBase")

nova_rel <- nova_rel %>% mutate(isLarge = case_when(changed_line >= median(nova_size$changed_line) ~ "Large Patches", changed_line < median(nova_size$changed_line) ~ "Small Patches"))
ironic_rel <- ironic_rel %>% mutate(isLarge = case_when(changed_line >= median(ironic_size$changed_line) ~ "Large Patches", changed_line < median(ironic_size$changed_line) ~ "Small Patches"))
base_rel <- base_rel %>% mutate(isLarge = case_when(changed_line >= median(base_size$changed_line) ~ "Large Patches", changed_line < median(base_size$changed_line) ~ "Small Patches"))

nova_rel_l <- nova_rel %>% filter(isLarge=="Large Patches") %>% ungroup() %>% select(c(initial_feedback_time))
nova_rel_s <- nova_rel %>% filter(isLarge=="Small Patches") %>% ungroup() %>% select(c(initial_feedback_time))
apply(nova_rel_l,2,median)
apply(nova_rel_s,2,median)


ironic_rel_l <- ironic_rel %>% filter(isLarge=="Large Patches") %>% ungroup() %>% select(c(initial_feedback_time))
ironic_rel_s <- ironic_rel %>% filter(isLarge=="Small Patches") %>% ungroup() %>% select(c(initial_feedback_time))
apply(ironic_rel_l,2,median)
apply(ironic_rel_s,2,median)


base_rel_l <- base_rel %>% filter(isLarge=="Large Patches") %>% ungroup() %>% select(c(initial_feedback_time))
base_rel_s <- base_rel %>% filter(isLarge=="Small Patches") %>% ungroup() %>% select(c(initial_feedback_time))
apply(base_rel_l,2,median)
apply(base_rel_s,2,median)

#median of patch size
median(ironic_rel$changed_line)
median(nova_rel$changed_line)
median(base_rel$changed_line)

g4_1 <- nova_rel %>% ggplot() + geom_boxplot(aes(x=isLarge, y = initial_feedback_time, fill=isLarge), show.legend = FALSE) + coord_cartesian(ylim= c(0,600))+ scale_fill_brewer(palette = "Blues", direction=-1) + theme_bw() + labs(x="") + labs(y="Waiting Hours") + facet_grid(~dataset) + facet_wrap(~dataset,strip.position = "right") + scale_x_discrete(limits=c('Large Patches','Small Patches'))
g4_2 <- ironic_rel %>% ggplot() + geom_boxplot(aes(x=isLarge, y = initial_feedback_time, fill=isLarge), show.legend = FALSE) + coord_cartesian(ylim= c(0,600))+ scale_fill_brewer(palette = "Blues", direction=-1) + theme_bw() + labs(x="") + labs(y="Waiting Hours") + facet_grid(~dataset) + facet_wrap(~dataset,strip.position = "right") + scale_x_discrete(limits=c('Large Patches','Small Patches'))
g4_3 <- base_rel %>% ggplot() + geom_boxplot(aes(x=isLarge, y = initial_feedback_time, fill=isLarge), show.legend = FALSE) + coord_cartesian(ylim= c(0,150))+ scale_fill_brewer(palette = "Blues", direction=-1) + theme_bw() + labs(x="") + labs(y="Waiting Hours")+ facet_grid(~dataset) + facet_wrap(~dataset,strip.position = "right") + scale_x_discrete(limits=c('Large Patches','Small Patches'))


nova_time <- initial_feedback_time %>% filter(dataset=="OpenstackNova")
ironic_time <- initial_feedback_time %>% filter(dataset=="OpenstackIronic")
base_time <- initial_feedback_time %>% filter(dataset=="QtBase")

nova_size <- code_size %>% filter(dataset=="OpenstackNova")
ironic_size <- code_size %>% filter(dataset=="OpenstackIronic")
base_size <- code_size %>% filter(dataset=="QtBase")

# The Mann-Whitney test
wilcox.test(nova_rel$initial_feedback_time~nova_rel$isLarge)
wilcox.test(ironic_rel$initial_feedback_time~ironic_rel$isLarge)
wilcox.test(base_rel$initial_feedback_time~base_rel$isLarge)

pdf("./figures/motivation.pdf", width=6, height = 6)
g2+g1 | g4_1/g4_2/g4_3
dev.off()



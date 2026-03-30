source("functions.R")

# Results (without normalization in score decomposition)
tab_c10 = run_exp("cifar10")
tab_c10_focal = run_exp("focal_cifar10",c("resnet50", "resnet110","wide_resnet","densenet121"))
tab_c10_focal$dataset = "cifar10"
tab_c10_focal$model = paste0("focal_",tab_c10_focal$model)
tab_c100 = run_exp("cifar100")
tab_c100_focal = run_exp("focal_cifar100",c("resnet50", "resnet110","wide_resnet","densenet121"))

tab_c10_focal$dataset = "cifar10"
tab_c10_focal$model = paste0("focal_",tab_c10_focal$model)
tab_c100_focal$dataset = "cifar100"
tab_c100_focal$model = paste0("focal_",tab_c100_focal$model)

tab = rbind(tab_c10,tab_c10_focal,tab_c100,tab_c100_focal)

# Results (with normalization in score decomposition)
tab_c10 = run_exp("cifar10",normalize_sd = TRUE)
tab_c10_focal = run_exp("focal_cifar10",c("resnet50", "resnet110","wide_resnet","densenet121"),normalize_sd = TRUE)
tab_c100 = run_exp("cifar100",normalize_sd = TRUE)
tab_c100_focal = run_exp("focal_cifar100",c("resnet50", "resnet110","wide_resnet","densenet121"),normalize_sd = TRUE)

tab_c10_focal$dataset = "cifar10"
tab_c10_focal$model = paste0("focal_",tab_c10_focal$model)
tab_c100_focal$dataset = "cifar100"
tab_c100_focal$model = paste0("focal_",tab_c100_focal$model)

tab_norm = rbind(tab_c10,tab_c10_focal,tab_c100,tab_c100_focal)

# Save/Load score decompositions
# save(tab,tab_norm,file = "tab.RData")
# load("tab.RData")

tab[,4:6] = 100*tab[,4:6]
tab = cbind(tab, unc = tab$brier - tab$mcb + tab$dsc)

tab_norm[,4:6] = 100*tab_norm[,4:6]
tab_norm = cbind(tab_norm, unc = tab_norm$brier - tab_norm$mcb + tab_norm$dsc)

# Tables: Aggregate score decompositions
# Table 1
# Average class-wise score decompositions
tab_mean = aggregate(tab[,-(1:3)],by = tab[,c(3,1)],FUN = mean)[c(1,2,5,3,6,4,7,8,9,12,10,13,11,14),c(2,1,3,4,5,7,6)]
print.xtable(xtable(tab_mean,digits = 3),include.rownames = FALSE)

# Table 2
# Average normalized score decompositions
tab_norm_mean = aggregate(tab_norm[,-(1:3)],by = tab_norm[,c(3,1)],FUN = mean)[c(1,2,5,3,6,4,7,8,9,12,10,13,11,14),c(2,1,3,4,5,7,6)]
print.xtable(xtable(tab_norm_mean[,-1],digits = 3),include.rownames = FALSE)

# Figures: DSC-MCB plots
# Figure 2 (left)
# Class-wise score decomposition for CIFAR-10
pdf("figs/Fig2_left.pdf",width = 6,height = 4)
plot_mcbdsc(tab)
dev.off()

# Figure 3 (left)
# Score decomposition with normalization for CIFAR-10
pdf("figs/Fig3_left.pdf",width = 6,height = 4)
plot_mcbdsc(tab_norm,leg_pos = "top")
dev.off()

# Figure 2 (right)
# Class-wise score decomposition for CIFAR-100
pdf("figs/Fig2_right.pdf",width = 6,height = 4)
plot_mcbdsc(tab,datasets = "cifar100",inc_iso = 0.1)
dev.off()

# Figure 3 (right)
# Score decomposition with normalization for CIFAR-100
pdf("figs/Fig3_right.pdf",width = 6,height = 4)
plot_mcbdsc(tab_norm,datasets = "cifar100",inc_iso = 0.1,leg_pos = "top")
dev.off()




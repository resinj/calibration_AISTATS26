options(scipen = 9999)

#### Packages ##################################################################
library(reticulate)
np = import("numpy")

library(xtable)
library(RColorBrewer)

#### Functions #################################################################
# Class-wise recalibration (without normalization)
# Class-wise histogram binning
cw_hb = function(probs,obs, nbins = 15, equal_mass = FALSE){
  n = length(obs)
  m = ncol(probs)
  cal_funs = list()
  for(i in 1:m){
    y = as.numeric(obs == i)
    p = probs[,i]
    if(equal_mass){
      hb_res = aggregate(y,list(pmax(ceiling((rank(p)-1)/n*nbins),1)),mean)
      hb_min = aggregate(p,list(pmax(ceiling((rank(p)-1)/n*nbins),1)),min)
      cal_funs[[i]] = stepfun(x = c(0,hb_min$x[-1]),y = c(0,hb_res$x))
    }
    else{
      hb_res_ag = aggregate(y,list(pmax(ceiling(p*nbins),1)),mean)
      hb_res = data.frame(Group.1 = 1:nbins,x = rep(1/m,nbins))
      hb_res$x[hb_res_ag$Group.1] = hb_res_ag$x
      cal_funs[[i]] = stepfun(x = (hb_res$Group.1-1)/nbins,y = c(0,hb_res$x))
    }
  }
  cal_funs
}

# Class-wise isotonic regression
cw_ir = function(probs,obs, smooth = FALSE){
  n = length(obs)
  m = ncol(probs)
  cal_funs = list()
  for(i in 1:m){
    y = as.numeric(obs == i)
    p = probs[,i]
    if(!smooth) cal_funs[[i]] = as.stepfun(isoreg(p,y))
    else{
    # smooth isoreg
    ir = isoreg(p,y)
    lbin = sapply(unique(ir$yf), \(yf) which(ir$yf == yf)[1])
    rbin = c(lbin[-1]-1,n)
    x = sort(p)
    bin_med = floor((rbin + lbin)/2) # use the lowest median instead of textbook definition for simplicity
    cal_funs[[i]] = approxfun(x = c(0,x[bin_med],1),y = c(0,ir$yf[bin_med],1))
    }
  }
  cal_funs
}

# Metrics
brier = function(probs,obs, normalize_pred = FALSE, ...){
  n = length(obs)
  m = ncol(probs)
  
  if(normalize_pred) probs = probs/rowSums(probs)
  
  brier = 0
  for(i in 1:m){
    y = as.numeric(obs == i)
    p = probs[,i]
    brier = brier + sum((p-y)^2)/n
  }
  brier/m
}

mcb = function(probs,obs, normalize_pred = FALSE, normalize_sd = FALSE, ...){
  n = length(obs)
  m = ncol(probs)
  
  if(normalize_pred) probs = probs/rowSums(probs)

  cal_fun = cw_ir(probs,obs)
  probs_recal = matrix(nrow = n,ncol = m)
  for(i in 1:m){
    probs_recal[,i] = cal_fun[[i]](probs[,i])
  }
  
  if(normalize_sd) probs_recal = probs_recal/rowSums(probs_recal)
  
  brier(probs,obs) - brier(probs_recal,obs)
}

dsc = function(probs,obs, normalize_pred = FALSE, normalize_sd = FALSE, ...){
  n = length(obs)
  m = ncol(probs)
  
  if(normalize_pred) probs = probs/rowSums(probs)
  
  cal_fun = cw_ir(probs,obs)
  probs_recal = matrix(nrow = n,ncol = m)
  probs_marg = c()
  
  for(i in 1:m){
    probs_recal[,i] = cal_fun[[i]](probs[,i])
    probs_marg[i] = mean(obs == i)
  }
  
  if(normalize_sd) probs_recal = probs_recal/rowSums(probs_recal)
  
  probs_marg = matrix(rep(probs_marg,times = n),ncol = m,byrow = TRUE)
  print(paste("UNC = ",brier(probs_marg,obs)))
  
  brier(probs_marg,obs) - brier(probs_recal,obs)
}

accuracy = function(probs,obs, normalize_pred = FALSE, ...){
  n = length(obs)
  m = ncol(probs)
  
  if(normalize_pred) probs = probs/rowSums(probs)
  
  sum(apply(probs,1,which.max) == obs)/n
}


# Experiments
run_exp = function(datasets = c("cifar10","focal_cifar10","cifar100","focal_cifar100"),
                   models = c("resnet50", "resnet110","wide_resnet"),
                   recalibration = c("base", "CW-HB", "N-HB", "CW-IR", "N-IR","CW-SIR","N-SIR"),
                   metrics = c("brier","mcb","dsc","accuracy"),
                   normalize_sd = FALSE){
  
  n_data = length(datasets)
  n_models = length(models)
  n_recal = length(recalibration)
  n_rows = n_data * n_models * n_recal

  tab = data.frame(dataset = rep(datasets,each = n_models * n_recal), model = rep(models, each = n_recal), 
                   recal = recalibration, matrix(nrow = n_rows,ncol = length(metrics)))
  tab = setNames(tab,c("dataset","model","recal",metrics))

  for(dataset in datasets){
    for(model in models){
      print(paste(dataset,model))
      data_path = paste0("data/",dataset,"_",model,"/")
      
      x_calib = np$load(paste0(data_path,"val_logits.npy"))
      x_calib = exp(x_calib)/rowSums(exp(x_calib))
      
      x_test = np$load(paste0(data_path,"test_logits.npy"))
      x_test = exp(x_test)/rowSums(exp(x_test))
      
      y_calib = np$load(paste0(data_path,"val_labels.npy"))
      y_calib = y_calib + 1
      
      y_test = np$load(paste0(data_path,"test_labels.npy"))
      y_test = y_test + 1
      
      n = length(y_test)
      m = ncol(x_test)
      
      if(is.element("base",recalibration)){
        tab[tab$dataset == dataset & tab$model == model & tab$recal == "base",-(1:3)] = sapply(metrics, \(f) get(f)(x_test,y_test,normalize_sd = normalize_sd))
      }
      
      if(is.element("CW-HB",recalibration) || is.element("N-HB",recalibration)){
        cal_fun = cw_hb(x_calib,y_calib,equal_mass = TRUE)
        x_recal = matrix(nrow = n,ncol = m)
        for(i in 1:m){
          x_recal[,i] = cal_fun[[i]](x_test[,i])
        }
        if(is.element("CW-HB",recalibration)) 
          tab[tab$dataset == dataset & tab$model == model & tab$recal == "CW-HB",-(1:3)] = sapply(metrics,\(f) get(f)(x_recal,y_test,normalize = FALSE,normalize_sd = normalize_sd))
        if(is.element("N-HB",recalibration)) 
          tab[tab$dataset == dataset & tab$model == model & tab$recal == "N-HB",-(1:3)] = sapply(metrics,\(f) get(f)(x_recal,y_test,normalize = TRUE,normalize_sd = normalize_sd))
      }
      
      if(is.element("CW-IR",recalibration) || is.element("N-IR",recalibration)){
        cal_fun = cw_ir(x_calib,y_calib,smooth = FALSE)
        x_recal = matrix(nrow = n,ncol = m)
        for(i in 1:m){
          x_recal[,i] = cal_fun[[i]](x_test[,i])
        }
        if(is.element("CW-IR",recalibration)) 
          tab[tab$dataset == dataset & tab$model == model & tab$recal == "CW-IR",-(1:3)] = sapply(metrics,\(f) get(f)(x_recal,y_test,normalize = FALSE,normalize_sd = normalize_sd))
        if(is.element("N-IR",recalibration)) 
          tab[tab$dataset == dataset & tab$model == model & tab$recal == "N-IR",-(1:3)] = sapply(metrics,\(f) get(f)(x_recal,y_test,normalize = TRUE,normalize_sd = normalize_sd))
      }
      
      if(is.element("CW-SIR",recalibration) || is.element("N-SIR",recalibration)){
        cal_fun = cw_ir(x_calib,y_calib,smooth = TRUE)
        x_recal = matrix(nrow = n,ncol = m)
        for(i in 1:m){
          x_recal[,i] = cal_fun[[i]](x_test[,i])
        }
        if(is.element("CW-SIR",recalibration)) 
          tab[tab$dataset == dataset & tab$model == model & tab$recal == "CW-SIR",-(1:3)] = sapply(metrics,\(f) get(f)(x_recal,y_test,normalize = FALSE,normalize_sd = normalize_sd))
        if(is.element("N-SIR",recalibration)) 
          tab[tab$dataset == dataset & tab$model == model & tab$recal == "N-SIR",-(1:3)] = sapply(metrics,\(f) get(f)(x_recal,y_test,normalize = TRUE,normalize_sd = normalize_sd))
      }
    }
  }
  return(tab)
}

# Plotting
# MCB-DSC plot
plot_mcbdsc = function(tab, datasets = c("cifar10","focal_cifar10"), 
                       mcb_lim = NULL, dsc_lim = NULL, inc_iso = 0.2,leg_pos = "bottom"){
  tab = tab[is.element(tab$dataset,datasets),]
  if(is.null(mcb_lim)){
    mcb_lim = c(min(tab$mcb),max(tab$mcb))
    print(paste("mcb_lim =",mcb_lim))
  }
  if(is.null(dsc_lim)){
    dsc_lim = c(min(tab$dsc),max(tab$dsc))
    print(paste("dsc_lim =",dsc_lim))
  }
    
  par(mfrow = c(1,1), lwd = 2,mar = c(2.2,2.2,0.2,0),mgp = c(1.2,0.4,0))
  plot(NULL, xlim = mcb_lim,ylim = dsc_lim,xlab = "MCB",ylab = "DSC")
  
  # Isolines
  score_lim = c(min(tab$brier),max(tab$brier))
  unc = (tab$brier - tab$mcb + tab$dsc)[1]
  dsc = round(dsc_lim[1] - mcb_lim[2],1) + unc - 0.3*floor(10*unc)
  while(dsc < dsc_lim[2]){
    abline(a = dsc, b = 1,col = "lightgray")
    text(x = mcb_lim[2]*0.9, y = dsc + mcb_lim[2]*0.9, labels = round(unc - dsc,3), col = "lightgray",pos = 1)
    dsc = dsc + inc_iso
  }

  model_short = factor(ifelse(tab$model == "resnet50","R50",
                              ifelse(tab$model == "resnet110","R110",
                                     ifelse(tab$model == "wide_resnet","WRN",
                                            ifelse(tab$model == "focal_resnet50","fR50",
                                                   ifelse(tab$model == "focal_resnet110","fR110",
                                                          ifelse(tab$model == "focal_wide_resnet","fWRN","fDN")))))))
  label = paste0(model_short,"_",tab$recal)
  
  N = length(label)
  
  n_models = nlevels(model_short)
  n_recal = nlevels(tab$recal)
  
  colors = brewer.pal(n_models,"Set2")
  symb = c(8,22,24,25,22,24,25) # c(19,15,17,3:6)

  recal = factor(tab$recal)
  n_recal = nlevels(recal)
  
  legend(leg_pos,legend = c("Model",levels(model_short),"Method",levels(recal)),
         col = c("white",colors,"white",rep("black",n_recal)), 
         pt.bg = c("white",colors,"white",rep("black",(n_recal + 1)/2),rep("white",(n_recal - 1)/2)),
         pch = c(NA,rep(19,n_models),NA,symb), ncol = 2, text.width = NA,bg = "white")
  
  
  for(i in 1:N){
    col = colors[model_short[i]]
    points(tab$mcb[i],tab$dsc[i],col = col,pch = symb[recal[i]],
           bg = ifelse(is.element(recal[i],levels(recal)[1:4]), col, adjustcolor("white",alpha.f = 0)))
  }
}


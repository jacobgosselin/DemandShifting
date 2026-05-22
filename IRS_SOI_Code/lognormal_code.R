  # ***************** 0. Setups *****************
  rm(list = ls())
  
  #commandArgs picks up the variables you pass from the command line
  args <- commandArgs(trailingOnly = F)
  
  # Set the wd by passing current directory
  # setwd(args[6]) # args 6 is the path variable passed to the program.
  setwd('/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/data/raw/misc/Ma_Replication')
  program_wd=getwd()
  print(program_wd)
  
  
  inpath <-paste0(program_wd,"/output/soi/brackets/")
  outpath <-paste0(program_wd,"/output/temp/")
  
  check.packages <- function(pkg){
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    if (length(new.pkg)) 
      install.packages(new.pkg, dependencies = TRUE)
    sapply(pkg, require, character.only = TRUE)
  }
  packages<-c("data.table","plyr","tidyverse","janitor","haven","readxl",
              "stringr","stringdist","kableExtra","fedmatch","nleqslv")
  check.packages(packages)
  
  # ***************** 1. Functions *****************
  
  # The conditional mean of a lognormal (mu, sigma) variable from a to b is given by:
  # Total probability: Phi((log b - mu)/sigma) - Phi((log a - mu)/sigma)
  # integral of e^x weighed by normal pdf from log a to log b:
  # exp(mu + 1/2 sigma^2) * (Phi((log b - (mu+sigma^2))/sigma) - Phi((log a - (mu+sigma^2))/sigma))
  # so conditional mean:
  # exp(mu + 1/2 sigma^2) * (Phi((log b - (mu+sigma^2))/sigma) - Phi((log a - (mu+sigma^2))/sigma))/
  # (Phi((log b - mu)/sigma) - Phi((log a - mu)/sigma))
  ln_cond_mean = function(lower, upper = Inf, mu, sigma){
    tot_prob = pnorm((log(upper)-mu)/sigma) - pnorm((log(lower)-mu)/sigma)
    integrand = exp(mu + sigma^2/2) * 
      (pnorm((log(upper) - (mu+sigma^2))/sigma) - pnorm((log(lower) - (mu+sigma^2))/sigma))
    return(integrand/tot_prob)
  }
  
  # mu.guess + z * sigma < log(thresh_low) iff z < (log(thresh_low) - mu.guess)/sigma
  # which has probability cdf_min = pnorm((log(thresh_low) - mu.guess)/sigma)
  # qnorm(cdf_min) = (log(thresh_low) - mu.guess)/sigma
  # if cdf_min > 0.5,  we need log(thresh) > mu.guess
  # if cdf_min < 0.5, we need log(thresh) < mu.guess
  sigma_given_mu = function(mu.guess, cdf_min, thresh_low){
    sigma.guess = (log(thresh_low)-mu.guess)/qnorm(cdf_min)
    if(sigma.guess < 0){
      return(NA)
    }else{
      return(sigma.guess)
    }
  }
  
  # Main function: compute mu, sigma for each bucket
  # thresh_low: the lower threshhold (we always set thresh_high to be Infinity)
  # tail_mean: the mean of all elements above thresh_low (so no longer interval mean)
  # cdf_min: empirical CDF evaluated at thresh_low (i.e. fraction of population lower than thresh_low)
  # cdf_min has to be between 0 and 1, and cannot be 0.5!
  # [absolute_min, absolute_max]: interval at which to search for mu
  # recommend the range: [log(min_thresh) - 4, log(max_thresh) + 4]
  log_normal_fit = function(thresh_low,  tail_mean, cdf_min, 
                            absolute_min, absolute_max){
    # cdf_min is the fraction below thresh_low, which is increasing from 0 to 1 in thresh_low
    # so you have to discard the lowest bucket
    if (cdf_min > 0 & cdf_min < 1){
      to_opt = function(mu.guess){
        # for a guess of mu, find sigma that makes the cdf_min consistent with the thresh_low
        sigma.guess = sigma_given_mu(mu.guess, cdf_min, thresh_low)
        if (!is.na(sigma.guess)){
          # compute conditional mean
          my_cond_mean = ln_cond_mean(lower = thresh_low, mu = mu.guess, sigma = sigma.guess)
          loss = (tail_mean - my_cond_mean)^2
        }else{
          loss = NA
        }
        if (is.na(loss)){
          return(Inf)
        }else{
          return(loss)
        }
      }
      
      if (cdf_min > 0.5){
        # then you have to search up to log(thresh_low)
        mean.guess = optimize(to_opt, interval = c(absolute_min, log(thresh_low))) 
      }else if (cdf_min < 0.5){
        # then you have to search from  log(thresh_low)
        mean.guess = optimize(to_opt, interval = c(log(thresh_low), absolute_max)) 
      }else{
        print("CDF cannot be 0.5.")
        return(list(mu = NA, sigma = NA))
      }
      mu.opt = mean.guess$minimum
      sigma.opt = sigma_given_mu(mu.opt, cdf_min, thresh_low)
      return(list(mu = mu.opt, sigma = sigma.opt))
    }else{
      return(list(mu = NA, sigma = NA))
    }
  }
  
  #### alternative method: log-normal-fit-2: fit based on the two cdf values
  log_normal_fit_2 = function(thresh_low, thresh_high, cdf_low, cdf_high){
    mu = (qnorm(cdf_high) * log(thresh_low) - qnorm(cdf_low) * log(thresh_high))/
      (qnorm(cdf_high) - qnorm(cdf_low))
    sigma= (log(thresh_low)-mu)/qnorm(cdf_low)
    if(sigma > 0){
      return(list(mu = mu, sigma = sigma))    
    }else{
      return(list(mu = NA, sigma = NA))
    }
  }
  
  
  # final method: for intermediate buckets, fit using the two cdf values.
  # for the final bucket, fit using the tail mean method. 
  # this ensures (implied) CDF monotonicity.
  final_fit = function(thresh_list, cdf_list, tail_mean_list){
    mu_list = rep(NA, length(thresh_list))
    sigma_list = rep(NA, length(thresh_list))
    if (length(thresh_list) == 1){
      temp.fit = log_normal_fit(thresh_low = thresh_list[length(thresh_list)], 
                                tail_mean =  tail_mean_list[length(thresh_list)], 
                                cdf_min = cdf_list[length(thresh_list)], 
                                #absolute_min = log(min(thresh_list))-4, 
                                absolute_min = 0,
                                absolute_max = log(max(thresh_list))+4)
      mu_list[length(thresh_list)] = temp.fit$mu
      sigma_list[length(thresh_list)] = temp.fit$sigma
    }
    else{
      for (f in 1:(length(thresh_list)-1)){
        temp.fit = log_normal_fit_2(thresh_list[f], thresh_list[f+1], cdf_list[f], cdf_list[f+1])
        mu_list[f] = temp.fit$mu
        sigma_list[f] = temp.fit$sigma
      }
      temp.fit = log_normal_fit(thresh_low = thresh_list[length(thresh_list)], 
                                tail_mean =  tail_mean_list[length(thresh_list)], 
                                cdf_min = cdf_list[length(thresh_list)], 
                                #absolute_min = log(min(thresh_list))-4, 
                                absolute_min = 0,
                                absolute_max = log(max(thresh_list))+4)
      mu_list[length(thresh_list)] = temp.fit$mu
      sigma_list[length(thresh_list)] = temp.fit$sigma
    }
    return(list(mu = mu_list, sigma = sigma_list))
  }
  
  # Helper: getting top 1 percent from a series of thresholds and mus and sigmas.
  # need total_mean: full distribution mean
  # tail_mean_list: the list of tail means above each thresh_low
  # mu_list, sigma_list: the interpolated mu and sigmas
  top_1_pct = function(top, thresh_low_list, mu_list, sigma_list, tail_mean_list, total_mean){
    # order in thresh_low_list, just in case
    mu_list = mu_list[order(thresh_low_list)]
    sigma_list = sigma_list[order(thresh_low_list)]
    thresh_low_list = thresh_low_list[order(thresh_low_list)]
    
    # bottom threshold should at least be lower than the top 1 percent
    if(plnorm(thresh_low_list[1], meanlog = mu_list[1], sdlog = sigma_list[1]) > 1-top){
      return(list(top_thresh = NA, top_shares = NA))
    }else{
      # find the threshold that is the minimum
      cdf_list = rep(NA, length(thresh_low_list))
      for (f in 1:length(cdf_list)){
        cdf_list[f] = plnorm(thresh_low_list[f], meanlog = mu_list[f], sdlog = sigma_list[f])
      }
      tail_mean_fitted_list = rep(NA, length(thresh_low_list))
      tail_mean_fitted_list[length(thresh_low_list)] = ln_cond_mean(lower = thresh_low_list[length(thresh_low_list)],
                                                                    upper = Inf, mu = mu_list[length(thresh_low_list)],
                                                                    sigma = sigma_list[length(thresh_low_list)])
      for(f in (length(tail_mean_fitted_list)-1):1){
        tail_mean_fitted_list[f] = ((1-cdf_list[f+1]) * tail_mean_fitted_list[f+1] + 
                                      (cdf_list[f+1] - cdf_list[f]) * ln_cond_mean(lower = thresh_low_list[f],
                                                                                   upper = thresh_low_list[f+1], 
                                                                                   mu = mu_list[f],
                                                                                   sigma = sigma_list[f]))/(1-cdf_list[f])
      }
      
      
      # largest index that is below the 99% cutoff: guaranteed to be non-empty by the above
      bottom.index = max(which(cdf_list <= 1-top))
      
      if (bottom.index == length(cdf_list)){
        # if top bucket is greater than 1 percent:
        top_thresh = qlnorm(1-top, meanlog = mu_list[length(cdf_list)], sdlog = sigma_list[length(cdf_list)])
        top_mass = top * ln_cond_mean(lower = top_thresh, mu = mu_list[length(cdf_list)], 
                                      sigma = sigma_list[length(cdf_list)])
      }else{
        # if top 1 percent is between a bucket:
        # use the mass above the max bucket, along with the interpolated mass between the buckets
        top_thresh = qlnorm(1-top, meanlog = mu_list[bottom.index], 
                            sdlog = sigma_list[bottom.index])
        lower.bucket.mean = ln_cond_mean(lower = top_thresh, upper = thresh_low_list[bottom.index + 1],
                                         mu = mu_list[bottom.index], sigma = sigma_list[bottom.index])
        upper.bucket.mean = tail_mean_fitted_list[bottom.index + 1]
        
        top_mass = (1- cdf_list[bottom.index + 1]) * upper.bucket.mean + 
          (top - (1-cdf_list[bottom.index + 1])) * lower.bucket.mean
      }
      return(list(top_thresh = top_thresh, top_shares = top_mass/total_mean))
    }
  }
  
  # ***************** Main Function for the Sector-Level Data *****************
  main_output=function(data_input,type_input,sector_input,name){
    data=data_input
    type=type_input
    
    setorder(data,year,sector_main) # reorder the data
    
    year_list<- vector()
    for(i in seq_along(data$year)) {if(!data$year[i] %in% year_list) year_list <- c(year_list, data$year[i])}
    
    sector_list <- vector()
    for(i in seq_along(data$sector_main)) {if(!data$sector_main[i] %in% sector_list) sector_list <- c(sector_list, data$sector_main[i])}
    
    setkey(data,year,sector_main)
    data_unique=unique(data, by = c("year","sector_main"))[,c("year","sector_main")]
    
    inter_output = as.data.table( matrix( NA, nrow = nrow(data_unique), ncol=7 )  )
    names(inter_output) <- c( "year","sector","top50","top10","top1","top0_1","number_bins") 
    
    num=1
    for (x in 1:nrow(data_unique)) {
        setkey(data,year,sector_main)
        data_sec=data[.(data_unique$year[x],data_unique$sector_main[x])]
        
        print(paste0("year:",data_unique$year[x]," sector:",data_unique$sector_main[x]))
        
        # skip the loop if the year-industry observation doesn't exist
        if (sum(is.na(data_sec[[type]]))==nrow(data_sec)) next
        
        inter_output$year[num]=data_unique$year[x]
        inter_output$sector[num]=data_unique$sector_main[x]
        
        if (nrow(data_sec)!=1){
          setorder(data_sec,thres_low) # reorder the data
          
          inter_output$number_bins[num]=nrow(data_sec)
          # merge the second last bin and the last bin if the last bin only has one company
          if (data_sec$number[nrow(data_sec)]==1){
            data_sec$number[nrow(data_sec)-1]=data_sec$number[nrow(data_sec)-1]+data_sec$number[nrow(data_sec)]
            data_sec[[type]][nrow(data_sec)-1]=data_sec[[type]][nrow(data_sec)-1]+data_sec[[type]][nrow(data_sec)]
            data_sec=slice(data_sec, 1:(n() - 1)) 
          }
          
          # calculate number_total and type_total without NAs
          data_sec[,number_total_new:=sum(number)]
          data_sec[,paste0(type,"_total_new"):=sum(data_sec[[type]])]
          data_sec[,paste0(type,"_total_avg"):=data_sec[[paste0(type,"_total_new")]]/data_sec$number_total_new] # total asset avg
          
          # cdf at low threshold
          setorder(data_sec,thres_low) # reorder the data
          data_sec$number_sum <- cumsum(data_sec[, data_sec$number])
          data_sec[,number_sum:=number_sum-number]
          data_sec[,cdf:=number_sum/number_total_new] # low threshold cdf
          
          # tail mean
          setorder(data_sec,-thres_low)
          data_sec$number_sum_rev <- cumsum(data_sec$number)
          data_sec[[paste0(type,"_sum_rev")]]<- cumsum(data_sec[[type]])
          data_sec$tail_mean=data_sec[[paste0(type,"_sum_rev")]]/data_sec$number_sum_rev
          
          setorder(data_sec,thres_low) # reorder the data
          data_sec[,mu:=NA]
          data_sec[,sigma:=NA]
          
          thres_low=as.numeric(data_sec$thres_low[2:as.numeric(nrow(data_sec))]) # remove the thresh_low of the first bin
          cdf_list=as.numeric(data_sec$cdf[2:as.numeric(nrow(data_sec))]) # remove the cdf of the first bin
          tail_mean_list=as.numeric(data_sec$tail_mean[2:as.numeric(nrow(data_sec))]) # remove the tail mean of the first bin
          
          # fitting
          sol=final_fit(thres_low,cdf_list,tail_mean_list)
          data_sec$mu[1]=sol$mu[1]
          data_sec$mu[2:as.numeric(nrow(data_sec))]=sol$mu
          data_sec$sigma[1]=sol$sigma[1]
          data_sec$sigma[2:as.numeric(nrow(data_sec))]=sol$sigma
          
          data_sec$lower_bound=0
          data_sec$upper_bound=log(max(thres_low))+4

          # top x% share
          inter_output$top50[num]=top_1_pct(0.5, thresh_low_list = as.numeric(data_sec$thres_low),
                                           mu_list = as.numeric(data_sec$mu),
                                           sigma_list = as.numeric(data_sec$sigma),
                                           tail_mean_list = as.numeric(data_sec$tail_mean),
                                           total_mean = as.numeric(data_sec[[paste0(type,"_total_avg")]][1]))$top_shares
          inter_output$top10[num]=top_1_pct(0.1, thresh_low_list = as.numeric(data_sec$thres_low),
                                           mu_list = as.numeric(data_sec$mu),
                                           sigma_list = as.numeric(data_sec$sigma),
                                           tail_mean_list = as.numeric(data_sec$tail_mean),
                                           total_mean = as.numeric(data_sec[[paste0(type,"_total_avg")]][1]))$top_shares
          inter_output$top1[num]=top_1_pct(0.01, thresh_low_list = as.numeric(data_sec$thres_low),
                                           mu_list = as.numeric(data_sec$mu),
                                           sigma_list = as.numeric(data_sec$sigma),
                                           tail_mean_list = as.numeric(data_sec$tail_mean),
                                           total_mean = as.numeric(data_sec[[paste0(type,"_total_avg")]][1]))$top_shares
          inter_output$top0_1[num]=top_1_pct(0.0001, thresh_low_list = as.numeric(data_sec$thres_low),
                                           mu_list = as.numeric(data_sec$mu),
                                           sigma_list = as.numeric(data_sec$sigma),
                                           tail_mean_list = as.numeric(data_sec$tail_mean),
                                           total_mean = as.numeric(data_sec[[paste0(type,"_total_avg")]][1]))$top_shares
          # store sigma as well 
          inter_output$sigma[num] = data_sec$sigma[2] # store the sigma of the second bin, which is the first bin with a fitted log-normal distribution
          num=num+1
        }else{
          inter_output$number_bins[num]=nrow(data_sec)
          num=num+1
        }
    }
    # reorder the data
    setkey(inter_output,sector,year)

    inter_output<-inter_output[number_bins!=1 & number_bins!=2]
    inter_output$top0_1=as.numeric(inter_output$top0_1)
    inter_output$top1=as.numeric(inter_output$top1)
    inter_output$top10=as.numeric(inter_output$top10)
    inter_output$top50=as.numeric(inter_output$top50)
    inter_output$sigma=as.numeric(inter_output$sigma)
    
    inter_output<-inter_output[,"number_bins":=NULL]
    
    names(inter_output) <- c( "year","sector_main","tsh_assets_ln_50pct","tsh_assets_ln_10pct","tsh_assets_ln_1pct","tsh_assets_ln_0_1pct", "sigma") 
    
    return(inter_output)
    # write_dta(inter_output,path=paste0(outpath,name,"_lognormal.dta"))
    # print(paste0("saved data to",outpath,name,"_lognormal.dta"))
  }

  
  # ***************** 2. Main *****************
  # data <- as.data.table(read_dta(paste0(inpath,"sector_brackets_assets_R5.dta"))) # load raw data
  data <- as.data.table(read_dta('/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings/data/raw/misc/Ma_Replication/programs_publish/output/soi/brackets/sector_brackets_assets_R5.dta')) # load raw data
  result <- main_output(data_input=data,type_input="assets",sector_input="sector",name="sector_assets")
  
  all_trend <- result %>% filter(sector_main == "All", year >= 1980) %>% select(year, sigma) 
  library(ggplot2) 
  ggplot(all_trend, aes(x = year, y = sigma)) +
    geom_line() +
    geom_point() +
    labs(title = "Trend of Sigma for All Sectors", x = "Year", y = "Sigma") +
    theme_minimal()

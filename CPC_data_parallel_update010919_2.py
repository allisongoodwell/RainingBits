#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:31:22 2018

January 18 update: compute info values for 3-year non-overlapping windows
instead of 5 year moving windows

@author: allisongoodwell
"""

from netCDF4 import Dataset
import numpy as np
np.seterr(all='ignore')
import pickle
from multiprocessing import Pool
from find_Icrit_binary import find_Icrit_binary
import scipy.stats as stats
import time
import gc


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

ncores=1
threshvals = [0, 10, 50]
seasons =[0,1,2,3]


#class NumPyArangeEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, np.ndarray):
#            return obj.tolist() # or map(int, obj)
#        return json.JSONEncoder.default(self, obj)

years = range(1948,2018)

def compute_pdfs(ppt_list_in):
    
    #print('in pdfs function')
    
    
    ppt_list_vals = ppt_list_in[0]
    thresholds = np.asarray(ppt_list_in[1])
    thresholds = np.reshape(thresholds,(3))
    
    #print(thresholds)
    
    if np.sum(thresholds)==0:
        return np.nan
    
    years = range(1948,2018)
    
       
    ndays_per_month = np.asfarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    ndays_cum = np.cumsum(ndays_per_month)
    
    svals=[]
    svals.append([i for i in range(0,365) if i<ndays_cum[1] or i >=ndays_cum[10]])
    svals.append([i for i in range(0,365) if i>=ndays_cum[1] and i < ndays_cum[5]])
    svals.append([i for i in range(0,365) if i>=ndays_cum[5] and i < ndays_cum[8]])
    svals.append([i for i in range(0,365) if i>=ndays_cum[8] and i < ndays_cum[10]])
       
    pdf_year_list=[]
    for y_ind,y in enumerate(years):
            pdf_season_list=[]
            for s in range(0,4):

                s_inds = svals[s]
                
                if s>0: #spring, summer, fall
                    pptvect_ys = ppt_list_vals[y_ind][s_inds]
                else: #winter, take previous Dec and current Jan+Feb
                    ppt1 = ppt_list_vals[y_ind-1][range(int(ndays_cum[10]),int(ndays_cum[11]))]
                    ppt2 = ppt_list_vals[y_ind][range(0,int(ndays_cum[1]))]
                    pptvect_ys = np.concatenate((ppt1,ppt2))
                
                pptvect_ys = pptvect_ys[~np.isnan(pptvect_ys)]
                
                if np.size(pptvect_ys)<30:
                    pdf_season_list.append()
                    continue #continue to next season/year
                
            
                pdf_prctile_list=[]
                
                for t_ind,thresh in enumerate(thresholds): 
                    
                    pptvect_binary = np.zeros(np.shape(pptvect_ys))
                    
                    pptvect_thresholds = pptvect_ys
                    pptvect_thresholds[pptvect_thresholds<thresh]=0
                 
                    
                    pptvect_binary[pptvect_thresholds>0]=1
                    
                                                          
                    #print(thresh, np.sum(pptvect_binary))
                                  
                    pdf_lag_list=[]
                    for lag2 in range(2,20):
                        Xs1 = pptvect_binary[(lag2-1):-1] #most recent lag
                        Xs2 = pptvect_binary[0 : (-lag2)] #tau lag
                        Xtar = pptvect_binary[lag2:]      #current
                        
                        Tuple = np.vstack((Xs1,Xs2,Xtar))
                        Tuple = np.transpose(Tuple)
                        Tuple = Tuple[~np.isnan(Tuple).any(axis=1)]
                           
                        pdf,edges = np.histogramdd(Tuple,bins=([-.01, .9, 1.1],[-.01, .9, 1.1],[-.01, .9, 1.1]))
                        sumpdf = np.sum(pdf)
                        if sumpdf>0:
                            pdf_lag_list.append(pdf/np.sum(pdf))
                        else:
                            pdf_lag_list.append(0)
                              
                    pdf_prctile_list.append(pdf_lag_list)
                    
                pdf_season_list.append(pdf_prctile_list)
            pdf_year_list.append(pdf_season_list)
 
    return pdf_year_list


def compute_info_measures(pdf):

         
    m_jk = np.sum(pdf,axis=0)
    m_ij = np.sum(pdf,axis=2)
    m_ik = np.sum(pdf,axis=1)
    
    m_i = np.sum(m_ij,axis=1)
    m_j = np.sum(m_ij,axis=0)
    m_k = np.sum(m_jk,axis=0)
    
    #local information measures: total information
    pi_pjk = np.reshape(np.outer(m_ij,m_k),(2,2,2))
    f_ijk = pdf/pi_pjk
    
    logf_main = np.where(pdf != 0, np.log2(f_ijk), 0)   
    plogf_main = pdf*logf_main
    Ix1x2y = np.sum(plogf_main)
      
    #for x1 and y
    f_ik = m_ik/(np.outer(m_i,m_k))
    Ix1y = np.sum(m_ik * np.where(m_ik != 0, np.log2(f_ik), 0))

    Hy = -np.sum(m_k * np.where(m_k !=0, np.log2(m_k),0))
    
    #Eta S+U2 / total information

    
    Eta = np.divide(Ix1x2y-Ix1y,Ix1x2y)
    
    #if Eta>1:
        #print('Eta problem: ', str(Eta))
       
    Ifrac = np.divide(Ix1x2y,Hy)     
    infodict = {'Eta':Eta,'Ifrac':Ifrac,'Itotal':Ix1x2y}
                
    return infodict

def YearlyStats(pdfs):
    
    #print('in yearly stats function')
    
    #where pdfs are for given lat, lon pair (a single grid cell)
    threshvals = [0, 10, 50]
    years = range(1948,2018) 
    years3 = range(1948,2018,3) #every 3 years
    seasons = [0,1,2,3]
    
    #yearly_X are actually 3 year averages
    
    yearly_lag=   np.empty((len(seasons),len(threshvals),len(years3)))
    yearly_Itot=  np.empty((len(seasons),len(threshvals),len(years3)))
    yearly_Ifrac= np.empty((len(seasons),len(threshvals),len(years3)))
    yearly_prain= np.empty((len(seasons),len(threshvals),len(years3)))
    yearly_Icrit= np.empty((len(seasons),len(threshvals),len(years3)))
    yearly_Etanorm =  np.empty((len(seasons),len(threshvals),len(years3)))
    yearly_pdf =  np.empty((len(seasons),len(threshvals),len(years3),2,2,2))
    
    yearly_lag.fill(np.nan)
    yearly_Itot.fill(np.nan)
    yearly_Ifrac.fill(np.nan)
    yearly_prain.fill(np.nan)
    yearly_Icrit.fill(np.nan)
    yearly_Etanorm.fill(np.nan)
    
    Trend_Eta=   np.empty((len(seasons),len(threshvals)))
    Trend_Lag=   np.empty((len(seasons),len(threshvals)))
    Trend_Ifrac=   np.empty((len(seasons),len(threshvals)))
    Trend_Itot=   np.empty((len(seasons),len(threshvals)))
    Trend_Prain=   np.empty((len(seasons),len(threshvals)))
    trendlist=   np.empty((len(seasons),len(threshvals)))
    
    Trend_Eta.fill(np.nan)
    Trend_Lag.fill(np.nan)
    Trend_Ifrac.fill(np.nan)
    Trend_Itot.fill(np.nan)
    Trend_Prain.fill(np.nan)
    trendlist.fill(np.nan)
    
    if np.size(pdfs)<2: #if no value for that lat,lon index (e.g. over water)
        return 0

    for p_ind,p in enumerate(threshvals):
        for s in range(0,4):
            
            #print(s)
            
            pdfs_years = [pdfs[y][s][p_ind][:] for y,i in enumerate(years)]
         
            #information measures for 3-year windows (non-overlapping)
            for y_ind,y in enumerate(years3):
                
                
                start_ind = y - 1948
                end_ind = np.min([start_ind+3,len(years)])
                
                pdfs_3years = pdfs_years[start_ind:end_ind][:]
                avgs = np.average(pdfs_3years,axis=0)
                info_list=[]
                for a in avgs:
                    info_list.append(compute_info_measures(a))
    
                Ivect = [float(info_list[i]['Itotal']) for i in range(0,18)]
                Ivect = np.asfarray(Ivect)
                           
                p_rain = np.sum(avgs[0][:][:][1]) 
                npts = 3*90
                n_ones = np.int(npts*p_rain)
                
                Icrit = find_Icrit_binary(n_ones,npts,100)
                Ivect[Ivect<Icrit]=0
    
                maxindex = np.argmax(Ivect)
                maxIval = np.max(Ivect)
                    
                pdfmax = avgs[maxindex] 
                infomax = info_list[maxindex]
                    
                if maxIval>0:
                    Itot = float(infomax['Itotal'])
                    Ifrac = float(infomax['Ifrac'])
                    LagMax_all = maxindex
                    Eta = float(infomax['Eta'])
                else:
                    Itot = 0
                    Ifrac=0
                    LagMax_all = float('NaN')
                    Eta = float('NaN')
              
                yearly_lag[s,p_ind,y_ind]=LagMax_all
                yearly_Itot[s,p_ind,y_ind] = Itot
                yearly_Ifrac[s,p_ind,y_ind] = Ifrac
                yearly_prain[s,p_ind,y_ind] =p_rain
                yearly_Icrit[s,p_ind,y_ind] =Icrit
                yearly_Etanorm[s,p_ind,y_ind] =Eta
                yearly_pdf[s,p_ind,y_ind,:,:,:]=pdfmax
         
    
            #determine annual trends
            
            Prain_avg = yearly_prain[s,p_ind,:]
            Eta_avg = yearly_Etanorm[s,p_ind,:]
            Lag_avg = yearly_lag[s,p_ind,:]
            Ifrac_avg = yearly_Ifrac[s,p_ind,:]
            
            #update: only compute trend on stat sig values (omit zeros and nan values)

            Prain_avg = Prain_avg[Ifrac_avg>0]
            Lag_avg = Lag_avg[Ifrac_avg>0]
            Eta_avg = Eta_avg[Ifrac_avg>0]
            YearVect = np.asarray(years3)
            YearVect= YearVect[Ifrac_avg>0]
            
            Ifrac_avg = Ifrac_avg[Ifrac_avg>0]
            
            
            if len(YearVect)>15:
                SenIfrac = stats.theilslopes(Ifrac_avg,YearVect, 0.9)
                SenPrain = stats.theilslopes(Prain_avg,YearVect, 0.9)
                SenEta = stats.theilslopes(Eta_avg,YearVect, 0.9)
                SenLag = stats.theilslopes(Lag_avg,YearVect, 0.9)
            
                #print SenItot[0], SenIfrac[0], SenPrain[0]
        
                sign_Ifrac_slope = np.sign(SenIfrac[2])*np.sign(SenIfrac[3])
                sign_Prain_slope = np.sign(SenPrain[2])*np.sign(SenPrain[3])
                sign_Eta_slope = np.sign(SenEta[2])*np.sign(SenEta[3])
                sign_Lag_slope = np.sign(SenLag[2])*np.sign(SenLag[3])
            
                SenIfrac_ind=np.sign(SenIfrac[0])
                SenPrain_ind=np.sign(SenPrain[0])
                SenEta_ind = np.sign(SenEta[0])
                SenLag_ind =np.sign(SenLag[0])
                
                if SenIfrac[0] >0:
                    SenIfrac_ind = 1
                else:
                    SenIfrac_ind = -1
                    
                if SenPrain[0] >0:
                    SenPrain_ind = 10
                else:
                    SenPrain_ind = -10
                    
                if SenEta[0] >0:
                    SenEta_ind = 100
                else:
                    SenEta_ind = -100
                    
                if SenLag[0] >0:
                    SenLag_ind = 1000
                else:
                    SenLag_ind = -1000
                    
                SenIfrac=SenIfrac[0]
                SenLag=SenLag[0]
                SenPrain=SenPrain[0]
                SenEta=SenEta[0]
                
                
                if sign_Ifrac_slope<0:
                    SenIfrac=0
                    SenIfrac_ind=0
                if sign_Prain_slope<0:
                    SenPrain=0
                    SenPrain_ind=0
                if sign_Eta_slope<0:
                    SenEta=0  
                    SenEta_ind=0  
                if sign_Lag_slope<0:
                    SenLag=0
                    SenLag_ind=0
                
           
                trendlist[s,p_ind] = SenIfrac_ind+SenPrain_ind+SenEta_ind+SenLag_ind
                Trend_Ifrac[s,p_ind]=SenIfrac
                Trend_Prain[s,p_ind]=SenPrain
                Trend_Eta[s,p_ind]=SenEta
                Trend_Lag[s,p_ind]=SenLag
                
            else:
                trendlist[s,p_ind] =  float('NaN')
                Trend_Ifrac[s,p_ind]= float('NaN')
                Trend_Prain[s,p_ind]= float('NaN')
                Trend_Eta[s,p_ind]= float('NaN')
                Trend_Lag[s,p_ind]= float('NaN')
            
    results_dict = {'yearly_lag':yearly_lag, 'yearly_eta':yearly_Etanorm,'yearly_Ifrac':yearly_Ifrac,
                    'yearly_Itot':yearly_Itot, 'yearly_pdf':yearly_pdf,'yearly_prain':yearly_prain,
                    'trendlist':trendlist, 'Trend_Ifrac':Trend_Ifrac,'Trend_Prain':Trend_Prain,
                    'Trend_Eta':Trend_Eta,'Trend_Lag':Trend_Lag}
    
    return results_dict

if __name__ == '__main__':

    ppt_list=[]
    for y in years:
        #string = 'precip.V1.0.'+str(y)+'.nc'
        string = 'C:\\Users\\goodwela\\Dropbox\\UCDenver\\rainfall_research\\CPC_raingage_gridded_dailydata\\original_precip_data\\precip.V1.0.'+str(y)+'.nc'
        dataset = Dataset(string)
        #print(dataset.file_format)
        #print(dataset.dimensions.keys())
        #print dataset.dimensions['time']
        #print(dataset.variables.keys())
        
        pptdata = np.asarray(dataset.variables['precip'][:])
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
        t = dataset.variables['time'][:]
        
        
        pptdata[pptdata<0]=np.nan
        
        ppt_list.append(pptdata)
        
        #plt.imshow(jan1data)
    
    f_myfile = open('Coordinates.pickle','wb')
    pickle.dump(lat, f_myfile) #ppt_long_list = rainfall data [0] and percentiles [1]
    #for all longitudes at a given latitude
    pickle.dump(lon,f_myfile) #pdf_long_list = pdfs [longitude][year][season][percentile][lag2]
    f_myfile.close()
    
        
    #%%  Determine percentiles (10,20,30...90) for each coordinate
    PrctPPT = np.zeros((3,np.size(lat),np.size(lon))) 
    
    #overall_lag=   np.empty((len(lat),len(lon),len(seasons),len(threshvals)))
    #overall_Itot=  np.empty((len(lat),len(lon),len(seasons),len(threshvals)))
    #overall_Ifrac= np.empty((len(lat),len(lon),len(seasons),len(threshvals)))
    #overall_prain= np.empty((len(lat),len(lon),len(seasons),len(threshvals)))
    #overall_Icrit= np.empty((len(lat),len(lon),len(seasons),len(threshvals)))
    #overall_Etanorm =  np.empty((len(lat),len(lon),len(seasons),len(threshvals)))
    #overall_pdf =  np.empty((len(lat),len(lon),len(seasons),len(threshvals),2,2,2))
    #
    #overall_lag.fill(np.nan)
    #overall_Itot.fill(np.nan)
    #overall_Ifrac.fill(np.nan)
    #overall_prain.fill(np.nan)
    #overall_Icrit.fill(np.nan)
    #overall_Etanorm.fill(np.nan)
    #
    #
    #
    pdf_all_list=[]
        
    
    for la_ind,la in enumerate(lat):
        print('latitude = '+str(la))
        pdf_long_list=[]
        ppt_long_list=[]
        
        #save lat and long value in a text file called "progress"
        textfile = open('progress.txt','w')
        textfile.write(str(la))
                
        for lo_ind,lo in enumerate(lon):
            
    
            #print('initial: latitude = '+str(la_ind) + '  longitude = ' + str(lo_ind))
            
            
            ppt_list_mini = [ppt_list[x][:,la_ind,lo_ind] for x in range(0,np.size(ppt_list))]
            PPTvect = np.concatenate(ppt_list_mini)
            #PPTvect is all precip for a given coordinate
            #for percentiles, use only rainy days (omit zero values)
            PPTvect_rainonly = PPTvect[PPTvect>0.3]
            
            for p_ind,p in enumerate(threshvals):
                if np.size(PPTvect_rainonly)>0:
                    PrctPPT[p_ind,la_ind,lo_ind] = np.percentile(PPTvect_rainonly,p)
                    
    #         
    #            
    #        ppt_list_with_thresholds = [ppt_list_mini, [PrctPPT[:,la_ind,lo_ind]]]
    #        
    #        pdfs = compute_pdfs(ppt_list_with_thresholds) 
    #        
    #        if np.size(pdfs)<2: #if no value for that lat,lon index (e.g. over water)
    #            continue
    #    
    #        
    #        for p_ind,p in enumerate(threshvals):
    #            print('latitude = '+str(la)+'  longitude = '+str(lo)+ '  threshold percent = '+str(p)+ ' thresh val ' + str(PrctPPT[p_ind,la_ind,lo_ind]))
    #            
    #            for s in range(0,4):
    #                
    #                
    #                pdfs_years = [pdfs[y][s][p_ind][:] for y,i in enumerate(years)]
    #
    #                avgs = np.average(pdfs_years,axis=0)
    #                info_list=[]
    #                for a in avgs:
    #                    info_list.append(compute_info_measures(a))
    #                                 
    #                #get vector of I_total values, statistical sig test
    #                Ivect = [float(info_list[i]['Itotal']) for i in range(0,18)]
    #                Ivect = np.asfarray(Ivect)
    #                           
    #                p_rain = np.sum(avgs[0][:][:][1]) 
    #                #print(' season ' +str(s) + 'p_rain = ', str(p_rain))
    #                npts = len(years)*90
    #                n_ones = np.int(npts*p_rain)
    #                
    #                Icrit = find_Icrit_binary(n_ones,npts,100)
    #                Ivect[Ivect<Icrit]=0
    #    
    #                maxindex = np.argmax(Ivect)
    #                maxIval = np.max(Ivect)
    #                    
    #                pdfmax = avgs[maxindex] 
    #                infomax = info_list[maxindex]
    #                    
    #                if maxIval>0:
    #                    Itot = float(infomax['Itotal'])
    #                    Ifrac = float(infomax['Ifrac'])
    #                    LagMax_all = maxindex
    #                    Eta = float(infomax['Eta'])
    #                else:
    #                    Itot = 0
    #                    Ifrac=0
    #                    LagMax_all = float('NaN')
    #                    Eta = float('NaN')
    #                    
    #                if Eta>1:
    #                    print('Eta problem HERE')
    #                    break
                        
    #                
    #                #print(Itot,Ifrac,Eta,LagMax_all)
    #                
    #                overall_lag[la_ind,lo_ind,s,p_ind]=LagMax_all
    #                overall_Itot[la_ind,lo_ind,s,p_ind] = Itot
    #                overall_Ifrac[la_ind,lo_ind,s,p_ind] = Ifrac
    #                overall_prain[la_ind,lo_ind,s,p_ind] =p_rain
    #                overall_Icrit[la_ind,lo_ind,s,p_ind] =Icrit
    #                overall_Etanorm[la_ind,lo_ind,s,p_ind] =Eta
    #                overall_pdf[la_ind,lo_ind,s,p_ind,:,:,:]=pdfmax
    #                                   
    #
    #                    
    #save_data_df = {'Lat':lat,'Lon':lon,
    #'Prain_all':overall_prain,'Icrit_all':overall_Icrit, 'Lag_all':overall_lag, 'Ifrac_all':overall_Ifrac,
    # 'Itot_all':overall_Itot,'Etanorm_all':overall_Etanorm,'Pdf_all':overall_pdf,'Percentiles':PrctPPT}
    #
    #site_filenamesave='gridded_info_overall.pickle'                  
    #f_myfile = open(site_filenamesave,'wb')
    #pickle.dump(save_data_df, f_myfile)
    #f_myfile.close()
            
    #%% after overall analysis, also do trend analysis
    
    AllResults =[]
    Lat=[]
    Long_Lists=[]
    
    for la_ind,la in enumerate(lat):
        #print('latitude = '+str(la))
        pdf_long_list=[]
        ppt_long_list=[]
        
        #save lat and long value in a text file called "progress"
        textfile = open('progresstrends.txt','w')
        textfile.write(str(la))
        
        #loop over lo_ind, get list of lists for pdfs - have function to work on that list  
        pdfs =[]   
        long_list=[]
        ppt_list_lons =[]
        for lo_ind,lo in enumerate(lon):
            
            #print(lo)
            #check to see if that lat,lon coord is valid (contains data)
            if PrctPPT[2,la_ind,lo_ind]<.0001:
                continue
                        
            ppt_list_mini = [ppt_list[x][:,la_ind,lo_ind] for x in range(0,np.size(ppt_list))]
            ppt_list_with_thresholds = [ppt_list_mini, [PrctPPT[:,la_ind,lo_ind]]]
            ppt_list_lons.append(ppt_list_with_thresholds)
            long_list.append(lon)
         
        gc.collect()    
        t1 = time.clock()
        print('going into pdfs with '+ str(len(ppt_list_lons))+ ' for latitude '+ str(la))    
        p = Pool(14)
        pdfs = p.map(compute_pdfs,ppt_list_lons) #pdfs should be a list of lists
        p.close
        p.join
        
      
        t2 = time.clock()  
        print('time elapsed for pdfs = '+ str(t2-t1))
        
        
        #run function on pdfs to get 3 year windows and trends
        if np.size(pdfs)>0:
            print('going into stats with '+str(len(pdfs))+'for latitude ' + str(la))
            p = Pool(14)
            results = p.map(YearlyStats,pdfs) 
            p.close()
            p.join()
        else:
            results = 0
        
        t3 = time.clock()
        print('time elapsed for stats = '+ str(t3-t2))

        gc.collect()
            #results should be a list of dictionaries
            
        #    {'yearly_lag':yearly_lag, 'yearly_eta':yearly_Etanorm,'yearly_Ifrac':yearly_Ifrac,
        #                    'yearly_Itot':yearly_Itot, 'yearly_pdf':yearly_pdf,'yearly_prain':yearly_prain,
        #                    'trendlist':trendlist, 'Trend_Ifrac':Trend_Ifrac,'Trend_Prain':Trend_Prain,
        #                    'Trend_Eta':Trend_Eta,'Trend_Lag':Trend_Lag}
                 
        AllResults.append(results)   
        Long_Lists.append(long_list)
        Lat.append(la)
        
    site_filenamesave = 'gridded_trends.pickle'
    f_myfile = open(site_filenamesave,'wb')
    pickle.dump(AllResults,f_myfile)
    pickle.dump(Long_Lists,f_myfile)
    pickle.dump(Lat,f_myfile)
    f_myfile.close()
          
        #save arrays as dictionary
        #save_data_df = {'Lat':lat,'Lon':lon,'Prain_yr':yearly_prain,'Icrit_yr':yearly_Icrit, 'Lag_yr':yearly_lag, 'Ifrac_yr':yearly_Ifrac,
        # 'Itot_yr':yearly_Itot,'Etanorm_yr':yearly_Etanorm,'Pdf_yr':yearly_pdf,'Pdf_all':overall_pdf, 'Trend_Ifrac':Trend_Ifrac,
        # 'Trend_Prain':Trend_Prain,'Trend_Eta':Trend_Eta,'Trend_Lag':Trend_Lag,'TrendListBinary':trendlist}
    
    
        
    #    save_data_df = {'Lat':lat,'Lon':lon, 'Trend_Ifrac':Trend_Ifrac,
    #     'Trend_Prain':Trend_Prain,'Trend_Eta':Trend_Eta,'Trend_Lag':Trend_Lag,'TrendListBinary':trendlist}
    #    
    #    site_filenamesave='gridded_info_trends.pickle'                  
    #    f_myfile = open(site_filenamesave,'wb')
    #    pickle.dump(save_data_df, f_myfile)
    #    f_myfile.close()
    #    
    #    save_data_df = {'Lat':lat,'Lon':lon,'Prain_yr':yearly_prain, 'Lag_yr':yearly_lag, 'Ifrac_yr':yearly_Ifrac,
    #                    'Etanorm_yr':yearly_Etanorm}
    #    
    #    site_filenamesave='gridded_info_annual.pickle'                  
    #    f_myfile = open(site_filenamesave,'wb')
    #    pickle.dump(save_data_df, f_myfile)
    #    f_myfile.close()

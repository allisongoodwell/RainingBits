# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:27:06 2018
Take the pickle file from from CPC gridded data analysis, determine watershed averages
create map files for viewing in ArcMap: Prain, Ifrac, Eta, Lag
overall US characteristics and trends


@author: goodwela
"""
import numpy as np
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import shapefile
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

from shapely.geometry import Polygon
from shapely.geometry import Point

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


threshval = [0, 10, 50]
HUC = 2

make_shapefiles = 0
make_plots = 0

site_filenameopen = 'gridded_info_overall.pickle'
f_myfile = open(site_filenameopen,'rb')
alldata = pickle.load(f_myfile)
f_myfile.close()

site_filenameopen = 'gridded_trends.pickle'
f_myfile = open(site_filenameopen,'rb')
AllResults =    pickle.load(f_myfile)
dummy = pickle.load(f_myfile)
Lat = pickle.load(f_myfile)

site_filenameopen = 'gridded_trends_annual.pickle'
f_myfile = open(site_filenameopen,'rb')
AllAnnualResults=pickle.load(f_myfile) 
dummy = pickle.load(f_myfile) 
Lat = pickle.load(f_myfile) 
f_myfile.close()

filenameopen = 'list_of_longitudes.pickle'
f_myfile = open(filenameopen,'rb') 
Long_Lists = pickle.load(f_myfile) 
f_myfile.close()

#AllResults is a list (for each lat) of lists (for each longitude) of dictionaries: 
#'yearly_lag':yearly_lag, 'yearly_eta':yearly_Etanorm,'yearly_Ifrac':yearly_Ifrac,
#                    'yearly_Itot':yearly_Itot, 'yearly_pdf':yearly_pdf,'yearly_prain':yearly_prain,
#                    'trendlist':trendlist, 'Trend_Ifrac':Trend_Ifrac,'Trend_Prain':Trend_Prain,
#                    'Trend_Eta':Trend_Eta,'Trend_Lag':Trend_Lag

lat = alldata['Lat']
lon= alldata['Lon']
altlon = 360-lon

Prain = alldata['Prain_all']
Lag = alldata['Lag_all']
Ifrac = alldata['Ifrac_all']
Itot = alldata['Itot_all']
Eta = alldata['Etanorm_all']

#Prain_Trend = alltrends['Trend_Prain']
#Lag_Trend = alltrends['Trend_Lag']
#Ifrac_Trend = alltrends['Trend_Ifrac']
#Eta_Trend = alltrends['Trend_Eta']

f_myfile = open('Coordinates.pickle','rb')
lat = pickle.load(f_myfile) 
lon=pickle.load(f_myfile)
f_myfile.close()

#load in HUC4 shapefile
#sf_HUC4 = shapefile.Reader('shapefiles/All_HUC4_shapefile/All_HUC4_shapefile')
sf_HUC4 = shapefile.Reader('C:\\Users\\goodwela\\Dropbox\\UCDenver\\rainfall_research\\CPC_raingage_gridded_dailydata\\shapefiles\\All_HUC4_shapefile\\All_HUC4_shapefile')
nshapes_H4 = len(sf_HUC4.shapes())
records_H4 = sf_HUC4.records()

#load in HUC2 shapefile
#sf_HUC2 = shapefile.Reader('shapefiles/All_HUC2_shapefile/All_HUC2_shapefile')
sf_HUC2 = shapefile.Reader('C:\\Users\\goodwela\\Dropbox\\UCDenver\\rainfall_research\\CPC_raingage_gridded_dailydata\\shapefiles\\All_HUC2_shapefile\\All_HUC2_shapefile')
nshapes_H2 = len(sf_HUC2.shapes())
records_H2 = sf_HUC2.records()


stringlist_HUC4_overallvals = [["HUC4_DJF0","HUC4_MAM0","HUC4_JJA0","HUC4_SON0"], 
                          ["HUC4_DJF10","HUC4_MAM10","HUC4_JJA10","HUC4_SON10"],
                          ["HUC4_DJF50","HUC4_MAM50","HUC4_JJA50","HUC4_SON50"]]

stringlist_HUC2_overallvals = [["HUC2_DJF0","HUC2_MAM0","HUC2_JJA0","HUC2_SON0"], 
                          ["HUC2_DJF10","HUC2_MAM10","HUC2_JJA10","HUC2_SON10"],
                          ["HUC2_DJF50","HUC2_MAM50","HUC2_JJA50","HUC2_SON50"]]

stringlist_HUC4_trends = [["DJF_HUC4_trends0","MAM_HUC4_trends0","JJA_HUC4_trends0","SON_HUC4_trends0"],
                     ["DJF_HUC4_trends10","MAM_HUC4_trends10","JJA_HUC4_trends10","SON_HUC4_trends10"],
                     ["DJF_HUC4_trends50","MAM_HUC4_trends50","JJA_HUC4_trends50","SON_HUC4_trends50"]]
                     
stringlist_HUC2_trends = [["DJF_HUC2_trends0","MAM_HUC2_trends0","JJA_HUC2_trends0","SON_HUC2_trends0"],
                     ["DJF_HUC2_trends10","MAM_HUC2_trends10","JJA_HUC2_trends10","SON_HUC2_trends10"],
                     ["DJF_HUC2_trends50","MAM_HUC2_trends50","JJA_HUC2_trends50","SON_HUC2_trends50"]]


stringlist_HUC2Annual_trends = ["HUC2_annualtrends0","HUC2_annualtrends10","HUC2_annualtrends50"]
stringlist_HUC4Annual_trends = ["HUC4_annualtrends0","HUC4_annualtrends10","HUC4_annualtrends50"]



if HUC==2:
    stringlist_overall = stringlist_HUC2_overallvals
    stringlist_annualtrends = stringlist_HUC2Annual_trends
    stringlist_trends = stringlist_HUC2_trends
    nshapes_H = nshapes_H2
    records_H = records_H2
    sf_HUC = sf_HUC2
else:
    stringlist_overall = stringlist_HUC4_overallvals
    stringlist_annualtrends = stringlist_HUC4Annual_trends
    stringlist_trends = stringlist_HUC4_trends   
    nshapes_H = nshapes_H4
    records_H = records_H4
    sf_HUC = sf_HUC4


HUCpolys = []
HUCnames=[]
HUCnums=[]
site_HUCnum = np.zeros((len(lat),len(lon)))
site_HUCname = np.empty((len(lat),len(lon)), dtype=object)


#%%
####### polygons for HUC4 watersheds #######
HUC_Ifrac_watersheds =[]
HUC_Itot_watersheds =[]
HUC_Lag_watersheds =[]
HUC_Eta_watersheds =[]
HUC_Prain_watersheds =[]
HUC_Npts_watersheds =[]

HUC_IfracTrend_watersheds =[]
HUC_LagTrend_watersheds =[]
HUC_EtaTrend_watersheds =[]
HUC_PrainTrend_watersheds =[]

HUC_IfracAnnualTrend_watersheds =[]
HUC_LagAnnualTrend_watersheds =[]
HUC_EtaAnnualTrend_watersheds =[]
HUC_PrainAnnualTrend_watersheds =[]

HUC_IfracAnnualAvg_watersheds =[]
HUC_LagAnnualAvg_watersheds =[]
HUC_EtaAnnualAvg_watersheds =[]
HUC_PrainAnnualAvg_watersheds =[]

HUC_Longitude_watersheds=[]

for s in range(0,nshapes_H):
    
    HUC_Ifrac=[]
    HUC_Itot=[]
    HUC_Lag=[]
    HUC_Eta=[]
    HUC_Prain=[]
    
    HUC_IfracTrend=[]
    HUC_LagTrend=[]
    HUC_EtaTrend=[]
    HUC_PrainTrend=[]
    
    HUC_IfracAnnualTrend=[]
    HUC_LagAnnualTrend=[]
    HUC_EtaAnnualTrend=[]
    HUC_PrainAnnualTrend=[]
    
    HUC_IfracAnnualAvg=[]
    HUC_LagAnnualAvg=[]
    HUC_EtaAnnualAvg=[]
    HUC_PrainAnnualAvg=[]
    
    HUC_Longitude =[]
    
    shape_ex = sf_HUC.shape(s)
    place_string = records_H[s][11]
    place_num=records_H[s][10]
    
    x_lon = np.zeros((len(shape_ex.points),))
    y_lat = np.zeros((len(shape_ex.points),))
    coords = np.zeros((len(shape_ex.points),2))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
        coords[ip,0]=shape_ex.points[ip][0]
        coords[ip,1]=shape_ex.points[ip][1]

    #create polygon of that watershed shape
    poly = Polygon((coords))
    #poly = MultiPoint(coords).convex_hull
    HUCpolys.append(poly)
    HUCnames.append(place_string)
    HUCnums.append(place_num)
    
    #only search in box near x_lon and y_lat points
    lat_short = lat[lat>=np.min(y_lat)]
    lat_short = lat_short[lat_short<=np.max(y_lat)]
    
    lon_short = altlon[altlon>=np.min(-x_lon)]
    lon_short = lon_short[lon_short<=np.max(-x_lon)]
    
    for la_ind,la in enumerate(lat):
        print('lat= '+str(la))
        long_list = Long_Lists[la_ind]
        trend_results = AllResults[la_ind]
        annualtrend_results = AllAnnualResults[la_ind]
        
        for lo_ind,lo in enumerate(altlon):
     
            inside = HUCpolys[s].contains(Point((-lo,la)))
            #print(inside, HUCnames[s])
            
            if inside == True and len(long_list)>0:
                
                long_index = [i for i, j in enumerate(long_list) if j == 360-lo]
                
                
                if len(long_index)<1:
                    continue
                
                long_index = long_index[0]
                
                site_HUCnum[la_ind,lo_ind]=int(HUCnums[s])
                site_HUCname[la_ind,lo_ind]=HUCnames[s]
                
                HUC_Ifrac.append(Ifrac[la_ind,lo_ind,:,:])
                HUC_Itot.append(Itot[la_ind,lo_ind,:,:])
                HUC_Lag.append(Lag[la_ind,lo_ind,:,:])
                HUC_Prain.append(Prain[la_ind,lo_ind,:,:])
                HUC_Eta.append(Eta[la_ind,lo_ind,:,:])
                
                HUC_Longitude.append(lo)
                
                Ifrac_trend = trend_results[long_index]['Trend_Ifrac']
                Lag_trend = trend_results[long_index]['Trend_Lag']
                Prain_trend = trend_results[long_index]['Trend_Prain']
                Eta_trend = trend_results[long_index]['Trend_Eta']
                
                Ifrac_annualtrend = annualtrend_results[long_index]['Trend_Ifrac']
                Lag_annualtrend = annualtrend_results[long_index]['Trend_Lag']
                Prain_annualtrend = annualtrend_results[long_index]['Trend_Prain']
                Eta_annualtrend = annualtrend_results[long_index]['Trend_Eta']
                
                Ifrac_annualaverage = annualtrend_results[long_index]['yearly_Ifrac']
                Lag_annualaverage = annualtrend_results[long_index]['yearly_lag']
                Prain_annualaverage = annualtrend_results[long_index]['yearly_prain']
                Eta_annualaverage = annualtrend_results[long_index]['yearly_eta']
                
                HUC_IfracTrend.append(Ifrac_trend)
                HUC_LagTrend.append(Lag_trend)
                HUC_PrainTrend.append(Prain_trend)
                HUC_EtaTrend.append(Eta_trend)
                
                HUC_IfracAnnualTrend.append(Ifrac_annualtrend)
                HUC_LagAnnualTrend.append(Lag_annualtrend)
                HUC_PrainAnnualTrend.append(Prain_annualtrend)
                HUC_EtaAnnualTrend.append(Eta_annualtrend)
                
                HUC_IfracAnnualAvg.append(Ifrac_annualaverage)
                HUC_LagAnnualAvg.append(Lag_annualaverage)
                HUC_PrainAnnualAvg.append(Prain_annualaverage)
                HUC_EtaAnnualAvg.append(Eta_annualaverage)
                
                
                print('lat= '+str(la)+' lon= ',str(lo)+ ' HUC location: ', HUCnames[s])
                continue
     
    HUC_Ifrac_watersheds.append(HUC_Ifrac) 
    HUC_Itot_watersheds.append(HUC_Itot)
    HUC_Eta_watersheds.append(HUC_Eta) 
    HUC_Lag_watersheds.append(HUC_Lag) 
    HUC_Prain_watersheds.append(HUC_Prain) 
    HUC_Npts_watersheds.append(int(np.size(HUC_Ifrac)/12)) 

    HUC_Longitude_watersheds.append(np.average(HUC_Longitude))    

    HUC_IfracTrend_watersheds.append(HUC_IfracTrend)  
    HUC_EtaTrend_watersheds.append(HUC_EtaTrend) 
    HUC_LagTrend_watersheds.append(HUC_LagTrend) 
    HUC_PrainTrend_watersheds.append(HUC_PrainTrend)  
    
    HUC_IfracAnnualTrend_watersheds.append(HUC_IfracAnnualTrend)  
    HUC_EtaAnnualTrend_watersheds.append(HUC_EtaAnnualTrend) 
    HUC_LagAnnualTrend_watersheds.append(HUC_LagAnnualTrend) 
    HUC_PrainAnnualTrend_watersheds.append(HUC_PrainAnnualTrend) 
    
    HUC_IfracAnnualAvg_watersheds.append(HUC_IfracAnnualAvg)  
    HUC_EtaAnnualAvg_watersheds.append(HUC_EtaAnnualAvg) 
    HUC_LagAnnualAvg_watersheds.append(HUC_LagAnnualAvg) 
    HUC_PrainAnnualAvg_watersheds.append(HUC_PrainAnnualAvg) 
    
    

#%%
#want to obtain national averages for trends
    
import pandas as pd
    
ti=2
Ifrac_percent_avg=[]
Prain_percent_avg=[]
Eta_percent_avg=[]

Ifrac_unit_avg=[]
Prain_unit_avg=[]
Eta_unit_avg=[]
Lag_unit_avg=[]
WatershedName =[]
WatershedArea =[]

Ifrac_inc_avg=[]
Prain_inc_avg=[]
Eta_inc_avg=[]
Lag_inc_avg=[]

Ifrac_dec_avg=[]
Prain_dec_avg=[]
Eta_dec_avg=[]
Lag_dec_avg=[]

Ifrac_inc_vect=[]
Prain_inc_vect=[]
Eta_inc_vect=[]
Lag_inc_vect=[]      
Ifrac_dec_vect=[]
Prain_dec_vect=[]
Eta_dec_vect=[]
Lag_dec_vect=[]          
Ifrac_none_vect=[]
Prain_none_vect=[]
Eta_none_vect=[]
Lag_none_vect=[]

PercentIfrac_inc=[]
PercentIfrac_dec=[]

for r,rec_all in enumerate(sf_HUC.records()):
    
    print(r,rec_all[11],HUC_Npts_watersheds[r])
    
    
    npoints = len(range(1948,2018)) #number of years (over 70 year period)
    
    if HUC_Npts_watersheds[r] > 0:
        Ifrac_trend70 =  np.asarray([HUC_IfracAnnualTrend_watersheds[r][i][ti]*npoints for i in range(0,HUC_Npts_watersheds[r])])
        Prain_trend70 = np.asarray([HUC_PrainAnnualTrend_watersheds[r][i][ti]*npoints for i in range(0,HUC_Npts_watersheds[r])])
        Eta_trend70 = np.asarray([HUC_EtaAnnualTrend_watersheds[r][i][ti]*npoints for i in range(0,HUC_Npts_watersheds[r])])
        Lag_trend70 = np.asarray([HUC_LagAnnualTrend_watersheds[r][i][ti]*npoints for i in range(0,HUC_Npts_watersheds[r])])
        
        Ifrac_avg= np.asarray([np.nanmean(HUC_IfracAnnualAvg_watersheds[r][i][ti]) for i in range(0,HUC_Npts_watersheds[r])])
        
        I1 = Ifrac_avg - Ifrac_trend70/float(2)
        I2 = Ifrac_avg + Ifrac_trend70/float(2)
        
        PercentIfrac = (I2/np.asfarray(I1) - 1)*100
        
        PercentIfrac_inc.append(np.nanmean(PercentIfrac[Ifrac_trend70>0]))
        PercentIfrac_dec.append(np.nanmean(PercentIfrac[Ifrac_trend70<0]))
        
        Ifrac_inc_avg.append(np.nanmean(Ifrac_trend70[Ifrac_trend70>0]))
        Prain_inc_avg.append(np.nanmean(Prain_trend70[Prain_trend70>0]*365))       
        Eta_inc_avg.append(np.nanmean(Eta_trend70[Eta_trend70>0]))        
        Lag_inc_avg.append(np.nanmean(Lag_trend70[Lag_trend70>0]))
        
        Ifrac_dec_avg.append(np.nanmean(Ifrac_trend70[Ifrac_trend70<0]))
        Prain_dec_avg.append(np.nanmean(Prain_trend70[Prain_trend70<0]*365))       
        Eta_dec_avg.append(np.nanmean(Eta_trend70[Eta_trend70<0]))        
        Lag_dec_avg.append(np.nanmean(Lag_trend70[Lag_trend70<0]))
        
        Ifrac_unit_avg.append(np.nanmean(Ifrac_trend70))
        Prain_unit_avg.append(np.nanmean(Prain_trend70)*365)
        Eta_unit_avg.append(np.nanmean(Eta_trend70))
        Lag_unit_avg.append(np.nanmean(Lag_trend70))
        

        #fractions of watershed area showing each trend direction (+,0,-)
        Ifrac_inc_vect.append(np.count_nonzero(Ifrac_trend70>0)/np.size(Ifrac_trend70))
        Prain_inc_vect.append(np.count_nonzero(Prain_trend70>0)/np.size(Ifrac_trend70))
        Eta_inc_vect.append(np.count_nonzero(Eta_trend70>0)/np.size(Ifrac_trend70))
        Lag_inc_vect.append(np.count_nonzero(Lag_trend70>0)/np.size(Ifrac_trend70))           
        Ifrac_dec_vect.append(np.count_nonzero(Ifrac_trend70<0)/np.size(Ifrac_trend70))
        Prain_dec_vect.append(np.count_nonzero(Prain_trend70<0)/np.size(Ifrac_trend70))
        Eta_dec_vect.append(np.count_nonzero(Eta_trend70<0)/np.size(Ifrac_trend70))
        Lag_dec_vect.append(np.count_nonzero(Lag_trend70<0) /np.size(Ifrac_trend70))            
        Ifrac_none_vect.append(np.count_nonzero(Ifrac_trend70==0)/np.size(Ifrac_trend70))
        Prain_none_vect.append(np.count_nonzero(Prain_trend70==0)/np.size(Ifrac_trend70))
        Eta_none_vect.append(np.count_nonzero(Eta_trend70==0)/np.size(Ifrac_trend70))
        Lag_none_vect.append(np.count_nonzero(Lag_trend70==0) /np.size(Ifrac_trend70))
    else:

        Ifrac_inc_avg.append(0)
        Prain_inc_avg.append(0)       
        Eta_inc_avg.append(0)        
        Lag_inc_avg.append(0)
        
        Ifrac_dec_avg.append(0)
        Prain_dec_avg.append(0)       
        Eta_dec_avg.append(0)        
        Lag_dec_avg.append(0)
        
        Ifrac_unit_avg.append(0)
        Prain_unit_avg.append(0)
        Eta_unit_avg.append(0)
        Lag_unit_avg.append(0)
        
        Ifrac_inc_vect.append(0)
        Prain_inc_vect.append(0)
        Eta_inc_vect.append(0)
        Lag_inc_vect.append(0)        
        Ifrac_dec_vect.append(0)
        Prain_dec_vect.append(0)
        Eta_dec_vect.append(0)
        Lag_dec_vect.append(0)          
        Ifrac_none_vect.append(0)
        Prain_none_vect.append(0)
        Eta_none_vect.append(0)
        Lag_none_vect.append(0)
        
        PercentIfrac_inc.append(0)
        PercentIfrac_dec.append(0)
        
        
    WatershedName.append(rec_all[11])
    WatershedArea.append(HUC_Npts_watersheds[r])
    
       
#save in excel spreadsheet:
    
WatershedArea = WatershedArea/np.sum(WatershedArea)
alltrends_df = pd.DataFrame({'WatershedName':WatershedName,'WatershedAreaFrac':WatershedArea, 
                             'Ifrac_inc_avg':Ifrac_inc_avg,'Ifrac_dec_avg':Ifrac_dec_avg, 'Ifrac_unit_avg':Ifrac_unit_avg,
                             'Ifrac_frac_inc':Ifrac_inc_vect,'Ifrac_frac_dec':Ifrac_dec_vect,'Ifrac_frac_none':Ifrac_none_vect,
                             'Prain_inc_avg':Prain_inc_avg, 'Prain_dec_avg':Prain_dec_avg,'Prain_unit_avg':Prain_unit_avg,
                             'Prain_frac_inc':Prain_inc_vect,'Prain_frac_dec':Prain_dec_vect,'Prain_frac_none':Prain_none_vect,
                             'Lag_inc_avg':Lag_inc_avg, 'Lag_dec_avg':Lag_dec_avg, 'Lag_unit_avg':Lag_unit_avg,
                             'Lag_frac_inc':Lag_inc_vect,'Lag_frac_dec':Lag_dec_vect,'Lag_frac_none':Lag_none_vect,
                             'Eta_inc_avg':Eta_inc_avg, 'Eta_dec_avg':Eta_dec_avg, 'Eta_unit_avg':Eta_unit_avg,
                             'Eta_frac_inc':Eta_inc_vect,'Eta_frac_dec':Eta_dec_vect,'Eta_frac_none':Eta_none_vect,
                             'Percent_Ifrac_inc':PercentIfrac_inc,'Percent_Ifrac_dec':PercentIfrac_dec})     
    
writer = pd.ExcelWriter('MEANTRENDS_HUC2_50thprctile.xlsx',engine='xlsxwriter')
alltrends_df.to_excel(writer,sheet_name='Sheet1')  
writer.save()           
              
              #"{:.2f}".format(Lag_percent), "{:.2f}".format(Eta_percent))
        
        
        #also want to consider in terms of % increases per year
        
    

#%%



##
if make_shapefiles ==1:
    for ti,t in enumerate(threshval):
        for s in range(0,4):
    
            
            newshape = shapefile.Writer()
            newshape.fields = list(sf_HUC.fields)
            
            newshape.field("Ifrac","N",40,10)
            newshape.field("Prain","N",40,10)
            newshape.field("Lag","N",40,10)
            newshape.field("Eta","N",40,10)
            
            newshape.field("Ifrac_std","N",40,10)
            newshape.field("Prain_std","N",40,10)
            newshape.field("Lag_std","N",40,10)
            newshape.field("Eta_std","N",40,10)
            
            newshape.field("Ifrac_max","N",40,10)
            newshape.field("Prain_max","N",40,10)
            newshape.field("Lag_max","N",40,10)
            newshape.field("Eta_max","N",40,10)
            
            newshape.field("Ifrac_min","N",40,10)
            newshape.field("Prain_min","N",40,10)
            newshape.field("Lag_min","N",40,10)
            newshape.field("Eta_min","N",40,10)
            
            newshape.field("Npoints","N",40,10)
            
            for r,rec_all in enumerate(sf_HUC.records()):
                
                if HUC_Npts_watersheds[r] > 0:
                    Ifrac =  [HUC_Ifrac_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    Prain = [HUC_Prain_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    Eta = [HUC_Eta_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    Lag = [HUC_Lag_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    
                    Ifrac_avg = np.nanmean(Ifrac)
                    Prain_avg = np.nanmean(Prain)
                    Eta_avg = np.nanmean(Eta)
                    Lag_avg = np.nanmean(Lag)               
                    Ifrac_std = np.nanstd(Ifrac)
                    Prain_std = np.nanstd(Prain)
                    Eta_std = np.nanstd(Eta)
                    Lag_std = np.nanstd(Lag)                
                    Ifrac_min = np.nanpercentile(Ifrac,10)
                    Prain_min = np.nanpercentile(Prain,10)
                    Eta_min = np.nanpercentile(Eta,10)
                    Lag_min = np.nanpercentile(Lag,10)                
                    Ifrac_max = np.nanpercentile(Ifrac,90)
                    Prain_max = np.nanpercentile(Prain,90)
                    Eta_max = np.nanpercentile(Eta,90)
                    Lag_max = np.nanpercentile(Lag,90)                
    
                else:              
                    Ifrac_avg = 0
                    Prain_avg = 0
                    Eta_avg = 0
                    Lag_avg = 0                
                    Ifrac_std = 0
                    Prain_std = 0
                    Eta_std = 0
                    Lag_std = 0                
                    Ifrac_min = 0
                    Prain_min = 0
                    Eta_min =0
                    Lag_min = 0                
                    Ifrac_max = 0
                    Prain_max = 0
                    Eta_max =0
                    Lag_max =0  
                    
      
                rec_all.append(Ifrac_avg)
                rec_all.append(Prain_avg)
                rec_all.append(Lag_avg)
                rec_all.append(Eta_avg)
                
                rec_all.append(Ifrac_std)
                rec_all.append(Prain_std)
                rec_all.append(Lag_std)
                rec_all.append(Eta_std)
                
                rec_all.append(Ifrac_max)
                rec_all.append(Prain_max)
                rec_all.append(Lag_max)
                rec_all.append(Eta_max)
                
                rec_all.append(Ifrac_min)
                rec_all.append(Prain_min)
                rec_all.append(Lag_min)
                rec_all.append(Eta_min)  
                
                rec_all.append(HUC_Npts_watersheds[r])
                    
                newshape.records.append(rec_all)
    
            print ("saving avg info for season ",s+1, "of 4 seasons, threshold", t)
            newshape._shapes.extend(sf_HUC.shapes())    
            newshape.save(stringlist_overall[ti][s])
                
    
    #%% trend shape files - seasonal
    for ti,t in enumerate(threshval):
        for s in range(0,4):
    
            
            newshape = shapefile.Writer()
            newshape.fields = list(sf_HUC.fields)
            
            newshape.field("IfracTrend","N",40,10)
            newshape.field("PrainTrend","N",40,10)
            newshape.field("LagTrend","N",40,10)
            newshape.field("EtaTrend","N",40,10)
            
            newshape.field("IfracPctInc","N",40,15)
            newshape.field("PrainPctInc","N",40,15)
            newshape.field("LagPctInc","N",40,10)
            newshape.field("EtaPctInc","N",40,10)
            
            newshape.field("IfracPctDec","N",40,15)
            newshape.field("PrainPctDec","N",40,15)
            newshape.field("LagPctDec","N",40,10)
            newshape.field("EtaPctDec","N",40,10)
            
            newshape.field("IfracPctNo","N",40,10)
            newshape.field("PrainPctNo","N",40,10)
            newshape.field("LagPctNo","N",40,10)
            newshape.field("EtaPctNo","N",40,10)
            
            newshape.field("Npoints","N",40,10)
            
            for r,rec_all in enumerate(sf_HUC.records()):
                
                print(r)
                
                if HUC_Npts_watersheds[r] > 0:
                    Ifrac =  [HUC_IfracTrend_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    Prain = [HUC_PrainTrend_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    Eta = [HUC_EtaTrend_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    Lag = [HUC_LagTrend_watersheds[r][i][s][ti] for i in range(0,HUC_Npts_watersheds[r])]
                    
                    Ifrac_avg = np.nanmean(Ifrac)
                    Prain_avg = np.nanmean(Prain)
                    Eta_avg = np.nanmean(Eta)
                    Lag_avg = np.nanmean(Lag)               
                    Ifrac_inc = np.count_nonzero(np.asarray(Ifrac)>0)/np.size(Ifrac)
                    Prain_inc = np.count_nonzero(np.asarray(Prain)>0)/np.size(Ifrac)
                    Eta_inc = np.count_nonzero(np.asarray(Eta)>0)/np.size(Ifrac)
                    Lag_inc =  np.count_nonzero(np.asarray(Lag)>0)/np.size(Ifrac)             
                    Ifrac_dec = np.count_nonzero(np.asarray(Ifrac)<0)/np.size(Ifrac)
                    Prain_dec = np.count_nonzero(np.asarray(Prain)<0)/np.size(Ifrac)
                    Eta_dec = np.count_nonzero(np.asarray(Eta)<0)/np.size(Ifrac)
                    Lag_dec =  np.count_nonzero(np.asarray(Lag)<0) /np.size(Ifrac)             
                    Ifrac_none = np.count_nonzero(np.asarray(Ifrac)==0)/np.size(Ifrac)
                    Prain_none = np.count_nonzero(np.asarray(Prain)==0)/np.size(Ifrac)
                    Eta_none = np.count_nonzero(np.asarray(Eta)==0)/np.size(Ifrac)
                    Lag_none =  np.count_nonzero(np.asarray(Lag)==0) /np.size(Ifrac)  
                    
                    #if over 50% of pixels in watershed have no trend - set to no trend (0 values)
                    if Ifrac_none > 0.5:
                        Ifrac_avg = 0
                    elif np.max([Ifrac_inc, Ifrac_dec])< 0.5: #no dominant trend
                        Ifrac_avg = 0
                        
                    if Prain_none > 0.5:
                        Prain_avg = 0
                    elif np.max([Prain_inc, Prain_dec])< 0.5: #no dominant trend
                        Prain_avg = 0
                        
                    if Eta_none > 0.5:
                        Eta_avg = 0
                    elif np.max([Eta_inc, Eta_dec])< 0.5: #no dominant trend
                        Eta_avg = 0
                        
                    if Lag_none > 0.5:
                        Lag_avg =0
                    elif np.max([Lag_inc, Lag_dec])< 0.5: #no dominant trend
                        Lag_avg = 0
                    
    #                print(Ifrac)
    #                print(np.count_nonzero(np.asarray(Ifrac)>0))
    #                print(Ifrac_inc,Ifrac_dec,Ifrac_none)
    
                else:              
                    Ifrac_avg = 0
                    Prain_avg = 0
                    Eta_avg = 0
                    Lag_avg = 0                
                    Ifrac_inc = 0
                    Prain_inc = 0
                    Eta_inc = 0
                    Lag_inc = 0                
                    Ifrac_dec = 0
                    Prain_dec = 0
                    Eta_dec =0
                    Lag_dec = 0                
                    Ifrac_none = 0
                    Prain_none = 0
                    Eta_none =0
                    Lag_none =0  
                    
      
                rec_all.append(Ifrac_avg)
                rec_all.append(Prain_avg)
                rec_all.append(Lag_avg)
                rec_all.append(Eta_avg)
                
                rec_all.append(Ifrac_inc)
                rec_all.append(Prain_inc)
                rec_all.append(Lag_inc)
                rec_all.append(Eta_inc)
                
                rec_all.append(Ifrac_dec)
                rec_all.append(Prain_dec)
                rec_all.append(Lag_dec)
                rec_all.append(Eta_dec)
                
                rec_all.append(Ifrac_none)
                rec_all.append(Prain_none)
                rec_all.append(Lag_none)
                rec_all.append(Eta_none)  
                
                rec_all.append(HUC_Npts_watersheds[r])
                    
                newshape.records.append(rec_all)
    
            print ("saving avg trend info for season ",s+1, "of 4 seasons, threshold", t)
            newshape._shapes.extend(sf_HUC.shapes())    
            newshape.save(stringlist_trends[ti][s])
                
    #%% annual trends    
    for ti,t in enumerate(threshval):
         
        newshape = shapefile.Writer()
        newshape.fields = list(sf_HUC.fields)
        
        newshape.field("IfracTrend","N",40,10)
        newshape.field("PrainTrend","N",40,10)
        newshape.field("LagTrend","N",40,10)
        newshape.field("EtaTrend","N",40,10)
        
        newshape.field("IfracPctInc","N",40,15)
        newshape.field("PrainPctInc","N",40,15)
        newshape.field("LagPctInc","N",40,10)
        newshape.field("EtaPctInc","N",40,10)
        
        newshape.field("IfracPctDec","N",40,15)
        newshape.field("PrainPctDec","N",40,15)
        newshape.field("LagPctDec","N",40,10)
        newshape.field("EtaPctDec","N",40,10)
        
        newshape.field("IfracPctNo","N",40,10)
        newshape.field("PrainPctNo","N",40,10)
        newshape.field("LagPctNo","N",40,10)
        newshape.field("EtaPctNo","N",40,10)
        
        newshape.field("Npoints","N",40,10)
        
        for r,rec_all in enumerate(sf_HUC.records()):
            
            print(r)
            
            if HUC_Npts_watersheds[r] > 0:
                Ifrac =  [HUC_IfracAnnualTrend_watersheds[r][i][ti] for i in range(0,HUC_Npts_watersheds[r])]
                Prain = [HUC_PrainAnnualTrend_watersheds[r][i][ti] for i in range(0,HUC_Npts_watersheds[r])]
                Eta = [HUC_EtaAnnualTrend_watersheds[r][i][ti] for i in range(0,HUC_Npts_watersheds[r])]
                Lag = [HUC_LagAnnualTrend_watersheds[r][i][ti] for i in range(0,HUC_Npts_watersheds[r])]
                
                Ifrac_avg = np.nanmean(Ifrac)
                Prain_avg = np.nanmean(Prain)
                Eta_avg = np.nanmean(Eta)
                Lag_avg = np.nanmean(Lag)               
                Ifrac_inc = np.count_nonzero(np.asarray(Ifrac)>0)/np.size(Ifrac)
                Prain_inc = np.count_nonzero(np.asarray(Prain)>0)/np.size(Ifrac)
                Eta_inc = np.count_nonzero(np.asarray(Eta)>0)/np.size(Ifrac)
                Lag_inc =  np.count_nonzero(np.asarray(Lag)>0)/np.size(Ifrac)             
                Ifrac_dec = np.count_nonzero(np.asarray(Ifrac)<0)/np.size(Ifrac)
                Prain_dec = np.count_nonzero(np.asarray(Prain)<0)/np.size(Ifrac)
                Eta_dec = np.count_nonzero(np.asarray(Eta)<0)/np.size(Ifrac)
                Lag_dec =  np.count_nonzero(np.asarray(Lag)<0) /np.size(Ifrac)             
                Ifrac_none = np.count_nonzero(np.asarray(Ifrac)==0)/np.size(Ifrac)
                Prain_none = np.count_nonzero(np.asarray(Prain)==0)/np.size(Ifrac)
                Eta_none = np.count_nonzero(np.asarray(Eta)==0)/np.size(Ifrac)
                Lag_none =  np.count_nonzero(np.asarray(Lag)==0) /np.size(Ifrac)  
                
                #if over 50% of pixels in watershed have no trend - set to no trend (0 values)
                if Ifrac_none > 0.5:
                    Ifrac_avg = 0
                elif np.max([Ifrac_inc, Ifrac_dec])< 0.5: #no dominant trend
                    Ifrac_avg = 0
                    
                if Prain_none > 0.5:
                    Prain_avg = 0
                elif np.max([Prain_inc, Prain_dec])< 0.5: #no dominant trend
                    Prain_avg = 0
                    
                if Eta_none > 0.5:
                    Eta_avg = 0
                elif np.max([Eta_inc, Eta_dec])< 0.5: #no dominant trend
                    Eta_avg = 0
                    
                if Lag_none > 0.5:
                    Lag_avg =0
                elif np.max([Lag_inc, Lag_dec])< 0.5: #no dominant trend
                    Lag_avg = 0
                
    #                print(Ifrac)
    #                print(np.count_nonzero(np.asarray(Ifrac)>0))
    #                print(Ifrac_inc,Ifrac_dec,Ifrac_none)
    
            else:              
                Ifrac_avg = 0
                Prain_avg = 0
                Eta_avg = 0
                Lag_avg = 0                
                Ifrac_inc = 0
                Prain_inc = 0
                Eta_inc = 0
                Lag_inc = 0                
                Ifrac_dec = 0
                Prain_dec = 0
                Eta_dec =0
                Lag_dec = 0                
                Ifrac_none = 0
                Prain_none = 0
                Eta_none =0
                Lag_none =0  
                
      
            rec_all.append(Ifrac_avg)
            rec_all.append(Prain_avg)
            rec_all.append(Lag_avg)
            rec_all.append(Eta_avg)
            
            rec_all.append(Ifrac_inc)
            rec_all.append(Prain_inc)
            rec_all.append(Lag_inc)
            rec_all.append(Eta_inc)
            
            rec_all.append(Ifrac_dec)
            rec_all.append(Prain_dec)
            rec_all.append(Lag_dec)
            rec_all.append(Eta_dec)
            
            rec_all.append(Ifrac_none)
            rec_all.append(Prain_none)
            rec_all.append(Lag_none)
            rec_all.append(Eta_none)  
            
            rec_all.append(HUC_Npts_watersheds[r])
                
            newshape.records.append(rec_all)
    
        print ("saving avg annual trend info, threshold", t)
        newshape._shapes.extend(sf_HUC.shapes())    
        newshape.save(stringlist_annualtrends[ti])        
        
        

       
#%% ################################## plots #########################################
        
if make_plots ==1:        
#Create plots for different seasons - works better with HUC2 data

    min, max = (-125, -65)
    step = 10
    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    levels = range(min,max,step)
    CS3 = plt.contourf(Z, levels, cmap='viridis_r')
    plt.clf()
    
    stringlist = ["DJF","MAM","JJA","SON"]
    c = [(1,0,0),(0,1,0),(0,0,1),(.5,0,.5)]
         
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plt.tight_layout()
    plt.hold
    
    xvect =np.arange(0,1,.001)
    yvect = -xvect*np.log2(xvect) - (1-xvect)*np.log2(1-xvect)
    ax1.plot(xvect,yvect,'-k')
    ax2.plot(xvect,yvect,'-k')
    ax3.plot(xvect,yvect,'-k')
    ax4.plot(xvect,yvect,'-k') 
    
    cmap = mpl.cm.get_cmap('viridis')
    
        
    legendnames=[]
    for r,rec_all in enumerate(sf_HUC.records()):
        
        if HUC_Npts_watersheds[r] > 0:
            Ifrac_mean = np.zeros((4))
            Prain_mean = np.zeros((4))
            Eta_mean = np.zeros((4))
            Lag_mean = np.zeros((4))
            
            for s in range(0,4):
                Ifrac =  [HUC_Ifrac_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Prain = [HUC_Prain_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Eta = [HUC_Eta_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Lag = [HUC_Lag_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Ifrac_mean[s]= np.average(Ifrac)
                Prain_mean[s] = np.average(Prain)
                Lag_mean[s] = np.average(Lag)
                Eta_mean[s] = np.average(Eta)
                

        #get name of watershed, match to names in dataset
        num_watershed = rec_all[10]
        name_watershed = rec_all[11]
        
        longitude = HUC_Longitude_watersheds[r]
        
        print(name_watershed, longitude)
        
        longitude_ratio = (np.abs(longitude)-65)/(125-65)

        Hval = -Prain_mean*np.log2(Prain_mean) - (1-Prain_mean)*np.log2(1-Prain_mean)
        
        Itot_mean = Hval * Ifrac_mean
           
        ax1.plot(Prain_mean[0],Hval[0],'.',color=cmap(longitude_ratio),markersize=12)
        ax2.plot(Prain_mean[1],Hval[1],'.',color=cmap(longitude_ratio),markersize=12)
        ax3.plot(Prain_mean[2],Hval[2],'.',color=cmap(longitude_ratio),markersize=12)
        ax4.plot(Prain_mean[3],Hval[3],'.',color=cmap(longitude_ratio),markersize=12)
        
#        ax1.plot(Prain_mean[0],Itot_mean[0],'.',color=cmap(longitude_ratio),markersize=12)
#        ax2.plot(Prain_mean[1],Itot_mean[1],'.',color=cmap(longitude_ratio),markersize=12)
#        ax3.plot(Prain_mean[2],Itot_mean[2],'.',color=cmap(longitude_ratio),markersize=12)
#        ax4.plot(Prain_mean[3],Itot_mean[3],'.',color=cmap(longitude_ratio),markersize=12)
        
        l = mpl.lines.Line2D([Prain_mean[0],Prain_mean[0]], [0.0, Itot_mean[0]],color=cmap(longitude_ratio))
        ax1.add_line(l)
        l = mpl.lines.Line2D([Prain_mean[1],Prain_mean[1]], [0.0,Itot_mean[1]],color=cmap(longitude_ratio))
        ax2.add_line(l)
        l = mpl.lines.Line2D([Prain_mean[2],Prain_mean[2]], [0.0, Itot_mean[2]],color=cmap(longitude_ratio))
        ax3.add_line(l)
        l = mpl.lines.Line2D([Prain_mean[3],Prain_mean[3]], [0.0, Itot_mean[3]],color=cmap(longitude_ratio))
        ax4.add_line(l)
    
    ax1.set_title('DJF')
    ax2.set_title('MAM')
    ax3.set_title('JJA')
    ax4.set_title('SON') 
    
    ax1.set_ylabel('H(PPT)')
    ax3.set_ylabel('H(PPT)')
    ax3.set_xlabel('P(PPT=1)')
    ax4.set_xlabel('P(PPT=1)')
    
    ax1.set_xlim(-.05, 1.05)   
    ax1.set_ylim(-.05, 1.05)          
    ax2.set_xlim(-.05, 1.05)   
    ax2.set_ylim(-.05, 1.05)  
    ax3.set_xlim(-.05, 1.05)   
    ax3.set_ylim(-.05, 1.05)  
    ax4.set_xlim(-.05, 1.05)   
    ax4.set_ylim(-.05, 1.05)
    
    cax = fig.add_axes([0.27, -.1, 0.5, 0.05])
    fig.colorbar(CS3,cax=cax, orientation='horizontal')
    
    
    plt.savefig('SeasonalEntropy.pdf',format='pdf')
    

    
    min, max = (-125, -65)
    step = 10
    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    levels = range(min,max,step)
    CS3 = plt.contourf(Z, levels, cmap='viridis_r')
    plt.clf()
    
    stringlist = ["DJF","MAM","JJA","SON"]
    c = [(1,0,0),(0,1,0),(0,0,1),(.5,0,.5)]
         
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plt.tight_layout()
    plt.hold
    
    
    cmap = mpl.cm.get_cmap('viridis')
    
        
    legendnames=[]
    for r,rec_all in enumerate(sf_HUC.records()):
        
        if HUC_Npts_watersheds[r] > 0:
            Ifrac_mean = np.zeros((4))
            Prain_mean = np.zeros((4))
            Eta_mean = np.zeros((4))
            Lag_mean = np.zeros((4))
            
            for s in range(0,4):
                Ifrac =  [HUC_Ifrac_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Prain = [HUC_Prain_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Eta = [HUC_Eta_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Lag = [HUC_Lag_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Ifrac_mean[s]= np.average(Ifrac)
                Prain_mean[s] = np.average(Prain)
                Lag_mean[s] = np.average(Lag)
                Eta_mean[s] = np.average(Eta)
                

        #get name of watershed, match to names in dataset
        num_watershed = rec_all[10]
        name_watershed = rec_all[11]
        
        longitude = HUC_Longitude_watersheds[r]
        
        print(name_watershed, longitude)
        
        longitude_ratio = (np.abs(longitude)-65)/(125-65)

        Hval = -Prain_mean*np.log2(Prain_mean) - (1-Prain_mean)*np.log2(1-Prain_mean)
        
        Itot_mean = Hval * Ifrac_mean
           
        
#        ax1.plot(Prain_mean[0],Eta_mean[0]*Itot_mean[0],'.',color=cmap(longitude_ratio),markersize=12)
#        ax2.plot(Prain_mean[1],Eta_mean[1]*Itot_mean[1],'.',color=cmap(longitude_ratio),markersize=12)
#        ax3.plot(Prain_mean[2],Eta_mean[2]*Itot_mean[2],'.',color=cmap(longitude_ratio),markersize=12)
#        ax4.plot(Prain_mean[3],Eta_mean[3]*Itot_mean[3],'.',color=cmap(longitude_ratio),markersize=12)
        
        l = mpl.lines.Line2D([Prain_mean[0],Prain_mean[0]], [0.0, Eta_mean[0]],color=cmap(longitude_ratio))
        ax1.add_line(l)
        l = mpl.lines.Line2D([Prain_mean[1],Prain_mean[1]], [0.0,Eta_mean[1]],color=cmap(longitude_ratio))
        ax2.add_line(l)
        l = mpl.lines.Line2D([Prain_mean[2],Prain_mean[2]], [0.0, Eta_mean[2]],color=cmap(longitude_ratio))
        ax3.add_line(l)
        l = mpl.lines.Line2D([Prain_mean[3],Prain_mean[3]], [0.0, Eta_mean[3]],color=cmap(longitude_ratio))
        ax4.add_line(l)
    
    ax1.set_title('DJF')
    ax2.set_title('MAM')
    ax3.set_title('JJA')
    ax4.set_title('SON') 
    
    ax1.set_ylabel('Eta')
    ax3.set_ylabel('Eta')
    ax3.set_xlabel('P(PPT=1)')
    ax4.set_xlabel('P(PPT=1)')
    
    ax1.set_xlim(0, .6)   
    ax1.set_ylim(0, .3)          
    ax2.set_xlim(0, .6)   
    ax2.set_ylim(0, .3)  
    ax3.set_xlim(0, .6)   
    ax3.set_ylim(0, .3)  
    ax4.set_xlim(0, .6)   
    ax4.set_ylim(0, .3)
    
    
    plt.savefig('SeasonalEta.pdf',format='pdf') 
    
    

#%%
################################### Part II plots #########################################
    #Create plots for different seasons - works better with HUC2 data
    stringlist = ["DJF","MAM","JJA","SON"]
    c = [(1,0,0),(0,1,0),(0,0,1),(.5,0,.5)]
         
    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    plt.tight_layout()
    plt.hold
    
    cmap = mpl.cm.get_cmap('viridis')
        
    legendnames=[]
    leglist=[]
    plot_lines =[]
    lines_lines=[]
    
    newshape = shapefile.Writer()
    newshape.fields = list(sf_HUC.fields)
        
    newshape.field("Longitude","N",40,10)
    
    
    for r,rec_all in enumerate(sf_HUC.records()):
        
        if HUC_Npts_watersheds[r] > 0:
            Ifrac_mean = np.zeros((4))
            Itot_mean = np.zeros((4))
            Prain_mean = np.zeros((4))
            Eta_mean = np.zeros((4))
            Lag_mean = np.zeros((4))
            
            for s in range(0,4):
                Ifrac =  [HUC_Ifrac_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Itot =  [HUC_Itot_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Prain = [HUC_Prain_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Eta = [HUC_Eta_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Lag = [HUC_Lag_watersheds[r][i][s][0] for i in range(0,HUC_Npts_watersheds[r])]
                Ifrac_mean[s]= np.average(Ifrac)
                Itot_mean[s]=np.average(Itot)
                Prain_mean[s] = np.average(Prain)
                Lag_mean[s] = np.average(Lag)
                Eta_mean[s] = np.average(Eta)
                

        #get name of watershed, match to names in dataset
        num_watershed = rec_all[10]
        name_watershed = rec_all[11]
        
        longitude = HUC_Longitude_watersheds[r]

        longitude_ratio = (np.abs(longitude)-65)/(125-65)

        
        Hval = -Prain_mean*np.log2(Prain_mean) - (1-Prain_mean)*np.log2(1-Prain_mean)
        
        #set markersize as a function of Etamean
        msize = Eta_mean*100
        
        #if float(num_watershed) in [1.,2.,10.,15.,17.,18.]:
        #ll,=ax1.plot(Hval,Ifrac_mean,color=cmap(longitude_ratio))
    
        s1,=ax1.plot(Hval[0],Itot_mean[0],'d',color=cmap(longitude_ratio),markersize=msize[0])
        s2,=ax1.plot(Hval[1],Itot_mean[1],'s',color=cmap(longitude_ratio),markersize=msize[1])
        s3,=ax1.plot(Hval[2],Itot_mean[2],'o',color=cmap(longitude_ratio),markersize=msize[2])
        s4,=ax1.plot(Hval[3],Itot_mean[3],'v',color=cmap(longitude_ratio),markersize=msize[3])
        legendnames.append(name_watershed)

        plot_lines.append([s1, s2, s3, s4])
        #lines_lines.append(ll)
        
        leglist.append(name_watershed)
        
        
        
        rec_all.append(longitude_ratio)
                
        newshape.records.append(rec_all)
    
#    print ("saving HUC2 longitude shapefile", t)
#    newshape._shapes.extend(sf_HUC.shapes())    
#    newshape.save("HUC2_shapefile_withlongitudes")     
    
    
    ax1.set_title('Itot vs H(PPT)')
    
    ax1.set_ylabel('Ifrac')
    ax1.set_xlabel('H(PPT)')  
    
    ax1.set_ylim((0,.25)) 
    
    legend1 = ax1.legend(plot_lines[0],['DJF','MAM','JJA','SON'],loc=3)
    #ax1.legend(lines_lines, leglist, loc=3)
    plt.gca().add_artist(legend1)
    
    
    plt.savefig('SeasonalHvsItot.pdf',format='pdf')


#%%
################ get matrix of combined trends ########  

          


    alltrendvals=[]
    allIfractrendvals=[]
    allLagtrendvals=[]
    for la_ind,la in enumerate(lat):
        if np.size(AllAnnualResults[la_ind])>1:
            trendvals = [AllAnnualResults[la_ind][i]['trendlist'][2] for 
                         i in range(0,np.size(AllAnnualResults[la_ind]))]
            
            Ifractrendvals = [AllAnnualResults[la_ind][i]['Trend_Ifrac'][2] for
                              i in range(0,np.size(AllAnnualResults[la_ind]))]
            
            Lagtrendvals = [AllAnnualResults[la_ind][i]['Trend_Lag'][2] for
                              i in range(0,np.size(AllAnnualResults[la_ind]))]
            
            alltrendvals.append(trendvals)
            allIfractrendvals.append(Ifractrendvals)
            allLagtrendvals.append(Lagtrendvals)

    alltrends = np.concatenate(alltrendvals)
    allIfractrends = np.concatenate(allIfractrendvals)
    allLagtrends = np.concatenate(allLagtrendvals)
    
    for a_ind,a in enumerate(alltrends):
        if a < -800:
            if allLagtrends[a_ind]>=0:
                alltrends[a_ind]=a+1000
                
    
    
    unique_categories,ind = np.unique(alltrends,return_index=True)
    
    unique_categories = unique_categories[~np.isnan(unique_categories)]
      
    counts =[]  
    for u_ind,u in enumerate(unique_categories):
       
        counts.append(np.sum(alltrends==u))
        
      
    counts = np.asfarray(counts)#/np.sum(counts)
    plt.bar(range(0,np.size(counts)),counts)
    
    #sort to find most common combinations
    order_cts = np.sort(counts)
    order_ind = np.argsort(counts)
    
    counts[order_ind[-5:]]
    unique_categories[order_ind[-5:]]
    
    xvals = [11, 10, 9, 1, 0, -1, -9, -10, -11]
    yvals = [1100, 1000, 900, 100, 0, -100, -900, -1000, -1100]
     
    matvalue = np.zeros((9,9))
    for x_ind, x in enumerate(xvals):
        for y_ind, y in enumerate(yvals):
            num = x+y
            indval = np.argwhere(unique_categories==num)
            if np.size(indval)>0:
                matvalue[x_ind,y_ind]=counts[indval]
    
    matvalue = matvalue/np.size(alltrends)
    
    fig, ax = plt.subplots()
    
    matfractions = (matvalue/np.sum(matvalue))*100
        
    ax.matshow(matfractions,cmap=plt.cm.Blues,vmin=0,vmax=20)
    
    for i in range(0,9):
        for j in range(0,9):
            c = matfractions[i,j]
            if c >0:
                #ax.text(j,i, str(c), va='center', ha='center')
                ax.text(j,i, "{:.1f}".format(c), va='center', ha='center')
                
#            if i==8:
#                totrow = np.sum(matfractions[:,j])
#                ax.text(j,9,"{:.1f}".format(totrow), va='center', ha='center')
#                
#            if j==8:
#                totrow = np.sum(matfractions[i,:])
#                ax.text(9,i,"{:.1f}".format(totrow), va='center', ha='center')
#    
    plt.savefig('MatrixOfWatersheds50prctile.pdf',format='pdf')
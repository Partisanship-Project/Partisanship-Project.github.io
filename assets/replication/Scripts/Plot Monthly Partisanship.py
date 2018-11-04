

import matplotlib.pyplot as plt
import pandas as pd

#import monthly data
with open('..\Results\MonthlyIdealPts_nb.csv','r') as f:
    df=pd.read_csv(f)

with open('..\Data\metadata.csv') as f:
    meta=pd.read_csv(f)    

meta.twitter = meta.twitter.replace(np.nan, '', regex=True)
meta['TwitterID'] = pd.Series([t.lower() for t in meta.twitter])
#dfs=df.join(meta)
dfs = pd.merge(df, meta, how='left', on=['TwitterID'])


##plot params
# min is -2.4 and max is 2.3
#need to calculate republican and democrat average each month
#need to set color of line to party of person
#need to label the axes

#for each person,
x=result.groupby(['party','Time_Label']).mean()
rep_df=x.zLkRatio.R
dem_df=x.zLkRatio.D
dem_series=pd.Series(dem_df.tolist(), index=[dt.datetime.strptime(d,'%b %Y') for d in dem_df.index])
rep_series=pd.Series(rep_df.tolist(), index=[dt.datetime.strptime(d,'%b %Y') for d in rep_df.index])

rep_col=(231/255.0, 18/255.0, 18/255.0)
dem_col=(18/255.0, 18/255.0, 231/255.0)
rep_col_all=(231/255.0, 18/255.0, 18/255.0,.4)
dem_col_all=(18/255.0, 18/255.0, 231/255.0,.4)


grouped = df.groupby('TwitterID')
i=0
for name, group in grouped:
#    i+=1
    print(name)
#    if i>10:
#        break
    ts = pd.Series(group.zLkRatio.tolist(), index=[dt.datetime.strptime(d,'%b %Y') for d in group.Time_Label])
    party=list(result[result.TwitterID==name].party)[0]
    if party =='D':
        col=dem_col
    elif party =='R':
        col=rep_col
    else:
        col=(128/255.0,128/255.0,128/255.0)
    fig=pd.DataFrame(dict(Republicans = rep_series, Democrats = dem_series, Candidate = ts)).plot(color=[col, dem_col_all,rep_col_all])
    #fig=ts.plot()
    #fig=dem_series.plot()
    #fig=rep_series.plot()
    fig.set_ylim(-2,2)
    fig.text(-.07, .95, 'Conservative',
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=14, color='red',
        transform=fig.transAxes)
    
    fig.text(-.07, .05, 'Liberal',
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=14, color='blue',
        transform=fig.transAxes)
        
        
    
    output = fig.get_figure()
    output.savefig('..\Images2\\'+name+'.jpg')#,figsize=(8, 6), dpi=80,)    #fig size sets image size in inches - 8in x 6in
    fig.clear()
    
    #create monthly plot

    #save plot as 200x200px jpg with ST-01-d.jpg name
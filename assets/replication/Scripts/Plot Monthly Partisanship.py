

import matplotlib.pyplot as plt
import pandas as pd

#import monthly data
with open('..\Results\MonthlyIdealPts_nb.csv','r') as f:
    df=pd.read_csv(f)
    
#for each person,
grouped = df.groupby('TwitterID')
i=0
for name, group in grouped:
    i+=1
    print(name)
    if i>10:
        break
    ts = pd.Series(group.zLkRatio.tolist(), index=[dt.datetime.strptime(d,'%b-%y') for d in group.Time_Label])

    fig=ts.plot()
    output = fig.get_figure()
    output.savefig('..\Images\\'+name+'.jpg')#,figsize=(8, 6), dpi=80,)    #fig size sets image size in inches - 8in x 6in
    fig.clear()
    
    #create monthly plot

    #save plot as 200x200px jpg with ST-01-d.jpg name
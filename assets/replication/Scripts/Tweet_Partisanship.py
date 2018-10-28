'''
This code estimates the ideal point for each tweet based on the twitter
ideal points estimated for each member of congress.
'''

import cPickle
import os
import numpy as np
import csv
import datetime as dt
from nltk import corpus 
import re
import random as rn

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf


def simple_tokenizer(text,stopwords='nltk'):
    '''
    function to tokenize tweets
    '''
    if stopwords=='nltk':
        stopwords=corpus.stopwords.words('english')
    else:
        stopwords=[]
    #print stopwords
    stopwords+=['bit','fb','ow','twitpic','ly','com','rt','http','tinyurl','nyti','www','co',
                'hlink','mnton','hshtg','mt','rt','amp','htt','rtwt']
    text = text.lower()
    text = re.findall(r"[\w'#@]+",text)
    res = []
    for w in text:
        if w not in stopwords and len(w)>2:
            res.append(w)
    return res

def getStopWords():
    stopwords=corpus.stopwords.words('english')
    stopwords+=['bit','fb','ow','twitpic','ly','com','rt','http','tinyurl','nyti','www','co',
                'hlink','mnton','hshtg','mt','rt','amp','htt','rtwt','https']
    return stopwords

def importMetaData():
    '''
    Import the ideal point data    
    '''
    
    print 'importing meta data'
    data={}
    scores={}
    fname='..\Results\Aggregate_Metadata_nb.csv'
    fname='../Results/MonthlyIdealPts_nb.csv'
    with open(fname) as f:
        dta=csv.reader(f)
        for i,line in enumerate(dta):
            if i>0:
                tid=line[2]
                #tid=line[0]
                row=line[0:9]
                ideal_point = line[len(line)-1]
                data[tid]=row+[ideal_point]
                scores[tid]=ideal_point         
    return data, scores


def MakeDateEssays(dates,metaData, indir=''):
    '''This function turns a directory (indir) composed of every individual's tweets each contained in a csv file (e.g. barackobama.csv)
    into a dictionary of {twitterID: 'parsed tweet1 parsed tweet2'}
    
    As a part of this data formating process, this code also produces counts of the total number of tweets and characters for each person
    returned as 'numTxt' and 'numWds' respectively.
    '''
    print 'importing tweets'    
    if indir=='':
        indir='G:\Congress\Twitter\Data'
    if dates==[]:
        dates=['1/01/2018','11/08/2018']
    begin=dt.datetime.strptime(dates[0],'%m/%d/%Y')
    end=dt.datetime.strptime(dates[1],'%m/%d/%Y')
    text={}
    for fn in os.listdir(indir):
        if fn.endswith('json'):
            continue
        tid=fn.split(".")[0].lower()
        if tid in metaData.keys():
            texts={}
            count=0
            with open(indir+'\\'+fn,'rb') as f:
                data=csv.reader(f)
                for i,line in enumerate(data):
                    if i>0:
                        time=line[3]
                        time=time.split(' ')[0]
                        time=dt.datetime.strptime(time,'%m/%d/%Y')# %a %b %d %H:%M:%S +0000 %Y').strftime('%m/%d/%Y %H:%M:%S')
                        if time>=begin and time<=end:
                            texts[line[6]]=line[0]
                        lastTime=time
            text[tid]=texts
    return text


def splitDevSample(text,scores, p=''):
    '''
    This code creates a subsample of tweets from each state to build the state classifier.
    Each state must have at least 5 members with at least 30 tweets (in total) or else it is excluded from classification
    The subsampling will draw half of the tweets from each of the members from the state, reserving the other half for
    estimation.
    '''
   
    Out={}
    In={}
    if p=='':
        p=.50
    
    Dev={}
    Out={}
    for handle,tweets in text.iteritems():
        Dev[handle]={}
        Out[handle]={}
        for tid, tweet in tweets.iteritems():
            if rn.random()<p:
                Dev[handle][tid]={scores[handle]:tweet}
            else:
                Out[handle][tid]={scores[handle]:tweet}
                
    
    with open('Partisan_Dev.pkl','wb') as fl:
        cPickle.dump(Dev,fl)
    with open('Partisan_holdout.pkl','wb') as fl:
        cPickle.dump(Out,fl)

    return Dev, Out
    

def Vectorize(texts, vectorizer=''):
    Tweets=[]
    labels=[]
    ids=[]
    for handle, scored_tweets in texts.iteritems():
        for tid,scores in scored_tweets.iteritems():
            for score,tweet in scores.iteritems():
                labels.append(score)
                Tweets.append(tweet)
                ids.append(tid)
    if vectorizer=='':
        vectorizer= tfidf(Tweets,ngram_range=(1,2),stop_words=getStopWords(),min_df=2,binary=True)   #the only real question I have with this is whether it ejects twitter-specific text (ie. @ or #)
        vectorizer=vectorizer.fit(Tweets)
        return vectorizer
    else:
        vec= vectorizer.transform(Tweets)
        labels=np.asarray(labels) 
        return vec,labels,vectorizer,ids

def FitRF(vec,labels,ids,n_estimators='',clf=''):
    print 'fitting the linear model'
    from sklearn.ensemble import RandomForestRegressor as rf  
    from sklearn.metrics import mean_squared_error as mserr
    if clf=='':
        clf=rf(n_estimators=n_estimators)
        clf.fit(vec,labels)
        return clf
    res={}
    prediction=clf.predict(vec)
    #report=Evaluate(labels,prediction,clf)
    mse=mserr(map(float,labels), prediction)
    print "MSE:", mse
    for idx in set(ids):
        res[idx]=[]
    for i,row in enumerate(prediction):
        res[ids[i]]=row
    return res,clf,mse


def makeDevSample(dates=[], p=.15):
    '''Function to import tweets and create the Dev and Holdout data '''
    if dates==[]: #default behavior is all tweets for 112.
        dates=['1/01/2018','11/08/2018']
    indir='G:\Congress\Twitter\Data'
    metaData,scores=importMetaData()
    texts=MakeDateEssays(dates,metaData, indir)  
    Dev,Out=splitDevSample(texts,scores,p)
    return

def TrainTestSplit(texts,sampler=.25):
    train_tweets=[]
    train_labels=[]
    train_ids=[]
    test_tweets,test_labels,test_ids=[],[],[]
    for handle, scored_tweets in texts.iteritems():
        for tweetid,scores in scored_tweets.iteritems():
            for score,tweet in scores.iteritems():
                if rn.random()<sampler:
                    train_labels.append(score)
                    train_tweets.append(tweet)
                    train_ids.append(tweetid)
                else:
                    test_labels.append(score)
                    test_tweets.append(tweet)
                    test_ids.append(tweetid)
    train_labels=np.asarray(train_labels)
    test_labels=np.asarray(test_labels)
    return train_tweets,train_labels,train_ids,test_tweets,test_labels,test_ids
    
def Report(res, texts):
    data={}
    for handle, scored_tweets in texts.iteritems():
        data[handle]=[]
        for tid,scores in scored_tweets.iteritems():
            for score,tweet in scores.iteritems():
                if tid in res:
                    data[handle].append([tid, res[tid], tweet])
                    
    outdir='G:/Congress/Twitter Ideology/Website/Partisan Scores/'
    out=[]
    for name,lines in data.iteritems():
        with open(outdir+name+'.txt','w') as fn:
            fn.write('tweetID,partisan_score,tweet \n')
            for line in lines:
                fn.write(','.join(map(str,line))+'\n')
        
        if len(lines)<10:
            out.append([name,'','',''])
        else:
            m=np.mean([line[1] for line in lines])
            var=np.std([line[1] for line in lines])
            out.append([name, m, var])
    
    with open('../Results/Tweet_Partisanship.csv','wb') as fl:
        writeit=csv.writer(fl)
        writeit.writerow(['TwitterID','tweetMean','tweetSd'])#,'tweetGini'])
        for line in out:
            writeit.writerow(line)
            
    return
 
def Train(train_tweets,train_labels,train_ids, n_estimators=''):
    print 'vectorizing texts'
    vectorizer= tfidf(train_tweets,ngram_range=(1,2),stop_words=getStopWords(),min_df=2,binary=True)   #the only real question I have with this is whether it ejects twitter-specific text (ie. @ or #)
    vectorizer=vectorizer.fit(train_tweets)
    vec= vectorizer.transform(train_tweets)   
    #clf= FitLM(vec,train_labels,train_ids,C,clf='')   #probably need to as.array(labels)
    clf= FitRF(vec,train_labels,train_ids,n_estimators,clf='')
    return vectorizer, vec, clf
    

def Test(vectorizer,clf, test_tweets,test_labels,test_ids):
    vec= vectorizer.transform(test_tweets)   
    #res,clf,report= FitLM(vec,test_labels,test_ids,C='',clf=clf)   #probably need to as.array(labels)
    res,clf,mse= FitRF(vec,test_labels,test_ids,'',clf)
    return res,clf,mse
    
def Run(develop=False,DevProb=.15,TrainProb=.3,dates=[]):
    '''
    function to use when creating the final, publishable data.
    Final MSE was .8229
    '''
    n=''
    if develop:
        makeDevSample(dates=[],p=DevProb) #create development set
        best=SelectModel()                    #find parameters for linear classifier
        n=best['rf']['n_estimators']
    with open('Partisan_holdout.pkl','rb') as fl:
        Out=cPickle.load(fl)
        print 'imported data'
        train_tweets,train_labels,train_ids,test_tweets,test_labels,test_ids=TrainTestSplit(Out,sampler=TrainProb)
        if n=='':
            n=70
        vectorizer, vectors, clf=Train(train_tweets,train_labels,train_ids,n_estimators=n)
        res,clf,mse=Test(vectorizer,clf, test_tweets,test_labels,test_ids)
        #topwords=TopWords(vectorizer,clf,n=30)
        Report(res,Out)
        #SaveWords(topwords)
    
    return

def SelectModel():
    '''
    
    IN THE ORIGINAL STUDY:
    ElasticNetCV shows alpha 2.2x10-5 and l1 of .5 are best. Minimium MSE found is .77
    LassoCV shows alpha 2.5x10-5 as best. Minimum MSE is .79
    Random forest shows more estimators is better (obviously), best score was .71 but that's not MSE, so can't benchmark
    SVM best score was .69 for c=10000, took 3 weeks to run in the dev sample. 
    
    '''
    out={}
    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor as rf
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV as gridcv
    
    vec,labels,ids=importDevelopmentSample()
    clf=ElasticNetCV()
    clf.fit(vec,labels)
    #alpha of 2.2x10-5, l1_ratio_ of .5
    print 'For ElasticNet'
    print '\t Chosen Alpha: ',clf.alpha_
    print '\t Chosen L1 Ratio: ', clf.l1_ratio_
    print '\t Minimum MSE was ', min([min(x) for x in clf.mse_path_])
    #print clf.score(vec,labels)
    out['elastic']={'alpha':clf.alpha_, 'l1_ratio':clf.l1_ratio_}
    
    #CV shows alpha of .00022 is best, but anything in the 5x10-4 or 1x10-5 range is pretty good
    #CV shows alpha of .000026
    clf=LassoCV()
    clf.fit(vec,labels)
    print '\t Chosen Alpha: ',clf.alpha_
    print '\t Minimum MSE was ', min([min(x) for x in clf.mse_path_])
    #print clf.score(vec,labels)
    out['lasso']={'alpha':clf.alpha_}
    

    parameters={'n_estimators':[5,10,20,40,70]}

    rf_clf=gridcv(rf(),parameters)
    rf_clf.fit(vec,labels)
    out['rf']=rf_clf.best_params_
    #Not running the more complex models that take forever for now. Just want to get some code running from start to finish.
    #tests=[
    #    ['gb',SVR(),{'C':np.logspace(-2, 10, 7)}  ["huber","squared_loss","epsilon_insensitive"]}],
    #    ['rf',rf(),{'n_estimators':[5,10,20,40,70]}],
    #]
    #for test in tests:
    #    name=test[0]
    #    model=test[1]
    #    parameters=test[2]
    #    clf,scores,params=GridSearch(vec,labels,model,parameters)
    #    print clf.grid_scores_
    #    print clf.best_params_
    #    out[name]=[clf,scores,params]
    #    out[name]=test[4]
    return out


def FindNonLinearModels():
    '''Unit Function to run all the potential models to find the ideal parameters
    
    NOTE: This should break now that I've changed the structure of In and Out

    results:
    svm - C
        1-5 - mean: 0.09147, std: 0.00041,
        1-4 - mean: 0.09147, std: 0.00041,
        1-3 - mean: 0.09766, std: 0.00090,
        .01 - mean: 0.12351, std: 0.00430,
        .3  - mean: 0.17737, std: 0.00857,
        3.6 - mean: 0.16453, std: 0.01467,
        46  - mean: 0.15741, std: 0.01202,
        600 - mean: 0.15741, std: 0.01136,
        7000- mean: 0.15385, std: 0.01188, 

    rf - n_estimators
        40: mean: 0.19208, std: 0.00948,
        5: mean: 0.17381, std: 0.01136,
        10: mean: 0.17907, std: 0.00672,
        20: mean: 0.18650, std: 0.00730,
        70: mean: 0.18960, std: 0.00545,
    gb -
        hinge - mean .18, sd = .012 ()
        log - mean: 0.15447, std: 0.00327, params: {'estimator__loss': 'log'}
        modified_huber - mean: 0.15106, std: 0.00740, params: {'estimator__loss': 'modified_huber'}
    '''
    from sklearn.ensemble import RandomForestRegressor as rf
    from sklearn.linear_model import SGDRegressor as sgd
    out={}
    
    print 'importing Sample'
    vec,labels,ids=importDevelopmentSample()
    
    out={}
    tests=[
        ['svm',svm.LinearSVC(multi_class='ovr'),{'estimator__C':np.logspace(-5, 5, 10)}],
        ['gb',sgd(),{'estimator__loss':["modified_huber","squared_loss"],'alpha':np.logspace(-5, 5, 10)}],
        ['rf',rf(),{'estimator__n_estimators':[5,10,20,40]}]
    ]
    
    for test in tests:
        print 'running ', test[0]
        name=test[0]
        model=test[1]
        parameters=test[2]
        clf,scores,params=GridSearch(vec,labels,model,parameters)
        out[name]=[clf,scores,params]
    return

def importDevelopmentSample():
    with open('Partisan_Dev.pkl','rb') as fl:
        Dev=cPickle.load(fl)
    print "Number of Tweets: ", len(Dev)
    vectorizer=Vectorize(Dev,'') #I use the same names so as not to save these vec/labels data
    vec,labels,vectorizer,ids=Vectorize(Dev, vectorizer=vectorizer)
    return vec, labels,ids


def NBTopWords(vec,clf,n=20):
    topWords={}
    feature_names =vec.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    for w in top:
        word=w[1]
        for j, f in enumerate(vec.get_feature_names()):
            if f==word:
                dem=np.exp(clf.feature_log_prob_[0][j])
                rep=np.exp(clf.feature_log_prob_[1][j])
                gen=1*(rep>dem)
                topWords[word]=[gen,rep,dem]
    return topWords

def getTwitterNames():
    names=[]
    for fn in os.listdir('C:\Users\Admin\Dropbox\Twitter_NaiveBayes\Data\Raw3'):
        names.append(fn.split('.')[0])
    return names

def getIndividualTweets(name,dates,typ='essay', indir=''):
    if indir=='':
        indir='Data\Raw3'
    begin=dt.datetime.strptime(dates[0],'%m/%d/%Y')
    end=dt.datetime.strptime(dates[1],'%m/%d/%Y')
    text={}
    broken=[]
    count=0
    times={}
    #for name in twitterIDs:
    for fn in os.listdir(indir):
        tid=fn.split(".")[0].lower()
        if tid == name:
            count+=1
            f=open(indir+'\\'+fn,'rb')
            data=csv.reader(f)
            if typ=='tweets':
                Texts={}
                for i,line in enumerate(data):
                    if i>0:
                        count+=1
                        time=line[3]
                        time=time.split(' ')[0]
                        time=dt.datetime.strptime(time,'%m/%d/%Y')# %a %b %d %H:%M:%S +0000 %Y').strftime('%m/%d/%Y %H:%M:%S')
                        if time>=begin and time<=end:
                            Texts[line[6]]=' '.join(Twitter_Tokenize(line[0],stopwords='nltk'))
                            times[line[6]]=time
            elif typ=='essay':
                texts=[]
                for i,line in enumerate(data):
                    if i>0:
                        count+=1
                        time=line[3]
                        time=time.split(' ')[0]
                        time=dt.datetime.strptime(time,'%m/%d/%Y')# %a %b %d %H:%M:%S +0000 %Y').strftime('%m/%d/%Y %H:%M:%S')
                        if time>=begin and time<=end:
                            texts.append(line[0])
                Texts=' '.join(Twitter_Tokenize(' '.join(texts),stopwords='nltk'))
                    
        
            #if typ=='essay':
            #    Texts=' '.join(Twitter_Tokenize(' '.join(texts),stopwords='nltk'))#generate feature space of tweet by eliminating stops and adding metatext features
            #elif typ=='tweets':
            #    for i,txt in enumerte(texts):
            #        text[tid]=Twitter_Tokenize(' '.join(txt),stopwords='nltk'))
            #else:
            #    Texts=' '.join(Twitter_Tokenize(' '.join(texts),stopwords='nltk'))#generate feature space of tweet by eliminating stops and adding metatext features
            #text[tid]=Texts
            #f.close()
    
    if count<1: #if the person is not in the Raw dataset from 113th congress
        return None,None
    return Texts, times
        

def writePerson(name,scoredict,times,tweets):
    out='G:/Research Data/2017-01-12 Backup/CongressTweets/PartisanScores2'
    with open(out+name+'.txt','w') as fn:
        fn.write('tweetID,mean,std,time,tweet \n')
        for tid,score in scoredict.iteritems():
            fn.write(', '.join([tid]+map(str,score)+[times[tid]]+[tweets[tid]])+'\n')
    return

def writeResults(Results):
    with open('PartisanTweets2.txt','w') as fn:
        fn.write('TwitterID,ProportionTweetsPartisan\n')
        for name, res in Results.iteritems():
            fn.write(name+', '+str(res)+'\n')
    return

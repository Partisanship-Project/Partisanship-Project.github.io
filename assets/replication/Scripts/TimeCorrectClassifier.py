
'''Notes: a few congress people are excluded in this analysis.

First, there were three duplicate IDs that weren't cleaned out from earlier
runs (this needs to be verified to ensure other previous scrapes are not still around - check pkl cache of tweet IDs.  If ID is in
directory, but not pkl files, then it is old duplicate and should be discarded).  They are:
['chuckschumer', '09/19/2013 14:40:11', 51],
['dannykdavis', '07/05/2009 17:27:21', 3],

Second, There were two incorrect IDs which has been corrected:
['repdanburton', 0, 1],
['jeanneshaheen', '03/11/2008 17:49:44', 174],

Third, others simply have >3200 tweets maxing out twitter's
individual dump parameters.  They are by name, earliest tweet, and total number:

'auctnr1', '01/18/2013 17:19:43', 3193], ['chakafattah', '05/10/2013 00:08:07', 3194], 
['danarohrabacher', '02/07/2013 05:44:12', 3228], ['darrellissa', '02/09/2013 14:49:06', 3235],
 ['johncornyn', '05/15/2013 11:39:01', 3204], 
['repkevinbrady', '04/06/2013 15:39:40', 3238], ['roslehtinen', '07/17/2013 19:40:56', 3196], 
['sensanders', '01/31/2013 18:20:21', 3240], ['speakerboehner', '01/24/2013 19:37:19', 3222]

Finally, others simply did not have tweets in the date range:
['repguthrie', '09/26/2013 16:25:17', 78], 
['repinsleenews', '04/23/2009 19:38:47', 59],
['repjohnduncanjr', '09/20/2013 17:49:40', 83],
['senatorisakson', '05/02/2013 15:34:39', 296],
'''



import datetime as dt
import re
import os
import csv
import nltk
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn import neighbors
import random as rn
import numpy as np
from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import cPickle


def Clean_Tweets(text,stopwords='nltk',onlyHashtags=False):
    '''this function tokenizes `tweets` using simple rules:
    tokens are defined as maximal strings of letters, digits
    and apostrophies.
    The optional argument `stopwords` is a list of words to
    exclude from the tokenzation.  This code eliminates hypertext and markup features to
    focus on the substance of tweets'''
    if stopwords=='nltk':
        stopwords=nltk.corpus.stopwords.words('english')
    else:
        stopwords=[]
    #print stopwords
    stopwords+=['bit','fb','ow','twitpic','ly','com','rt','http','tinyurl','nyti','www']
    retweet=False
    if 'RT @' in text:
        retweet=True
    # make lowercase
    text = text.lower()
    #ELEMINATE HYPERLINKS
    ntext=[]
    text=text.split(' ')
    for t in text:
        if t.startswith('http'):
            #ntext.append('hlink')
            continue
        else:
            ntext.append(t)
    text=' '.join(ntext)
    # grab just the words we're interested in
    text = re.findall(r"[\d\w'#@]+",text)
    # remove stopwords
    if onlyHashtags==True:
        htags=[]
        for w in text:
            if w.startswith('#'):
                htags.append(w)
        return htags
    res = []
    for w in text:
        if w=='hlink':
            res.append('HLINK')
            continue
        if w.startswith('@') and w!='@':
            res.append('MNTON'+'_'+w[1:])
            continue
        if w.startswith('#'):
            res.append('HSHTG'+'_'+w[1:])
            continue
        if w not in stopwords:
            res.append(w)
    if retweet:
        res.append('RTWT')
    return res


def Twitter_Tokenize(text,stopwords='nltk'):
    '''this function tokenizes `text` using simple rules:
    tokens are defined as maximal strings of letters, digits
    and apostrophies.
    The optional argument `stopwords` is a list of words to
    exclude from the tokenzation'''
    if stopwords=='nltk':
        nltk.corpus.stopwords.words('english')
    else:
        stopwords=[]
    retweet=False
    if 'RT @' in text:
        retweet=True
    # make lowercase
    text = text.lower()
    # grab just the words we're interested in
    text = re.findall(r"[\d\w'#@]+",text)
    # remove stopwords
    res = []
    for w in text:
        if w=='http':
            res.append('HLINK')
            continue
        if w.startswith('@') and w!='@':
            res.append('MNTON')
        if w.startswith('#'):
            res.append('HSHTG')
        if w not in stopwords:
            res.append(w)
    if retweet:
        res.append('RTWT')
    return(res)


def makeMetaData():
    '''metadata header is DWNominate score, name, party, ICPSR, state code, State-District,
    data is a dictionary of data[twitterID]=[]'''
    print 'importing metadata'
    data={}
    polKey={}
    missing=[]
    f=open('Data\IdealPts_112_House-Twitter.csv')
    dta=csv.reader(f)
    for i,line in enumerate(dta):
        if i>0:
            if line[8]!='':
                if line[5]=='100':        #recode to 1,0 fits with previous code and keeps direction of analysis the sm
                    party=1
                elif line[5]=='200':
                    party=0
                else:
                    continue
                    party=''
                tid=line[8].lower()
                
                data[tid]=[line[7],line[6].lower(),party,0,line[1],line[2],line[4]+'-'+line[3]]#,numTxt[tid],numWds[tid]]
                polKey[tid]=party
                #else:
                #    missing.append([tid,'House member has Twitter but not in scraped data'])
    f.close()
    f=open('Data\IdealPts_112_Senate-Twitter.csv')
    dta=csv.reader(f)
    for i,line in enumerate(dta):
        if i>0:
            if line[8]!='':
                if line[5]=='100':
                    party=1
                elif line[5]=='200':
                    party=0
                else:
                    continue
                    party=''
                tid=line[8].lower()
                #if tid in numTxt.keys():
                data[tid]=[line[7],line[6].lower(),party,1,line[1],line[2],line[4]]#,numTxt[tid],numWds[tid]]
                polKey[tid]=party
                #else:
                #    missing.append([tid,'Senator has Twitter but not in scraped data'])
    f.close()
    
    #f=open('Metadata\112Tweets3.txt','r')
    #for n in f.readlines():
    #    print n
    #    d=n.split(',')
    #    data[d[0].lower()]+=[d[1],d[2].strip()]
    #f.close()
    header=['TwitterID','dw1', 'name', 'party','senate', 'ICPSR', 'stateCode', 'district','Num112Twts','Num112Words','TotalTwts']
    
    return data, header, polKey, missing


def MakeDateEssays(dates,metaData, indir=''):
    '''This function turns a directory (indir) composed of every individual's tweets each contained in a csv file (e.g. barackobama.csv)
    into a directory (outdir) of .txt files (e.g. barackobama.txt) of person-level aggregates of those tweets within the date window given in 'dates'
    dates=[beginningDate,endDate] formatted as ['1/1/1900','12/30/1900'] 
    
    For example, the .csv file AlFranken.csv in Raw2 contains each tweet collected with the parameters: 
        text\tfavorited\treplyToSN\tcreated\ttruncated\treplyToSID\tid\treplyToUID\tstatusSource\tscreenName
    With this code, that file is turned into a .txt file called AlFranken.txt containing of all tweets under the "text" field
    separated by \t
    
    As a part of this data formating process, this code also produces counts of the total number of tweets and characters for each person
    returned as 'numTxt' and 'numWds' respectively.
    '''
    print 'importing tweets'    
    if indir=='':
        indir='Data\Raw3'
    begin=dt.datetime.strptime(dates[0],'%m/%d/%Y')
    end=dt.datetime.strptime(dates[1],'%m/%d/%Y')
    text={}
    broken=[]
    for fn in os.listdir(indir):
        tid=fn.split(".")[0].lower()
        if tid in metaData.keys():
            texts=[]
            count=0
            f=open(indir+'\\'+fn,'rb')
            data=csv.reader(f)
            for i,line in enumerate(data):
                if i>0:
                    count+=1
                    time=line[3]
                    time=time.split(' ')[0]
                    time=dt.datetime.strptime(time,'%m/%d/%Y')# %a %b %d %H:%M:%S +0000 %Y').strftime('%m/%d/%Y %H:%M:%S')
                    if time>=begin and time<=end:
                        texts.append(line[0])
                    lastTime=time
            if count>3000 and lastTime>begin:
                broken.append([tid,'hit twitter limit in time period - '+str(count)])
                #continue
            #if len(texts)<2:
                #broken.append([tid,'not enough data in time period (min of 2)'])
                #continue
            numTxt=len(texts)
            texts=' '.join(Clean_Tweets(' '.join(texts),stopwords='nltk'))#generate feature space of tweet by eliminating stops and adding metatext features
            text[tid]=texts
            numWds=len(texts)
            metaData[tid]+=[numTxt,numWds,count]
            f.close()
        
    return text, broken, metaData


def Vectorize(texts,polKey):
    vectorizer= tfidf(texts.values(),ngram_range=(1,2),stop_words='english',min_df=2)   #the only real question I have with this is whether it ejects twitter-specific text (ie. @ or #)
    vec=vectorizer.fit_transform(texts.values()) 
    labels=[]
    for k in texts.keys():
        labels.append(polKey[k])
    labels=np.asarray(labels)   
    return vec,labels,vectorizer

def Sample(vec,labels,texts,clf='knn',pct=.2):
    '''This code creates the randomized test/train samples and the trains and tests the classifier
    and returns the vectors of test and train texts and labels as well as keys for linking results to TwitterIDs'''
    trainIds=rn.sample(xrange(np.shape(labels)[0]),int(round(np.shape(labels)[0]*pct)))
    testIds=[]
    trainKey={}
    testKey={}
    ts=0
    tr=0
    for t in xrange(np.shape(labels)[0]):    
        if t not in trainIds:
            testIds.append(t)
            testKey[ts]=texts.keys()[t]
            ts+=1
        else:
            trainKey[tr]=texts.keys()[t]
            tr+=1
    trainTexts=vec[trainIds]
    trainLabels=labels[trainIds]
    testTexts=vec[testIds]
    testLabels=labels[testIds]
    
    return trainTexts, trainLabels, testTexts,testLabels,trainKey,testKey


def Classify(trainT,trainL,testT,testL,clf='knn'):
    '''Code to train and test classifiers.  type can be 'knn' 'nb' or 'svm'
    returns the fit matrix #a dictionary of {twitterID: likelihood ratio}'''
    if clf=='knn':
        cl = neighbors.KNeighborsClassifier()
        cl.fit(trainT,trainL)
        fit=cl.predict_proba(testT)
        #print(cl.score(testT,testL))
    if clf=='svm':
        cl=svm.SVC(C=100,gamma=.1,probability=True)
        cl.fit(trainT,trainL)
        fit=cl.predict_proba(testT)
        #print(cl.score(testT,testL))
    if clf=='nb':
        cl=mnb()
        cl.fit(trainT,trainL)
        fit=cl.predict_proba(testT)
        #print(cl.score(testT,testL))
    return fit, cl


def TopWords(vec,clf,n=20):
    topWords={}
    feature_names =vec.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    for w in top:
        word=w[1]
        for j, f in enumerate(vec.get_feature_names()):
            if f==word:
                demscore=np.exp(clf.feature_log_prob_[0][j])
                repscore=np.exp(clf.feature_log_prob_[1][j])
                repflag=1*(repscore>demscore)
                topWords[word]=[repflag,repscore,demscore]
    return topWords

def CleanResults(fit,testKeys):
    '''This code takes the results of classifier.predict_proba() and cleans out extreme scores and produces z-scored likelihood ratios.
    It replaces any probabilites of 1 or 0 (which produce inf likelihoods) with the nearest max and min probabilites given.
    It then computes the likelihood ratio and z-scores them, returning res as a dictionary of {twitterID: z-scored likelihood ratio}'''
    #identify any possible infinite values and recode using the next maximum probability
    if 0 in fit:
        lis=sorted(fit[:,0],reverse=True)
        lis+=sorted(fit[:,1],reverse=True)
        for l in sorted(lis,reverse=True):
            if l!=1.0:
                fit[fit==1.0]=l
                break
        for l in sorted(lis):
            if l!=0.0:
                fit[fit==0.0]=l
                break

    res=dict(zip(testKeys.values(),[0 for i in xrange(len(testKeys.keys()))]))
    for i,line in enumerate(fit):
        res[testKeys[i]]=[line[0],line[1],np.log(line[0]/line[1])]
    vals=[i[2] for i in res.values()]
    m=np.mean(vals)
    sd=np.std(vals)
    for k,v in res.iteritems():
        res[k]=[v[0],v[1],(v[2]-m)/sd]
    
    adjust=[m,sd]
    return res,adjust
        

def ClassificationReport(testLabels, testTexts,classifier):
    
    y_pred=classifier.predict(testTexts)
    print(classification_report(testLabels, y_pred))
    #print(accuracy(testLabels, y_pred))
    report=classification_report(testLabels, y_pred)
    
    return report


def SaveResults(data,metaHeader,classHeader,outfile=''):
    #check code for writing correctness then validate format and add headers for initial data creation
    '''This function joins the classifier results with the classifier metadata and the person metadata to
    the existing data of the same structure:
    PersonData, classifier data, classificaiton results
    res is the z-scored likelihood ratio data {twitterid: scored ratio}
    metadata is the dictionary {twitterID: list of person data}
    classMeta is a list of classifier features including sample pct, type of classifier, and iteration
    fname is the name of the data file being dumped to.'''
    
    print 'saving data'
    header=metaHeader+classHeader+['RepProb','DemProb','zLkRatio']
    f=open(outfile,'wb')
    writeit=csv.writer(f)
    writeit.writerow(header)
    for line in data:
        writeit.writerow(line)
    f.close()
    return


def GridSearch(clfs = ['nb','knn','svm'],samples=[.5],iters=200,outfile=''):
    ''' This code runs the classifiers across iterations and sample sizes producing the core data used in the final
    analysis.  test data for the classifier in 'clf' for 'iters' number of iterations.
    It pulls in the metadata, keys, and essays, and then iterates by random sampling, classification,
    and data cleaning producing lists of dictionaries of {TwitterID: z-scored likelihood ratios} for each iteration'''
    data=[]
    dates=['1/03/2011','1/03/2013']
    indir='Data\Raw3'
    adjust={}
    words={}
    metaData, metaHeader, polKey, missing=makeMetaData()
    #print len(missing), " Number of missing congress members.  Here's who and why: "
    #print missing
    texts, broken, metaData=MakeDateEssays(dates,metaData, indir)        #get polkey from makeMetaData and go from there
    #print len(broken), " Number of excluded congress members.  Here's who and why: "
    #print broken
    vec,labels,vectorizer=Vectorize(texts,polKey)
    f='G:/Research Data/2017-01-12 Backup/CongressTweets/Classifiers/Vectorizer.pkl'
    with open(f,'wb') as fl:
        cPickle.dump(vectorizer,fl)
    for clf in clfs:
        for samp in samples:
            #accs=[]
            for it in xrange(iters):
                print "Doing: ", clf, ' ', samp, ' ', it
                classMeta=[clf,samp,it]
                classHeader=['Classifier','SamplePct','Iteration']
                trainTexts, trainLabels, testTexts,testLabels,trainKey,testKey =Sample(vec,labels,texts,pct=samp)
                fit,classifier=Classify(trainTexts, trainLabels, testTexts,testLabels,clf=clf)
                report=ClassificationReport(testLabels, testTexts,classifier)
                print(report)
                #accs.append([classifier.score(testTexts,testLabels),np.mean([int(l) for l in testLabels])])
                res,adj=CleanResults(fit, testKey)
                if clf=='nb':
                    words[it]=TopWords(vectorizer,classifier,n=200)
                print adj
                adjust[clf+'-'+str(it)]=adj
                for k,r in res.iteritems():
                    data.append([k]+metaData[k]+classMeta+r)
                f='G:/Research Data/2017-01-12 Backup/CongressTweets/Classifiers/'+clf+str(it)+'clf.pkl'
                with open(f,'wb') as fl:
                    cPickle.dump(classifier,fl)
                
            #print "Accuracy of ", clf, " classifer on ",samp," samples is ",np.mean([a[0] for a in accs]),' from ', np.mean([a[1] for a in accs])/2.0  , ' probability'
            
    print 'writing data'
    SaveResults(data,metaHeader,classHeader,outfile=outfile)
    with open('Essay_StandardizeValues.txt','w') as f:
        f.write('Classifier,Iteration,Mean,Std\n')
        for fn,val in adjust.iteritems():
            f.write(fn.replace('-',',')+','+','.join(map(str,val))+'\n')
    
    

    return data,adjust,words

#os.chdir('C:\Users\Admin\Dropbox\Twitter_NaiveBayes')
data,adjust,words=GridSearch(clfs = ['nb','svm'],samples=[.5],iters=200,outfile='Results\\112Results2.csv')
f=open('Results\FinalPartisanWords.txt','w')
f.write('Iteration,Word,IsRepub,RepScore,DemScore\n')
for it, ws in words.iteritems():
    for w,scores in ws.iteritems():
        f.write(str(it)+","+w+','.join(map(str,scores))+'\n')
f.close()
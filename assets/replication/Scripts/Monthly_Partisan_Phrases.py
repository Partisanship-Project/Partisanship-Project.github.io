
'''
Code to detect and visualize the most partisan phrases every month.
'''

#cd 'C:\Users\Boss\Dropbox\Twitter_NaiveBayes'

import datetime as dt
from dateutil import rrule
import re,os,csv,nltk,operator

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn import neighbors
import random as rn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import cPickle
from sklearn.metrics import classification_report


def Twitter_Tokenize(text,stopwords='nltk'):
    '''this function tokenizes `text` using simple rules:
    tokens are defined as maximal strings of letters, digits
    and apostrophies.
    The optional argument `stopwords` is a list of words to
    exclude from the tokenzation'''
    if stopwords=='nltk':
        stopwords=nltk.corpus.stopwords.words('english')
    else:
        stopwords=[]
    stopwords+=['bit','fb','ow','twitpic','ly','com','rt','http','tinyurl','nyti','www','https','amp']
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
    stopwords+=['bit','fb','ow','twitpic','ly','com','rt','http','tinyurl','nyti','www','https','amp']
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

def makeMetaData():
    '''metadata header is DWNominate score, name, party, ICPSR, state code, State-District,
    data is a dictionary of data[twitterID]=[]'''
    data={}
    polKey={}
    missing=[]
    with open('..\Data\metadata.csv') as f:
        
        dta=csv.reader(f)
        for i,line in enumerate(dta):
            if i>0:
                if line[5]!='':
                    tid=line[5].lower()
                    party=line[3]
                    if line[3]=='R':        #recode to 1,0 fits with previous code and keeps direction of analysis the sm
                        biparty=1
                    elif line[3]=='D':
                        biparty=0
                    else:
                        biparty=''
                    name=line[0]
                    state=line[1]
                    district=line[2]
                    incumbent=line[4]
                    polldate=line[6]
                    winprob=line[12]
                    predVoteShare=line[13]
                    if district.lower()=='senate':
                        district=0
                    if int(district)==0:
                        senate='1'
                        
                    else:
                        senate='0'
                    data[tid]=[name,biparty,party,state,district,senate,incumbent,polldate,winprob,predVoteShare]
                    polKey[tid]=biparty
                    #else:
                    #    missing.append([tid,'House member has Twitter but not in scraped data'])
    
    
    #f=open('Metadata\112Tweets3.txt','r')
    #for n in f.readlines():
    #    print n
    #    d=n.split(',')
    #    data[d[0].lower()]+=[d[1],d[2].strip()]
    #f.close()
    header=['TwitterID','name','lib-con','party','state','district','senate','incumbent','polldate','winprob','predVoteShare']
    
    return data, header, polKey


def MakeDateEssays(months,metaData, indir=''):
    '''This function turns a directory (indir) composed of every individual's tweets each contained in a csv file 
    into a directory (outdir) of .txt files of person-level aggregates of those tweets within the date window given in 'dates'
    dates=[beginningDate,endDate] formatted as ['1/1/1900','12/30/1900']
    
    For example, the .csv file AlFranken.csv in Raw2 contains each tweet collected with the parameters: 
        text\tfavorited\treplyToSN\tcreated\ttruncated\treplyToSID\tid\treplyToUID\tstatusSource\tscreenName
    With this code, that file is turned into a .txt file called AlFranken.txt containing of all tweets under the "text" field
    separated by \t
    
    As a part of this data formating process, this code also produces counts of the total number of tweets and characters for each person
    returned as 'numTxt' and 'numWds' respectively.
    '''
    if indir=='':
        indir='G:\Congress\Twitter\Data'
    text={}
    broken=[]
    for fn in os.listdir(indir):
        if fn.endswith('json'):
            continue
        tid=fn.split(".")[0].lower()
        if tid in metaData.keys():
            texts=[]
            count=0
            with open(indir+'\\'+fn,'rb') as f:
                data=csv.reader(f)
                for i,line in enumerate(data):
                    if i>0:
                        count+=1
                        time=line[3]
                        time=time.split(' ')[0]
                        time=dt.datetime.strptime(time,'%m/%d/%Y')# %a %b %d %H:%M:%S +0000 %Y').strftime('%m/%d/%Y %H:%M:%S')
                        if time>=months[0] and time<=months[1]:
                            texts.append(line[0])
            #if count>3000 and lastTime>begin:
            #    broken.append([tid,'hit twitter limit in time period - '+str(count)])
            #    continue
            #if len(texts)<2:
            #    broken.append([tid,'not enough data in time period (min of 2)'])
            #    continue
            #numTxt=len(texts)
            texts=' '.join(Clean_Tweets(' '.join(texts),stopwords='nltk',onlyHashtags=False))#generate feature space of tweet by eliminating stops and adding metatext features
            text[tid]=texts
            #numWds=len(texts)        
    return text, broken, metaData


def Vectorize(texts,polKey):
    vectorizer= tfidf(texts.values(),ngram_range=(1,2),stop_words='english',min_df=5)   #the only real question I have with this is whether it ejects twitter-specific text (ie. @ or #)
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
    return res, adjust
        

def NBTopWords(vec,clf,n=20):
    topWords={}
    feature_names =vec.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names,xrange(clf.coef_.shape[1])))
    top = coefs_with_fns[:-(n + 1):-1]
    for w in top:
        word=w[1]
        for j, f in enumerate(vec.get_feature_names()):
            if f==word:
                dem=np.exp(clf.feature_log_prob_[0][j])
                rep=np.exp(clf.feature_log_prob_[1][j])
                party=1*(rep>dem)
                
                topWords[word]=[party,rep,dem]
    return topWords

def TopWordsperPerson(vectorizer,vec,clf,texts,n=500):
    print 'getting top partisan words for members'
    personWords={}
    feature_names =vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names,xrange(classifier.coef_.shape[1])))
    top = coefs_with_fns[:-(n + 1):-1]
    for i,row in enumerate(vec): 
        person=texts.keys()[i]
        personWords[person]={}
        for (r,w,idx) in top:
            personWords[person][w]=vec[i,idx]*r
    
    for person, worddict in personWords.iteritems():
        personWords[person] = sorted(worddict.iteritems(), key=operator.itemgetter(1))[0:20]
        
    return personWords


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
    #header=metaHeader+classHeader+['RepProb','DemProb','zLkRatio']
    header=['Time_Index',"Time_Label",'TwitterID','zLkRatio']
    with open(outfile,'wb') as f:
        writeit=csv.writer(f)
        writeit.writerow(header)
        for line in data:
            writeit.writerow([line[1],line[2],line[3],line[len(line)-1]])   #save 
    return


def SaveWords(words,outfile=''):
    #check code for writing correctness then validate format and add headers for initial data creation
    '''This function saves the top words each month.'''
    
    print 'saving data'
    header=['month','word','party','repprob','demprob']
    header=['Time','Word','Log Likelihood']
    f=open(outfile,'wb')
    writeit=csv.writer(f)
    writeit.writerow(header)
    for month,worddict in words.iteritems():
        for word, res in worddict.iteritems():
            logprob=np.log(res[1]/res[2])
            writeit.writerow([month, word,logprob])
        #line=[month]
        #for word,dat in worddict.iteritems():
        #    line.append(word)
        #    #line=[month,word]+dat
        #writeit.writerow(line)
    f.close()
    return

def AggregateEstimate(clfs = ['nb','knn','svm'],samples=[.5],iters=200,outfile=''):
    ''' This code runs the classifiers across iterations and sample sizes producing the core data used in the final
    analysis.  test data for the classifier in 'clf' for 'iters' number of iterations.
    It pulls in the metadata, keys, and essays, and then iterates by random sampling, classification,
    and data cleaning producing lists of dictionaries of {TwitterID: z-scored likelihood ratios} for each iteration'''
    Data={}
    for c in clfs:
        Data[c]={}
    dates=['1/03/2000','1/03/2019']
    begin=dt.datetime.strptime(dates[0],'%m/%d/%Y')
    end=dt.datetime.strptime(dates[1],'%m/%d/%Y')
    indir='G:\Congress\Twitter\Data'
    adjust={}
    words={}
    cache=True
    if cache:
        with open('G:\cache\mpp.pkl','rb') as f:
            print 'importing from cache'
            metaData,metaHeader,texts,vec,labels,vectorizer=cPickle.load(f)
    else:
        print 'getting meta data'
        metaData, metaHeader, polKey=makeMetaData()
        #print len(missing), " Number of missing congress members.  Here's who and why: "
        #print missing
        print 'importing tweets'
        texts, broken, metaData=MakeDateEssays([begin,end],metaData, indir)        #get polkey from makeMetaData and go from there
        #print len(broken), " Number of excluded congress members.  Here's who and why: "
        #print broken
        print 'vectorizing texts'
        vec,labels,vectorizer=Vectorize(texts,polKey)
        print 'caching files'
        f='G:\cache\mpp.pkl'
        with open(f,'wb') as fl:
            cPickle.dump([metaData,metaHeader,texts,vec,labels,vectorizer],fl)
            
    for clf in clfs:
        best=0
        data={}
        for samp in samples:
            #accs=[]
            for it in xrange(iters):
                print "Doing: ", clf, ' ', samp, ' ', it
                classMeta=[clf,samp,it]
                classHeader=['Classifier','SamplePct','Iteration']
                trainTexts, trainLabels, testTexts,testLabels,trainKey,testKey =Sample(vec,labels,texts,pct=samp)
                fit,classifier=Classify(trainTexts, trainLabels, testTexts,testLabels,clf=clf)
                classifier.score(testTexts,testLabels)                
                report=ClassificationReport(testLabels, testTexts,classifier)
                print(report)
                #accs.append([classifier.score(testTexts,testLabels),np.mean([int(l) for l in testLabels])])
                print "Accuracy of ", clf, ' was ',classifier.score(testTexts,testLabels)
                res,adj=CleanResults(fit, testKey)
                if clf=='nb' and best < classifier.score(testTexts,testLabels):
                    #words[it]=NBTopWords(vectorizer,classifier,n=200)
                    personWords=TopWordsperPerson(vectorizer,vec,classifier,texts,n=500)
                    best=classifier.score(testTexts,testLabels)
                #print adj
                adjust[clf+'-'+str(it)]=adj
                for k,r in res.iteritems():
                    if k in data.keys():
                        for j,num in enumerate(r):
                            data[k][j].append(num)
                    else:
                            data[k]=[[num] for num in r]
                
                #f='G:/Research Data/2017-01-12 Backup/CongressTweets/Classifiers/'+clf+str(it)+'clf.pkl'
                #with open(f,'wb') as fl:
                #    cPickle.dump(classifier,fl)
                
        #print "Averge Accuracy of ", clf, " for ", month, ' was ', np.mean(accs)
        for k, res in data.iteritems():
            Data[clf][k]=map(np.mean,res)   #Data['nb']['barackobama'][avgprobreb,avgprobdem,avgzlkration]
                    
                    
                
                
            #print "Accuracy of ", clf, " classifer on ",samp," samples is ",np.mean([a[0] for a in accs]),' from ', np.mean([a[1] for a in accs])/2.0  , ' probability'
            
    print 'saving data'
    outdata={}
    for clf, tdata in Data.iteritems():
        outdata[clf]={}
        for tid,res in tdata.iteritems():
            #line=[tid]+res
            outdata[clf][tid]=res[2]
        
    print 'saving data'
    #header=metaHeader+classHeader+['RepProb','DemProb','zLkRatio']
    for clf, data in outdata.iteritems():
        output_file='..\Results\Aggregate_Metadata_'+clf+'.csv'
        output_header=metaHeader+['TwitterID','zLkRatio']
        with open(outfile,'wb') as f:
            writeit=csv.writer(f)
            writeit.writerow(output_header)
            for k,v in metaData.iteritems():
                if k in outdata[clf].keys():
                    writeit.writerow([k]+v+[outdata[clf][k]])   #save
                else:
                    writeit.writerow([k]+v+[''])   #save
            
    header=['Time Index',"Time Label",'TwitterID','zLkRatio']
    
    #header=metaHeader+classHeader+['RepProb','DemProb','zLkRatio']
    header=['CLF','TwitterID','zLkRatio']
    with open(outfile,'wb') as f:
        writeit=csv.writer(f)
        writeit.writerow(header)
        for line in data:
            writeit.writerow([line[1],line[2],line[3],line[len(line)-1]])   #save 
    #SaveResults(data,metaHeader,classHeader,outfile=outfile)
    with open('..\Results\Essay_StandardizeValues.txt','w') as f:
        f.write('Classifier,Iteration,Mean,Std\n')
        for fn,val in adjust.iteritems():
            f.write(fn.replace('-',',')+','+','.join(map(str,val))+'\n')
    
    with open('..\Results\AggregatePartisanWords.csv','w') as f:
        writeit=csv.writer(f)        
        writeit.writerow(['Iteration','Word','IsRepub','RepScore','DemScore'])
        for it, ws in words.iteritems():
            for w,scores in ws.iteritems():
                writeit.writerow([it,w]+[scores])

    with open('..\Results\PartisanWordsPerMember.csv','w') as f:
        writeit=csv.writer(f)
        for person, wordset in personWords.iteritems():
            #for (w,scores) in wordset:
            writeit.writerow([person]+[w[0] for w in wordset]) 
        
    
    return data,adjust,words

def MonthlyEstimate(clfs = ['nb','knn','svm'],samples=[.5],iters=200,outfile=''):
    ''' This code runs the classifiers across iterations and sample sizes producing the core data used in the final
    analysis.  test data for the classifier in 'clf' for 'iters' number of iterations.
    It pulls in the metadata, keys, and essays, and then iterates by random sampling, classification,
    and data cleaning producing lists of dictionaries of {TwitterID: z-scored likelihood ratios} for each iteration'''
    Data={}
    for c in clfs:
        Data[c]={}
    indir='G:\Congress\Twitter\Data'
    dates=['1/01/2018','10/19/2018']
    begin=dt.datetime.strptime(dates[0],'%m/%d/%Y')
    end=dt.datetime.strptime(dates[1],'%m/%d/%Y')
    monthKey={}
    words={}
    metaData, metaHeader, polKey=makeMetaData()
    for i,d in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=begin, until=end)):
        if i==0:
            oldDate=d
        if i>0:
            months=[oldDate,d]
            month=oldDate.strftime('%b')+' '+d.strftime('%Y')
            texts, broken, metaData=MakeDateEssays(months,metaData, indir)        #get polkey from makeMetaData and go from there
            vec,labels,vectorizer=Vectorize(texts,polKey)
            monthKey[i]=month
            for clf in clfs:
                Data[clf][i]={}
                data={}
                for samp in samples:
                    print "Doing: ", clf, ' on ',month,
                    accs=[]
                    for it in xrange(iters):
                        trainTexts, trainLabels, testTexts,testLabels,trainKey,testKey =Sample(vec,labels,texts,pct=samp)
                        fit,classifier=Classify(trainTexts, trainLabels, testTexts,testLabels,clf=clf)
                        words[month]=NBTopWords(vectorizer,classifier,n=20)
                        
                        
                        
                        accs.append(classifier.score(testTexts,testLabels))
                        res,adjust=CleanResults(fit, testKey)
                        for k,r in res.iteritems():
                            if k in data.keys():
                                for j,num in enumerate(r):
                                    data[k][j].append(num)
                            else:
                                    data[k]=[[num] for num in r]
                print "Averge Accuracy of ", clf, " for ", month, ' was ', np.mean(accs)
                report=ClassificationReport(testLabels, testTexts,classifier)
                print(report)
                for k, res in data.iteritems():
                    Data[clf][i][k]=map(np.mean,res)
#                    print clf, samp, accs
#                
#                    print "Accuracy of ", clf, " classifer on ",samp," samples is ",np.mean([a[0] for a in accs]),' from ', np.mean([a[1] for a in accs])/2.0  , ' probability'
            oldDate=d
    outdata={}
    for clf, mons in Data.iteritems():
        outdata[clf]=[]
        for mon,tdata in mons.iteritems():
            for tid,res in tdata.iteritems():
                line=[clf]+[mon]+[monthKey[mon]]+[tid]+res
                outdata[clf].append(line)
    for clf, data in outdata.iteritems():
        metaHeader=['Classifier','Month','StrMonth','TwitterID']
        classHeader=[]
        print 'writing data for ', clf
        outfile='..\Results\MonthlyIdealPts_'+clf+'.csv'
        SaveResults(data,metaHeader,classHeader,outfile=outfile)
    SaveWords(words,outfile='..\Results\TEST_MonthlyPartisanWords.csv')
    return 



AggregateEstimate(clfs = ['nb'],samples=[.3],iters=30,outfile='..\Results\TEST_AggregatePartisanship.csv')
#MonthlyEstimate(clfs = ['nb'],samples=[.5],iters=30,outfile='..\Results\TEST_MonthlyIdeals.csv')
#SaveWords(words,outfile='..\Results\TEST_MonthlyPartisanWords.csv')
#f=open('TEST_monthly112Words.pkl','w')
#cPickle.dump(words,f)
#f.close()
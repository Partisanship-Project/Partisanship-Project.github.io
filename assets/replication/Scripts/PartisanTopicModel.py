
import re, os, csv
import nltk
import random as rn
#import hashlib
import numpy as np
from gensim import corpora
#from itertools import groupby
#import cPickle
#import scipy
import TopicModels
reload(TopicModels)

def simple_clean(text,stopwords='nltk'):
    if stopwords=='nltk':
        stopwords=nltk.corpus.stopwords.words('english')
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



def ImportTweets(decile=10.0,typ='nonpartisan'):
    '''
    This function imports tweets from partisanscores.csv, filters out those that are more
    partisan than cut, and returns a list with the tweets and ids.
    
    Input:
        cut - max partisanship rating (scale of 0-inf, mean is .69)
        
    Output
        tweets- dictionary of tweet id and pre-cleaned tweet (e.g. {123456789: "This tweet HLINK})
        ids - dictionary of tweet id and twitter id (e.g. {123456789: 'barackobama'})
        
    '''
    
    print 'importing tweets'    
    name='G:\Homestyle\Data\\partisanscores.csv'
    data=[]
    directory='G:/Congress/Twitter Ideology/Partisan Scores/'
    for fn in os.listdir(directory):
        with open(directory+'\\'+fn,'r') as f:
            twitterid=fn.split('.')[0]
            for i,line in enumerate(f.readlines()):
                if i==1:
                    newline=[line.split(",")[0],float(line.split(",")[1]),','.join(line.split(",")[2:])]
                   
                if i>0:
                    try: 
                        float(line.split(",")[0])
                        data.append([twitterid]+newline)
                        newline=[line.split(",")[0],float(line.split(",")[1]),','.join(line.split(",")[2:])]
                   
                    except:
                        newline[2]+line
                   #data.append([twitterid]+newline)
                    
    texts={}
    ids={}
    #    data=[]
    #    with open(name,'rb') as f:
    #        dta=csv.reader(f)
    #        for i,line in enumerate(dta):
    #            if i >0:
    #                data.append(line)
    if typ=='nonpartisan':
        cut=sorted([abs(float(d[2])) for d in data])[int(len(data)/decile)]
        for line in data:
            if float(line[2])<=cut:
                text=line[3].strip('\n')
                text=' '.join(simple_clean(text,stopwords='nltk'))
                texts[line[1]]=text
                ids[line[1]]=line[0]
    if typ=='partisan':
        cut=sorted([abs(float(d[2])) for d in data],reverse=True)[int(len(data)/decile)]
        for line in data:
            if float(line[2])>=cut:
                text=line[3].strip('\n')
                text=' '.join(simple_clean(text,stopwords='nltk'))
                texts[line[1]]=text
                ids[line[1]]=line[0]
    return texts,ids

def sample(texts,pct):
    train={}
    test={}
    heldout={}
    for idx, tweet in texts.iteritems():
        ran=rn.random()
        if ran<pct:
            train[idx]=tweet
        elif ran<pct+pct and ran>=pct:
            test[idx]=tweet
        else:
            heldout[idx]=tweet
    return train,test,heldout

def gensimGenerator(texts,dictionary):
    for text in TopicModels.Tokenize(texts):
        yield dictionary.doc2bow(text)


def getTopicsGenerator(model,texts,dictionary):
    for text in texts:
        bow = dictionary.doc2bow(TopicModels.Tokenize(text))
        topic_in_doc=dict(model.get_document_topics(corpus))
        yield topic_in_doc


def RawTopicScore(texts,numtopics=200,iterations=500,passes=10,name='Twitter_Raw',**exargs):
    '''
    This code runs a topic model on the texts and returns a vector of texts and proportions of topics in texts
    Input:
        texts = {id1: "text1",id2:'text2',...}
    
    '''
    from gensim import corpora
    print 'doing topic modelling on ', len(texts), ' texts and ', numtopics, ' topics'
    #runfile('C:\Users\Boss\Documents\Python Scripts\onlineldavb.py')
    print 'tokenizing ', name
    #texts=RemoveStops(texts)
    toktexts=TopicModels.Tokenize(texts)
    dictionary=TopicModels.vocabulary(toktexts)
    print 'original vocabulary size is ', len(dictionary)
    dictionary.filter_extremes(**exargs)#)
    print 'reduced vocabulary size is ',len(dictionary)
    dictionary.compactify()
    print 'reduced vocabulary size is ',len(dictionary)
    #corpus = [dictionary.doc2bow(text) for text in TopicModels.Tokenize(texts)]
    corpusgenerator=gensimGenerator(texts,dictionary)
    corpora.MmCorpus.serialize('Data/'+name+'_Corpus.mm', corpusgenerator) 
    #print 'vectorizing ', name
    #tfidf_corpus,tfidf,corpus=TopicModels.vectorize(toktexts,dictionary)
    print 'Doing lda ', name
    mm = corpora.MmCorpus('Data/'+name+'_Corpus.mm',)
    model,topic_in_document=TopicModels.topics(mm,dictionary,strategy='lda', num_topics=numtopics,passes=passes,iterations=iterations) #passes=4
    
    print 're-formatting data'   
    topic_in_documents=[dict(res) for res in topic_in_document] #Returns list of lists =[[(top2, prob), (top8, prob8)],[top1,prob]]
    Data=[]
    for doc, resdict in enumerate(topic_in_documents):
        line=[texts.keys()[doc]]    #This should line up. If it doesn't the model results will be random noise
        for i in xrange(numtopics):
            if i in resdict.keys():
                line.append(resdict[i])
            else:
                line.append(0.0)
        Data.append(line)
    print "writing Document by Topic scores for  ", name
    with open('Results/'+name+'_Topic_Scores_'+str(numtopics)+'.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+["Topic_"+str(n) for n in xrange(numtopics)])
        for info in Data:
            writer.writerow([str(i) for i in info])
                
    print 'writing topic words to Results Folder for ', name
    words=TopicModels.wordsInTopics(model, numWords = 25)
    with open('Results/'+name+'_TopicsByWords_'+str(numtopics)+'.csv','wb') as f:
        writer=csv.writer(f,delimiter=',')
        for topic,wordlis in words.iteritems():
            writer.writerow([topic]+[" ".join(wordlis)])
    model.save('Results/'+name+'_model.model')
    #save(fname, ignore=('state', 'dispatcher'), separately=None, *args, **kwargs)
    #NOTE: LOAD MODEEL WITH model =  models.LdaModel.load('lda.model')
    return model,dictionary

def PrepTopicModelTexts(texts,name,**exargs):
    
    print 'tokenizing ', name
    #texts=RemoveStops(texts)
    toktexts=TopicModels.Tokenize(texts)
    dictionary=TopicModels.vocabulary(toktexts)
    print 'original vocabulary size is ', len(dictionary)
    dictionary.filter_extremes(**exargs)#)
    print 'reduced vocabulary size is ',len(dictionary)
    dictionary.compactify()
    print 'reduced vocabulary size is ',len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in TopicModels.Tokenize(texts)]
    corpusgenerator=gensimGenerator(texts,dictionary)
    corpora.MmCorpus.serialize('Data/'+name+'_Corpus.mm', corpusgenerator) 
    mm=corpora.MmCorpus('Data/'+name+'_Corpus.mm',)
    return toktexts,dictionary, corpus,mm

def GridSearch(test,train,dictionary,numtopics,iterations=10,passes=3,**exargs):
    #load corpus
    out=[]
    train=corpora.MmCorpus('Data/train_Corpus.mm')
    test=corpora.MmCorpus('Data/test_Corpus.mm',)
    for num in numtopics:
        print 'Doing lda ', num
        train=corpora.MmCorpus('Data/Train_Corpus.mm')
        model,topic_in_document=TopicModels.topics(train,dictionary,strategy='lda', num_topics=num,passes=passes,iterations=iterations) #passes=4
        print 'fit model', num
        
        p=model.bound(test)
        print'perplexity: ', num, p
        out.append([num,p])
   
    return out
    
def FitTopicModel(dictionary,numtopics=10,passes=10,iterations=50):#strategy='lda', num_topics=numtopics,passes=passes,iterations=iterations
    print 'Doing lda ', name
    mm = corpora.MmCorpus('Data/'+name+'_Corpus.mm',)
    model,topic_in_document=TopicModels.topics(mm,dictionary,strategy='lda', num_topics=numtopics,passes=passes,iterations=iterations) #passes=4
    return
    
def SaveWordsInTopics(model,filename,numWords=25):
    print 'writing topic words to Results Folder for ', name
    words=TopicModels.wordsInTopics(model)
    #with open('Results/'+name+'_TopicsByWords_'+str(numtopics)+'.csv','wb') as f:
    with open(filename,'wb') as f:
        writer=csv.writer(f,delimiter=',')
        for topic,wordlis in words.iteritems():
            writer.writerow([topic]+wordlis)
    return
    
def SaveTopicsInDocuments(topic_in_documents,filename):
    print 're-formatting data'   
    topic_in_documents=[dict(res) for res in topic_in_document] #Returns list of lists =[[(top2, prob), (top8, prob8)],[top1,prob]]
    Data=[]
    for doc, resdict in enumerate(topic_in_documents):
        line=[texts.keys()[doc]]    #This should line up. If it doesn't the model results will be random noise
        for i in xrange(numtopics):
            if i in resdict.keys():
                line.append(resdict[i])
            else:
                line.append(0.0)
        Data.append(line)    
    print "writing Topic by Document scores"
    #with open('Results/'+name+'_Topic_Scores.csv','wb') as f:    
    with open(filename,'wb') as f:
        writer=csv.writer(f,delimiter=',')
        writer.writerow(['id']+["Topic_"+str(n) for n in xrange(len(Data[0]))])
        for info in Data:
            writer.writerow([str(i) for i in info])
    return
    
    
def TopicPipeline(name):
    alltexts,ids=ImportTweets(indir='')
    train,test,heldout=sample(alltexts,.20)  #used to reduce the number of tweets pulled in for code-building and testing purposes.
    trainToktexts,trainDictionary, train_bow,serialTrain=PrepTopicModelTexts(train,name)
    testToktexts,testDictionary, test_bow, serialTest=PrepTopicModelTexts(test,name='gridtest')
    model=FitTopicModel(filename,dictionary)
    return out

def GridSearchPipeline():
    print 'importing tweets'
    alltexts,ids=ImportTweets(decile=.6)
    print 'creating samples'
    train,test,heldout=sample(alltexts,.25)  #used to reduce the number of tweets pulled in for code-building and testing purposes.
    print 'vectorizing texts of length', len(train)
    exargs={'no_below':5,'no_above':.90,'keep_n':10000}
    toktexts,dictionary, corpus, serialTrain=PrepTopicModelTexts(train,name='train',**exargs)  
    toktexts,dictionary, corpus, serialTest=PrepTopicModelTexts(test,name='test',**exargs)  
    print 'launching grid search'
    out = GridSearch(serialTrain,serialTest,dictionary,numtopics=[5,10,20,30,50,70,100],iterations=2,passes=1,**exargs)
    return out
#
print 'importing tweets'
alltexts,ids=ImportTweets(decile=10,typ='nonpartisan')
exargs={'no_below':20,'no_above':.90,'keep_n':10000}
model,dictionary=RawTopicScore(alltexts,numtopics=10,iterations=100,passes=10,name='nonpartisan',**exargs)

print 'importing tweets'
alltexts,ids=ImportTweets(decile=10,typ='nonpartisan')
exargs={'no_below':20,'no_above':.90,'keep_n':10000}
model,dictionary=RawTopicScore(alltexts,numtopics=31,iterations=100,passes=10,name='nonpartisan',**exargs)

alltexts,ids=ImportTweets(decile=10,typ='partisan')
exargs={'no_below':20,'no_above':.90,'keep_n':10000}
model,dictionary=RawTopicScore(alltexts,numtopics=30,iterations=100,passes=10,name='partisan',**exargs)


#out=GridSearch(dictionary,[10,20,30,50,70,100],iterations=10,passes=3,**exargs)
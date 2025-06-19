import numpy as np
class Evaluate():
    
    def __init__(self,query_instance,population,permitted_features):
        self.query_instance=query_instance
        self.population=population
        self.permitted_features=permitted_features
    def getMetrics(self,cf):
        distance=[]
        implausibility=[]
        actionability=0
        for i in range(len(cf)):
            distance.append(np.linalg.norm(self.query_instance - cf[i])/np.shape(cf)[0])
            s=0
            for j in range(np.shape(cf[i])[1]):
                 if(np.linalg.norm(self.query_instance[:,j]-cf[i][:,j])>0 and (j not in self.permitted_features) ):
                      s+=1
            if (s==0):
                 actionability+=1
            diversity = (np.linalg.norm(self.population[0]-cf[i])/np.shape(cf)[0]) if(len(self.population)!=0) else 3
            for j in range(len(self.population)):
                d =  (np.linalg.norm(self.population[j]-cf[i])/np.shape(cf)[0])
                if(diversity>d):
                        diversity=d
            implausibility.append(diversity)
        return np.mean(distance)/len(cf),np.mean(implausibility)/len(cf),actionability
    def Instability(self,qr2,cf1,cf2):
        distance=0
        for i in range(len(cf1)):
              for j in range(len(cf2)):
                   distance+=np.linalg.norm(cf1[i]-cf2[j])/np.shape(cf1)[0]
        distance=distance/(len(cf1)*len(cf2))
        d=np.linalg.norm(self.query_instance-qr2)/np.shape(self.query_instance)[0]
        d=1/(1+d)
        return d*distance
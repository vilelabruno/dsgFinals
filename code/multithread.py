import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import queue
import threading
import time
import sys
import warnings

warnings.filterwarnings("ignore")

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, q):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.q = q
   def run(self):
      print ("Starting " + self.name)
      process_data(self.name, self.q)
      print ("Exiting " + self.name)

def process_data(threadName, q):
   while not exitFlag:
      queueLock.acquire()
      if not workQueue.empty():
         data = q.get()
         queueLock.release()
         line = train['products'][data]
         ll = eval(line)
         avgg = 0
         maxx = 0
         minn = 0
         stdd = 0
         soma = 0
         if ll[0] != 'null':
            minn = ll[0]['price']
            maxx = ll[0]['price']
            for j in range(0, len(ll)):
               if ll[j]['price'] > maxx:
                  maxx = ll[j]['price']
               if ll[j]['price'] < minn:
                  minn = ll[j]['price']

               avgg += ll[j]['price']
            soma = avgg
            avgg = avgg/len(ll)
            stdd = (((soma - avgg)**2)/len(ll))**(1/2)
            train['avgProdPrice'][train['index'] == data] = avgg 
            train['minProdPrice'][train['index'] == data] = minn 
            train['maxProdPrice'][train['index'] == data] = maxx
            train['stdProdPrice'][train['index'] == data] = stdd 
      else:
         queueLock.release()

pathin = "~/felipe/DSG-Finals/temp/Home/Tre/"
train = pd.read_csv(pathin + 'data' + str(sys.argv[1]) + '.csv')
df = pd.DataFrame()
train['products'] = train['products'].fillna('["null"]')
train = train.reset_index()
train['avgProdPrice'] = 0
train['minProdPrice'] = 0
train['maxProdPrice'] = 0
train['stdProdPrice'] = 0
threadList = ["Thread-1", "Thread-2", "Thread-3", "Thread-4", "Thread-5", "Thread-6", "Thread-7", "Thread-8"]
#threadList = ["Thread-1", "Thread-2", "Thread-3", "Thread-4", "Thread-5", "Thread-6", "Thread-7", "Thread-8",
              #"Thread-9", "Thread-10", "Thread-11", "Thread-12", "Thread-13", "Thread-14", "Thread-15", "Thread-16",
              #"Thread-17", "Thread-18", "Thread-19", "Thread-20", "Thread-21", "Thread-22", "Thread-23", "Thread-24",
              #"Thread-25", "Thread-26", "Thread-27", "Thread-28", "Thread-29", "Thread-30", "Thread-31", "Thread-32",
              #"Thread-33", "Thread-34", "Thread-35", "Thread-36", "Thread-37", "Thread-38", "Thread-39", "Thread-40",
              #"Thread-41", "Thread-42", "Thread-43", "Thread-44", "Thread-45", "Thread-46", "Thread-47", "Thread-48",
              #"Thread-49", "Thread-50", "Thread-51", "Thread-52", "Thread-53", "Thread-54", "Thread-55", "Thread-56",
              #"Thread-57", "Thread-58", "Thread-59", "Thread-60", "Thread-61", "Thread-62", "Thread-63", "Thread-64"]
nameList = list(range(len(train)))
queueLock = threading.Lock()
workQueue = queue.Queue(len(train)+1)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
   thread = myThread(threadID, tName, workQueue)
   thread.start()
   threads.append(thread)
   threadID += 1

# Fill the queue
queueLock.acquire()
for word in nameList:
   workQueue.put(word)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
   pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
pathout = "output/"
train.to_csv(pathout + 'train' + str(sys.argv[1]) + '.csv', index=False)
print ("Exiting Main Thread")
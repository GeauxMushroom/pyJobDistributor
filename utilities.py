import subprocess
import os
from multiprocessing import Pool
from numpy import vstack, hstack
import time
from numpy.random import randint


nCpuPerNode=4
nMicPerNode=0
nProcPerCPU=20
nProcPerMIC=240


def getHostList():
    hostfile = os.environ.get('PBS_NODEFILE')
    if hostfile == None:
        return ["LOCAL"]
    else:
        nodeList = set(open(hostfile).readlines())
        nodeList = map(lambda s: s.strip(), nodeList)
        return nodeList


def callFunc(parList):
    [nodeName,procType, procId], jobId, jobPar = parList
    HOST = nodeName
    print "Proc %d is going to work for %d seconds" %(int(procId), jobPar)
    time.sleep(jobPar)
    if HOST == "LOCAL":
        result = subprocess.check_output("./printHost.cpu")
        return result

    CPU_COMMAND = '~/pyJobDistributor/printHost.cpu'
    MIC_COMMAND = 'ssh mic%d ~/pyJobDistributor/printHost.mic' %int(micId)
    if procType == 'cpu':
        ssh = subprocess.Popen(["ssh", "%s" % HOST, CPU_COMMAND],
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    else:
        ssh = subprocess.Popen(["ssh", "%s" % HOST, MIC_COMMAND],
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    result = ssh.stdout.readlines()
    error = ssh.stderr.readlines()
    return result, error

def runJobSerial():
    nodeList = getHostList()
    nProcPerNode = nCpuPerNode + nMicPerNode
    procType = ['cpu'] * nCpuPerNode + ['mic'] * nMicPerNode
    micId =  [0] * nCpuPerNode + range(nMicPerNode)
    for node in nodeList:
        for i in range(nProcPerNode):
            callFunc(procType[i],node,micId[i],0)
    return


def getProcList():
    nodeList = getHostList()
    nNodes = len(nodeList)
    nProcPerNode = nCpuPerNode + nMicPerNode
    procType = ['cpu'] * nCpuPerNode + ['mic'] * nMicPerNode 
    procType = procType * nNodes
    procId = range(nCpuPerNode) + range(nMicPerNode) 
    procId = procId * nNodes
    hostName = hstack(map(lambda s:[s] * nProcPerNode, nodeList))
    procList = vstack([ hostName, procType, procId]).transpose()                    
    return procList
                     

def runJobAsync():

    procList = getProcList()
    nProc = len(procList)
    pool = Pool(processes = nProc)
    result = []

    nJobs = 10
    jobPres = [ None ] * nJobs 
    jobFinished = [False] * nJobs
    jobTaken = [False] * nJobs
    jobPar = randint(10, size= nJobs)+5

    for i in range(nProc):
        result.append(pool.apply_async(callFunc,[[procList[i],i,jobPar[i]]]))
    currentJob = range(nProc)
    jobTaken[:nProc] = [True] * nProc

    allFinished = False
    while(not allFinished):
        time.sleep(1)
        for i in range(nProc):
            if currentJob[i] == -1:
                # this process has retired
                continue
            if (result[currentJob[i]].ready()):
                print "Proc %d finished with Job %d" %(i,currentJob[i])
                print result[currentJob[i]].get()
                jobFinished[currentJob[i]] = True
                j = 0
                while( j < nJobs and (jobFinished[j] or jobTaken[j])):
                    j += 1
                if j == nJobs:
                    print "Proc %d has retired " %(i)
                    currentJob[i] = -1
                else:
                    result.append(pool.apply_async(callFunc,[[procList[i],j,jobPar[j]]]))
                    currentJob[i] = j
                    jobTaken[j] = True
                    print "Proc %d starts working on Job %d" %(i,j)

        allFinished = True
        for j in range(nJobs):
            if(not jobFinished[j]):
                allFinished = False

    print "all jobs finished!"

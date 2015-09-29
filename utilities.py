import subprocess
import os
from multiprocessing import Pool
from numpy import vstack
import time
from random import randint


nCpuPerNode=1
nMicPerNode=2
nProcPerCPU=20
nProcPerMIC=240


def getHostList():
    hostfile = os.environ['PBS_NODEFILE']
    nodeList = set(open(hostfile).readlines())
    nodeList = map(lambda s: s.strip(), nodeList)
    return nodeList


def callFunc(parList):
    nodeType,nodeName, micId, jobId = parList
    HOST = nodeName
    CPU_COMMAND = '~/pyJobDistributor/printHost.cpu'
    MIC_COMMAND = 'ssh mic%d ~/pyJobDistributor/printHost.mic' %int(micId)
    if nodeType == 'cpu':
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
    time.sleep(randint(0,10))
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


def setupPar():
    nodeList = getHostList()
    nProcPerNode = nCpuPerNode + nMicPerNode
    procType = ['cpu'] * nCpuPerNode + ['mic'] * nMicPerNode
    micId =  [0] * nCpuPerNode + range(nMicPerNode) 
    jobId = range(nProcPerNode)
    hostName = [nodeList[0]] * nProcPerNode
    parList = vstack([procType, hostName, micId, jobId]).transpose()
    return parList

def runJobAsync():
    nProcPerNode = nCpuPerNode + nMicPerNode
    parList=setupPar()
    pool = Pool(processes=nProcPerNode)
    result = []
    for i in range(len(parList)):
        result.append(pool.apply_async(callFunc,[parList[i]]))
    allFinished = False
    while(not allFinished):
        allFinished = True
        time.sleep(1)
        for i in range(len(parList)):
            if result[i].ready()==False:
                print "Job %d still running" %i
                allFinished = False
            else:
                print result[i].get()
    print "all jobs finished!"

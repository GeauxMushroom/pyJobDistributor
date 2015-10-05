import subprocess
import os
from multiprocessing import Pool
import numpy as np
from numpy import vstack, hstack, genfromtxt

import time
from numpy.random import randint

import shutil

#from DMFT import DMFT_LOOP, DMFT_LOOP_MIC

nCpuPerNode=1
nMicPerNode=0
nThrdsPerCPU=20
nThrdsPerMIC=240

def genInput(procType,jobId,wkDir):
    #generate the input files for CTQMC solver
    dirName = wkDir+'/jobDir_%d' % jobId
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    par = np.genfromtxt('parameters.txt',skip_header=1)[jobId]
    #par:
    #beta e0 g w0 a U0 t**2 D iterPM iterPC iterTot iterWarm noThrds rngSeed
    if(procType == 'mic'):
        #on mic, split job into more markov chains/threads
        par[10] = par[10] / nThrdsPerMIC * nThrdsPerCPU
        par[12] = nThrdsPerMIC

    e0 = par[1]
    g = par[2]
    w0 = par[3]
    a = par[4]
    U0 = par[5]
    ee= w0 * np.sqrt(1 + g * g)

    hamiltonian = np.zeros([8,8]) 
    hamiltonian[0][0] = ee;
    hamiltonian[1][1] = ee + e0 + g * w0;
    hamiltonian[2][2] = ee + e0 + g * w0;
    hamiltonian[3][3] = ee + 2.0 * e0 + U0 - (a - 1.0) * g * w0;
    hamiltonian[0][4] = w0;
    hamiltonian[1][5] = w0;
    hamiltonian[2][6] = w0;
    hamiltonian[3][7] = w0;
    hamiltonian[4][0] = w0;
    hamiltonian[5][1] = w0;
    hamiltonian[6][2] = w0;
    hamiltonian[7][3] = w0;

    hamiltonian[4][4] = ee;
    hamiltonian[5][5] = ee + e0 - g * w0;
    hamiltonian[6][6] = ee + e0 - g * w0;
    hamiltonian[7][7] = ee + 2.0 * e0 + U0 + (a-1.0) * g * w0;
    np.savetxt(dirName+'/hamiltonian_dhm.txt', hamiltonian, fmt='%1.3f',delimiter='\t')
    par2 = [par[0],-0.5*U0, U0, par[6],par[7],par[8],par[9],par[10],par[11],par[12],par[13]]
    np.savetxt(dirName+'/parameters.txt', [par2], fmt='%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%d\t%d\t%d\t%d\t%d\t%d')

    return


def getHostList():
    hostfile = os.environ.get('PBS_NODEFILE')
    if hostfile == None:
        return ["LOCAL"]
    else:
        nodeList = set(open(hostfile).readlines())
        nodeList = map(lambda s: s.strip(), nodeList)
        return nodeList

def genWkDir():
    jobId = os.environ.get('PBS_JOBID')
    if jobId == None:
        dirName = os.getcwd()+"/LocalJob_"+str(time.time())+'/'
    else:
        dirName='/work/sfeng/'+jobId+'/'

    #if not os.path.exists(dirName):
    os.makedirs(dirName)

    return dirName
    

def callFunc(parList):
    [nodeName,procType, procId], jobId, jobPar, wkDir = parList
    genInput(procType,jobId, wkDir)
    prevPath = os.path.abspath(os.getcwd())
    #print "Proc %d is going to work on Job %d" %(int(procId), jobId)
    dirName= wkDir+'/jobDir_%d/' % jobId

    hostNameFile=open(dirName+'/hostName.txt','w')
    hostNameFile.write(nodeName)
    hostNameFile.close()

    if procType == 'mic':
        shutil.copyfile('/home/sfeng4/GeauxCTQMC/ctqmc_dmft_mic',dirName+'ctqmc_dmft_mic')
        os.chmod(dirName+'ctqmc_dmft_mic', 0744)
        sub=subprocess.Popen('python /home/sfeng4/pyJobDistributor/run_DMFT_mic.py',shell='True',stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd = dirName)
    else:
        shutil.copyfile('/home/sfeng4/GeauxCTQMC/ctqmc_dmft',dirName+'ctqmc_dmft')
        os.chmod(dirName+'ctqmc_dmft', 0744)
        sub=subprocess.Popen('python /home/sfeng4/pyJobDistributor/run_DMFT_cpu.py',shell='True',stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd = dirName)

    sub.wait()
    result = sub.stdout.readlines()
    error = sub.stderr.readlines()

    return result,error

def callFuncDummy(parList):
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

    nJobs = len(np.genfromtxt('parameters.txt',skip_header=1))
    print "nJobs=", nJobs
    jobPres = [ None ] * nJobs 
    jobFinished = [False] * nJobs
    jobTaken = [False] * nJobs
    jobPar = randint(10, size= nJobs)+5

    wkDir = genWkDir()

    for i in range(nProc):
        result.append(pool.apply_async(callFunc,[[procList[i],i,jobPar[i], wkDir]]))
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
                    result.append(pool.apply_async(callFunc,[[procList[i],j,jobPar[j], wkDir]]))
                    currentJob[i] = j
                    jobTaken[j] = True
                    print "Proc %d starts working on Job %d" %(i,j)

        allFinished = True
        for j in range(nJobs):
            if(not jobFinished[j]):
                allFinished = False

    print "all jobs finished!"

"""
Created on Wed Feb 20 18:14:41 2019

@author: ke
"""


from textwrap import dedent as d
import json
import dash
from dash.dependencies import Input, Output,State
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import os
import root_pandas
import pandas as pd
import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector

from matplotlib import pyplot as plt
import numpy as np
from numpy import cos,sin,tan,sqrt,absolute,real,conjugate,imag,abs,max,min

import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector

import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import Data, Layout, Figure
from plotly.graph_objs import Scatter


dht_D0=root_pandas.read_root('model_BplusD0.root',key='DecayTree')
dht_D0['W_PX_TRUE']=dht_D0['B_PX_TRUE']-dht_D0['D0_PX_TRUE']
dht_D0['W_PY_TRUE']=dht_D0['B_PY_TRUE']-dht_D0['D0_PY_TRUE']
dht_D0['W_PZ_TRUE']=dht_D0['B_PZ_TRUE']-dht_D0['D0_PZ_TRUE']
dht_D0['W_E_TRUE']=dht_D0['B_E_TRUE']-dht_D0['D0_E_TRUE']
dh_D0=dht_D0.head(100)


dht_Dst=root_pandas.read_root('model_tree.root',key='DecayTree')
dht_Dst['W_PX_TRUE']=dht_Dst['B_PX_TRUE']-dht_Dst['Dst_PX_TRUE']
dht_Dst['W_PY_TRUE']=dht_Dst['B_PY_TRUE']-dht_Dst['Dst_PY_TRUE']
dht_Dst['W_PZ_TRUE']=dht_Dst['B_PZ_TRUE']-dht_Dst['Dst_PZ_TRUE']
dht_Dst['W_E_TRUE']=dht_Dst['B_E_TRUE']-dht_Dst['Dst_E_TRUE']
dh_Dst=dht_Dst.head(100)

dht_2460=root_pandas.read_root('model_2460_tree.root',key='DecayTree')
dht_2460['W_PX_TRUE']=dht_2460['B_PX_TRUE']-dht_2460['Dstst_PX_TRUE']
dht_2460['W_PY_TRUE']=dht_2460['B_PY_TRUE']-dht_2460['Dstst_PY_TRUE']
dht_2460['W_PZ_TRUE']=dht_2460['B_PZ_TRUE']-dht_2460['Dstst_PZ_TRUE']
dht_2460['W_E_TRUE']=dht_2460['B_E_TRUE']-dht_2460['Dstst_E_TRUE']
dh_2460=dht_2460.head(100)

dht_2420=root_pandas.read_root('model_2420_tree.root',key='DecayTree')
dht_2420['W_PX_TRUE']=dht_2420['B_PX_TRUE']-dht_2420['Dstst_PX_TRUE']
dht_2420['W_PY_TRUE']=dht_2420['B_PY_TRUE']-dht_2420['Dstst_PY_TRUE']
dht_2420['W_PZ_TRUE']=dht_2420['B_PZ_TRUE']-dht_2420['Dstst_PZ_TRUE']
dht_2420['W_E_TRUE']=dht_2420['B_E_TRUE']-dht_2420['Dstst_E_TRUE']
dh_2420=dht_2460.head(100)



def particles(filename):
    if filename=='dh_D0':
        df=dh_D0
    if filename=='dh_Dst':
        df=dh_Dst
    if filename=='dh_2460':
        df=dh_2460
    if filename=='dh_2420':
        df=dh_2420
    if filename=='dh_D0':
        B=LorentzVector(df['B_PX_TRUE'],df['B_PY_TRUE'],df['B_PZ_TRUE'],df['B_E_TRUE'])
        W=LorentzVector(df['W_PX_TRUE'],df['W_PY_TRUE'],df['W_PZ_TRUE'],df['W_E_TRUE'])
        D0=LorentzVector(df['D0_PX_TRUE'],df['D0_PY_TRUE'],df['D0_PZ_TRUE'],df['D0_E_TRUE'])
        tau=LorentzVector(df['Tau_PX_TRUE'],df['Tau_PY_TRUE'],df['Tau_PZ_TRUE'],df['Tau_E_TRUE'])
        nuB=LorentzVector(df['B_nu_PX_TRUE'],df['B_nu_PY_TRUE'],df['B_nu_PZ_TRUE'],df['B_nu_E_TRUE'])
        K=LorentzVector(df['D0_K_PX_TRUE'],df['D0_K_PY_TRUE'],df['D0_K_PZ_TRUE'],df['D0_K_E_TRUE'])
        piD0=LorentzVector(df['D0_Pi_PX_TRUE'],df['D0_Pi_PY_TRUE'],df['D0_Pi_PZ_TRUE'],df['D0_Pi_E_TRUE'])
        pitau1=LorentzVector(df['Tau_Pi1_PX_TRUE'],df['Tau_Pi1_PY_TRUE'],df['Tau_Pi1_PZ_TRUE'],df['Tau_Pi1_E_TRUE'])
        pitau2=LorentzVector(df['Tau_Pi2_PX_TRUE'],df['Tau_Pi2_PY_TRUE'],df['Tau_Pi2_PZ_TRUE'],df['Tau_Pi2_E_TRUE'])
        pitau3=LorentzVector(df['Tau_Pi3_PX_TRUE'],df['Tau_Pi3_PY_TRUE'],df['Tau_Pi3_PZ_TRUE'],df['Tau_Pi3_E_TRUE'])
        nutau=LorentzVector(df['Tau_nu_PX_TRUE'],df['Tau_nu_PY_TRUE'],df['Tau_nu_PZ_TRUE'],df['Tau_nu_E_TRUE'])
        particle_list=[B,W,D0,tau,nuB,K,piD0,pitau1,pitau2,pitau3,nutau]
 
    elif filename=='dh_Dst':
        B=LorentzVector(df['B_PX_TRUE'],df['B_PY_TRUE'],df['B_PZ_TRUE'],df['B_E_TRUE'])
        W=LorentzVector(df['W_PX_TRUE'],df['W_PY_TRUE'],df['W_PZ_TRUE'],df['W_E_TRUE'])
        Dst=LorentzVector(df['Dst_PX_TRUE'],df['Dst_PY_TRUE'],df['Dst_PZ_TRUE'],df['Dst_E_TRUE'])
        tau=LorentzVector(df['Tau_PX_TRUE'],df['Tau_PY_TRUE'],df['Tau_PZ_TRUE'],df['Tau_E_TRUE'])
        D0=LorentzVector(df['D0_PX_TRUE'],df['D0_PY_TRUE'],df['D0_PZ_TRUE'],df['D0_E_TRUE'])
        nuB=LorentzVector(df['B_nu_PX_TRUE'],df['B_nu_PY_TRUE'],df['B_nu_PZ_TRUE'],df['B_nu_E_TRUE'])
        K=LorentzVector(df['D0_K_PX_TRUE'],df['D0_K_PY_TRUE'],df['D0_K_PZ_TRUE'],df['D0_K_E_TRUE'])
        piDst=LorentzVector(df['Dst_Pi_PX_TRUE'],df['Dst_Pi_PY_TRUE'],df['Dst_Pi_PZ_TRUE'],df['Dst_Pi_E_TRUE'])
        piK=LorentzVector(df['D0_Pi_PX_TRUE'],df['D0_Pi_PY_TRUE'],df['D0_Pi_PZ_TRUE'],df['D0_Pi_E_TRUE'])
        pitau1=LorentzVector(df['Tau_Pi1_PX_TRUE'],df['Tau_Pi1_PY_TRUE'],df['Tau_Pi1_PZ_TRUE'],df['Tau_Pi1_E_TRUE'])
        pitau2=LorentzVector(df['Tau_Pi2_PX_TRUE'],df['Tau_Pi2_PY_TRUE'],df['Tau_Pi2_PZ_TRUE'],df['Tau_Pi2_E_TRUE'])
        pitau3=LorentzVector(df['Tau_Pi3_PX_TRUE'],df['Tau_Pi3_PY_TRUE'],df['Tau_Pi3_PZ_TRUE'],df['Tau_Pi3_E_TRUE'])
        nutau=LorentzVector(df['Tau_nu_PX_TRUE'],df['Tau_nu_PY_TRUE'],df['Tau_nu_PZ_TRUE'],df['Tau_nu_E_TRUE'])
        particle_list=[B,W,Dst,D0,tau,nuB,K,piDst,piK,pitau1,pitau2,pitau3,nutau]
    elif filename=='dh_2460' : 
        B=LorentzVector(df['B_PX_TRUE'],df['B_PY_TRUE'],df['B_PZ_TRUE'],df['B_E_TRUE'])
        W=LorentzVector(df['W_PX_TRUE'],df['W_PY_TRUE'],df['W_PZ_TRUE'],df['W_E_TRUE'])
        Dstst=LorentzVector(df['Dstst_PX_TRUE'],df['Dstst_PY_TRUE'],df['Dstst_PZ_TRUE'],df['Dstst_E_TRUE'])
        Dst=LorentzVector(df['Dst_PX_TRUE'],df['Dst_PY_TRUE'],df['Dst_PZ_TRUE'],df['Dst_E_TRUE'])
        tau=LorentzVector(df['Tau_PX_TRUE'],df['Tau_PY_TRUE'],df['Tau_PZ_TRUE'],df['Tau_E_TRUE'])
        D0=LorentzVector(df['D0_PX_TRUE'],df['D0_PY_TRUE'],df['D0_PZ_TRUE'],df['D0_E_TRUE'])
        nuB=LorentzVector(df['B_nu_PX_TRUE'],df['B_nu_PY_TRUE'],df['B_nu_PZ_TRUE'],df['B_nu_E_TRUE'])
        K=LorentzVector(df['D0_K_PX_TRUE'],df['D0_K_PY_TRUE'],df['D0_K_PZ_TRUE'],df['D0_K_E_TRUE'])
        piDst=LorentzVector(df['Dst_Pi_PX_TRUE'],df['Dst_Pi_PY_TRUE'],df['Dst_Pi_PZ_TRUE'],df['Dst_Pi_E_TRUE'])
        piDstst=LorentzVector(df['Dstst_Pi_PX_TRUE'],df['Dstst_Pi_PY_TRUE'],df['Dstst_Pi_PZ_TRUE'],df['Dstst_Pi_E_TRUE'])
        piD0=LorentzVector(df['D0_Pi_PX_TRUE'],df['D0_Pi_PY_TRUE'],df['D0_Pi_PZ_TRUE'],df['D0_Pi_E_TRUE'])
        pitau1=LorentzVector(df['Tau_Pi1_PX_TRUE'],df['Tau_Pi1_PY_TRUE'],df['Tau_Pi1_PZ_TRUE'],df['Tau_Pi1_E_TRUE'])
        pitau2=LorentzVector(df['Tau_Pi2_PX_TRUE'],df['Tau_Pi2_PY_TRUE'],df['Tau_Pi2_PZ_TRUE'],df['Tau_Pi2_E_TRUE'])
        pitau3=LorentzVector(df['Tau_Pi3_PX_TRUE'],df['Tau_Pi3_PY_TRUE'],df['Tau_Pi3_PZ_TRUE'],df['Tau_Pi3_E_TRUE'])
        nutau=LorentzVector(df['Tau_nu_PX_TRUE'],df['Tau_nu_PY_TRUE'],df['Tau_nu_PZ_TRUE'],df['Tau_nu_E_TRUE'])
        particle_list=[B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]      
    elif filename=='dh_2420':
        B=LorentzVector(df['B_PX_TRUE'],df['B_PY_TRUE'],df['B_PZ_TRUE'],df['B_E_TRUE'])
        W=LorentzVector(df['W_PX_TRUE'],df['W_PY_TRUE'],df['W_PZ_TRUE'],df['W_E_TRUE'])
        Dstst=LorentzVector(df['Dstst_PX_TRUE'],df['Dstst_PY_TRUE'],df['Dstst_PZ_TRUE'],df['Dstst_E_TRUE'])
        Dst=LorentzVector(df['Dst_PX_TRUE'],df['Dst_PY_TRUE'],df['Dst_PZ_TRUE'],df['Dst_E_TRUE'])
        tau=LorentzVector(df['Tau_PX_TRUE'],df['Tau_PY_TRUE'],df['Tau_PZ_TRUE'],df['Tau_E_TRUE'])
        D0=LorentzVector(df['D0_PX_TRUE'],df['D0_PY_TRUE'],df['D0_PZ_TRUE'],df['D0_E_TRUE'])
        nuB=LorentzVector(df['B_nu_PX_TRUE'],df['B_nu_PY_TRUE'],df['B_nu_PZ_TRUE'],df['B_nu_E_TRUE'])
        K=LorentzVector(df['D0_K_PX_TRUE'],df['D0_K_PY_TRUE'],df['D0_K_PZ_TRUE'],df['D0_K_E_TRUE'])
        piDst=LorentzVector(df['Dst_Pi_PX_TRUE'],df['Dst_Pi_PY_TRUE'],df['Dst_Pi_PZ_TRUE'],df['Dst_Pi_E_TRUE'])
        piDstst=LorentzVector(df['Dstst_Pi_PX_TRUE'],df['Dstst_Pi_PY_TRUE'],df['Dstst_Pi_PZ_TRUE'],df['Dstst_Pi_E_TRUE'])
        piD0=LorentzVector(df['D0_Pi_PX_TRUE'],df['D0_Pi_PY_TRUE'],df['D0_Pi_PZ_TRUE'],df['D0_Pi_E_TRUE'])
        pitau1=LorentzVector(df['Tau_Pi1_PX_TRUE'],df['Tau_Pi1_PY_TRUE'],df['Tau_Pi1_PZ_TRUE'],df['Tau_Pi1_E_TRUE'])
        pitau2=LorentzVector(df['Tau_Pi2_PX_TRUE'],df['Tau_Pi2_PY_TRUE'],df['Tau_Pi2_PZ_TRUE'],df['Tau_Pi2_E_TRUE'])
        pitau3=LorentzVector(df['Tau_Pi3_PX_TRUE'],df['Tau_Pi3_PY_TRUE'],df['Tau_Pi3_PZ_TRUE'],df['Tau_Pi3_E_TRUE'])
        nutau=LorentzVector(df['Tau_nu_PX_TRUE'],df['Tau_nu_PY_TRUE'],df['Tau_nu_PZ_TRUE'],df['Tau_nu_E_TRUE'])
        particle_list=[B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]      
    return particle_list

def change_frame(COM,filename):
    res=[]
    list_particles=particles(filename)
    for i in range(len(list_particles)):
        newvec=list_particles[i].boost(-COM.boostp3)
        res.append(newvec)
    return res



def calc_angles(filename):    
    if filename=='dh_D0':
        particle_list=particles(filename)
        [B,W,D0,tau,nuB,K,piD0,pitau1,pitau2,pitau3,nutau]=particle_list


    elif filename=='dh_Dst':
        particle_list=particles(filename)
        [B,W,Dst,D0,tau,nuB,K,piDst,piK,pitau1,pitau2,pitau3,nutau]=particle_list

    elif filename=='dh_2420': 
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
    elif filename=='dh_2460':
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
    if filename=='dh_D0':
        nouvtau=tau.boost(-(tau+nuB).boostp3)
        nouvnu=nuB.boost(-(tau+nuB).boostp3)
        unittau=(nouvtau.p3).unit
        unitnu=(nouvnu.p3).unit
        nnewtau=tau.boost(-B.boostp3)
        unitau=nnewtau.unit
        nouvpi=piD0.boost(-(piD0+K).boostp3)
        nouvK=K.boost(-(piD0+K).boostp3)
        nouvD0=D0.boost(-B.boostp3)
        unitD0=(nouvD0.p3).unit
        unitK=(nouvK.p3).unit
        costhetast=unitD0.dot(unitK)
        costhetal=unitD0.dot(unittau)
        nnewD0=D0.boost(-B.boostp3)
        uniD0=nnewD0.unit
        nnormal1=unitD0.cross(unitK)
        normal1=nnormal1.unit
        nnormal2=unitD0.cross(unitau)
        normal2=nnormal2.unit
        pparallel=normal1.cross(unitD0)
        parallel=pparallel.unit
        co = normal1.dot(normal2)
        si = parallel.dot(normal2)
        chi = np.arctan2(si,co)

        angles=[costhetast,costhetal,chi,nouvpi,nouvK,nouvD0]
    if filename=='dh_Dst':
        nouvtau=tau.boost(-(tau+nuB).boostp3)
        nouvnu=nuB.boost(-(tau+nuB).boostp3)
        unittau=(nouvtau.p3).unit
        unitnu=(nouvnu.p3).unit
        nnewtau=tau.boost(-B.boostp3)
        unitau=nnewtau.unit
        nouvpi=piDst.boost(-(piDst+D0).boostp3)
        nouvD0=D0.boost(-(piDst+D0).boostp3)
        nouvDst=D0.boost(-B.boostp3)
        unitDst=(nouvDst.p3).unit
        unitD0=(nouvD0.p3).unit
        nnewD0=D0.boost(-B.boostp3)
        uniD0=nnewD0.unit
        nnormal1=unitDst.cross(uniD0)
        normal1=nnormal1.unit
        nnormal2=unitDst.cross(unitau)
        normal2=nnormal2.unit
        pparallel=normal1.cross(unitDst)
        parallel=pparallel.unit
        co = normal1.dot(normal2)
        si = parallel.dot(normal2)
        chi = np.arctan2(si,co)
        costhetast=unitD0.dot(unitDst)
        costhetal=unitDst.dot(unittau)



        angles=[costhetast,costhetal,chi,nouvpi,nouvD0,nouvDst]
        
    if filename=='dh_2460':
        nouvtau=tau.boost(-(tau+nuB).boostp3)
        nouvnu=nuB.boost(-(tau+nuB).boostp3)
        unittau=(nouvtau.p3).unit
        unitnu=(nouvnu.p3).unit
        nnewtau=tau.boost(-B.boostp3)
        unitau=nnewtau.unit
        nouvpi=piDstst.boost(-(piDstst+Dst).boostp3)
        nouvDst=D0.boost(-(piDstst+Dst).boostp3)
        nouvDstst=Dstst.boost(-B.boostp3)
        unitDstst=(nouvDstst.p3).unit
        unitDst=(nouvDst.p3).unit
        nnewDst=Dst.boost(-B.boostp3)
        uniDst=nnewDst.unit
        nnormal1=unitDstst.cross(uniDst)
        normal1=nnormal1.unit
        nnormal2=unitDstst.cross(unitau)
        normal2=nnormal2.unit
        pparallel=normal1.cross(unitDstst)
        parallel=pparallel.unit
        co = normal1.dot(normal2)
        si = parallel.dot(normal2)
        chi = np.arctan2(si,co)
        costhetast=unitDst.dot(unitDstst)
        costhetal=unitDstst.dot(unittau)
        angles=[costhetast,costhetal,chi,nouvpi,nouvtau,nouvDst,nouvDstst,nouvnu]
    if filename=='dh_2420':
        nouvtau=tau.boost(-(tau+nuB).boostp3)
        nouvnu=nuB.boost(-(tau+nuB).boostp3)
        unittau=(nouvtau.p3).unit
        unitnu=(nouvnu.p3).unit
        nnewtau=tau.boost(-B.boostp3)
        unitau=nnewtau.unit
        nouvpi=piDstst.boost(-(piDstst+Dst).boostp3)
        nouvDst=D0.boost(-(piDstst+Dst).boostp3)
        nouvDstst=Dstst.boost(-B.boostp3)
        unitDstst=(nouvDstst.p3).unit
        unitDst=(nouvDst.p3).unit
        nnewDst=Dst.boost(-B.boostp3)
        uniDst=nnewDst.unit
        nnormal1=unitDstst.cross(uniDst)
        normal1=nnormal1.unit
        nnormal2=unitDstst.cross(unitau)
        normal2=nnormal2.unit
        pparallel=normal1.cross(unitDstst)
        parallel=pparallel.unit
        co = normal1.dot(normal2)
        si = parallel.dot(normal2)
        chi = np.arctan2(si,co)
        costhetast=unitDst.dot(unitDstst)
        costhetal=unitDstst.dot(unittau)
        angles=[costhetast,costhetal,chi,nouvpi,nouvtau,nouvDst,nouvDstst,nouvnu]

    return angles

def com(filename,frame):
    if filename=='dh_D0':
        df=dh_D0
    if filename=='dh_Dst':
        df=dh_Dst
    if filename=='dh_2460':
        df=dh_2460
    if filename=='dh_2420':
        df=dh_2420
        
    if frame=='B' or frame=='W':
        COM=LorentzVector(df[frame+'_PX_TRUE'],df[frame+'_PY_TRUE'],df[frame+'_PZ_TRUE'],df[frame+'_E_TRUE'])
    elif frame=='D0':
        if filename=='dh_D0':
            COM=LorentzVector(df['D0_PX_TRUE'],df['D0_PY_TRUE'],df['D0_PZ_TRUE'],df['D0_E_TRUE'])
        elif filename=='dh_Dst':
            COM=LorentzVector(df['Dst_PX_TRUE'],df['Dst_PY_TRUE'],df['Dst_PZ_TRUE'],df['Dst_E_TRUE'])
        elif filename=='dh_2460' :
            COM=LorentzVector(df['Dstst_PX_TRUE'],df['Dstst_PY_TRUE'],df['Dstst_PZ_TRUE'],df['Dstst_E_TRUE'])   
        elif filename=='dh_2420':
            COM=LorentzVector(df['Dstst_PX_TRUE'],df['Dstst_PY_TRUE'],df['Dstst_PZ_TRUE'],df['Dstst_E_TRUE'])   
    return COM

def coordinates(filename):
    if filename=='dh_D0':
        df=dh_D0
    if filename=='dh_Dst':
        df=dh_Dst
    if filename=='dh_2460':
        df=dh_2460
    if filename=='dh_2420':
        df=dh_2420
    PV_X,PV_Y,PV_Z=(df['B_Ori_z_TRUE'][i],df['B_Ori_x_TRUE'][i],df['B_Ori_y_TRUE'][i])
    B_X,B_Y,B_Z=(df['B_End_z_TRUE'][i],df['B_End_x_TRUE'][i],df['B_End_y_TRUE'][i])
    pitau1_X=df['Tau_Pi1_PZ_TRUE'][i]*dis/df['Tau_Pi1_P_TRUE'][i]+tau_X
    pitau1_Y=df['Tau_Pi1_PX_TRUE'][i]*dis/df['Tau_Pi1_P_TRUE'][i]+tau_Y
    pitau1_Z=df['Tau_Pi1_PY_TRUE'][i]*dis/df['Tau_Pi1_P_TRUE'][i]+tau_Z
    pitau2_X=df['Tau_Pi2_PZ_TRUE'][i]*dis/df['Tau_Pi2_P_TRUE'][i]+tau_X
    pitau2_Y=df['Tau_Pi2_PX_TRUE'][i]*dis/df['Tau_Pi2_P_TRUE'][i]+tau_Y
    pitau2_Z=df['Tau_Pi2_PY_TRUE'][i]*dis/df['Tau_Pi2_P_TRUE'][i]+tau_Z
    pitau3_X=df['Tau_Pi3_PZ_TRUE'][i]*dis/df['Tau_Pi3_P_TRUE'][i]+tau_X
    pitau3_Y=df['Tau_Pi3_PX_TRUE'][i]*dis/df['Tau_Pi3_P_TRUE'][i]+tau_Y
    pitau3_Z=df['Tau_Pi3_PY_TRUE'][i]*dis/df['Tau_Pi3_P_TRUE'][i]+tau_Z
    nutau_X=df['Tau_nu_PZ_TRUE'][i]*dis/df['Tau_nu_P_TRUE'][i]+tau_X
    nutau_Y=df['Tau_nu_PX_TRUE'][i]*dis/df['Tau_nu_P_TRUE'][i]+tau_Y
    nutau_Z=df['Tau_nu_PY_TRUE'][i]*dis/df['Tau_nu_P_TRUE'][i]+tau_Z
    tau_X,tau_Y,tau_Z=[df['Tau_End_z_TRUE'][i],df['Tau_End_x_TRUE'][i],df['Tau_End_y_TRUE'][i]]
    nu_X=df['B_nu_PZ_TRUE'][i]*dis/df['B_nu_P_TRUE'][i]+B_X
    nu_Y=df['B_nu_PX_TRUE'][i]*dis/df['B_nu_P_TRUE'][i]+B_Y
    nu_Z=df['B_nu_PY_TRUE'][i]*dis/df['B_nu_P_TRUE'][i]+B_Z
    if filename=='dh_D0':
        D0_X,D0_Y,D0_Z=(df['D0_End_z_TRUE'][i],df['D0_End_x_TRUE'][i],df['D0_End_y_TRUE'][i])
        K_X=df['D0_K_PZ_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_X
        K_Y=df['D0_K_PX_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Y
        K_Z=df['D0_K_PY_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Z

        piD0_X=df['D0_Pi_PZ_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_X
        piD0_Y=df['D0_Pi_PX_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Y
        piD0_Z=df['D0_Pi_PY_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Z


        coord=[PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piD0_X,piD0_Y,piD0_Z,pitau1_X,pitau1_Y,pitau1_Z,
               pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]
    if filename=='dh_Dst':
        Dst_X,Dst_Y,Dst_Z=(df['Dst_End_z_TRUE'][i],df['Dst_End_x_TRUE'][i],df['Dst_End_y_TRUE'][i])
        D0_X,D0_Y,D0_Z=[df['D0_End_z_TRUE'][i],df['D0_End_x_TRUE'][i],df['D0_End_y_TRUE'][i]]
        K_X=df['D0_K_PZ_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_X
        K_Y=df['D0_K_PX_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Y
        K_Z=df['D0_K_PY_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Z
        piK_X=df['D0_Pi_PZ_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_X
        piK_Y=df['D0_Pi_PX_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Y
        piK_Z=df['D0_Pi_PY_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Z
        piDst_X=df['Dst_Pi_PZ_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_X
        piDst_Y=df['Dst_Pi_PX_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_Y
        piDst_Z=df['Dst_Pi_PY_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_Z

        coord=[PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,Dst_X,Dst_Y,Dst_Z,
               D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piDst_X,piDst_Y,piDst_Z,
               piK_X,piK_Y,piK_Z,pitau1_X,pitau1_Y,pitau1_Z,
               pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]
    
    if filename=='dh_2420' :
        Dstst_X,Dstst_Y,Dstst_Z=(df['Dstst_End_z_TRUE'][i],df['Dstst_End_x_TRUE'][i],df['Dstst_End_y_TRUE'][i])
        Dst_X,Dst_Y,Dst_Z=(df['Dst_End_z_TRUE'][i],df['Dst_End_x_TRUE'][i],df['Dst_End_y_TRUE'][i])
        D0_X,D0_Y,D0_Z=[df['D0_End_z_TRUE'][i],df['D0_End_x_TRUE'][i],df['D0_End_y_TRUE'][i]]
        K_X=df['D0_K_PZ_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_X
        K_Y=df['D0_K_PX_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Y 
        K_Z=df['D0_K_PY_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Z
        piK_X=df['D0_Pi_PZ_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_X
        piK_Y=df['D0_Pi_PX_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Y
        piK_Z=df['D0_Pi_PY_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Z
        piDst_X=df['Dst_Pi_PZ_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_X
        piDst_Y=df['Dst_Pi_PX_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_Y
        piDst_Z=df['Dst_Pi_PY_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_Z
        piDstst_X=df['Dstst_Pi_PZ_TRUE'][i]*dis/df['Dstst_Pi_P_TRUE'][i]+Dst_X
        piDstst_Y=df['Dstst_Pi_PX_TRUE'][i]*dis/df['Dstst_Pi_P_TRUE'][i]+Dst_Y
        piDstst_Z=df['Dstst_Pi_PY_TRUE'][i]*dis/df['Dstst_Pi_P_TRUE'][i]+Dst_Z
        coord=[PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,Dst_X,Dst_Y,Dst_Z,Dstst_X,Dstst_Y,Dstst_Z,
               D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piDst_X,piDst_Y,piDst_Z,
               piDstst_X,piDstst_Y,piDstst_Z,
               piK_X,piK_Y,piK_Z,pitau1_X,pitau1_Y,pitau1_Z,
               pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]
    elif filename=='dh_2460':
        Dstst_X,Dstst_Y,Dstst_Z=(df['Dstst_End_z_TRUE'][i],df['Dstst_End_x_TRUE'][i],df['Dstst_End_y_TRUE'][i])
        Dst_X,Dst_Y,Dst_Z=(df['Dst_End_z_TRUE'][i],df['Dst_End_x_TRUE'][i],df['Dst_End_y_TRUE'][i])
        D0_X,D0_Y,D0_Z=[df['D0_End_z_TRUE'][i],df['D0_End_x_TRUE'][i],df['D0_End_y_TRUE'][i]]
        K_X=df['D0_K_PZ_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_X
        K_Y=df['D0_K_PX_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Y 
        K_Z=df['D0_K_PY_TRUE'][i]*dis/df['D0_K_P_TRUE'][i]+D0_Z
        piK_X=df['D0_Pi_PZ_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_X
        piK_Y=df['D0_Pi_PX_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Y
        piK_Z=df['D0_Pi_PY_TRUE'][i]*dis/df['D0_Pi_P_TRUE'][i]+D0_Z
        piDst_X=df['Dst_Pi_PZ_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_X
        piDst_Y=df['Dst_Pi_PX_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_Y
        piDst_Z=df['Dst_Pi_PY_TRUE'][i]*dis/df['Dst_Pi_P_TRUE'][i]+Dst_Z
        piDstst_X=df['Dstst_Pi_PZ_TRUE'][i]*dis/df['Dstst_Pi_P_TRUE'][i]+Dst_X
        piDstst_Y=df['Dstst_Pi_PX_TRUE'][i]*dis/df['Dstst_Pi_P_TRUE'][i]+Dst_Y
        piDstst_Z=df['Dstst_Pi_PY_TRUE'][i]*dis/df['Dstst_Pi_P_TRUE'][i]+Dst_Z
        coord=[PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,Dst_X,Dst_Y,Dst_Z,Dstst_X,Dstst_Y,Dstst_Z,
               D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piDst_X,piDst_Y,piDst_Z,
               piDstst_X,piDstst_Y,piDstst_Z,
               piK_X,piK_Y,piK_Z,pitau1_X,pitau1_Y,pitau1_Z,
               pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]
    return coord

def calc_xrange(df):
    if df=='dh_D0':
        xrange=max(abs([PV_X,B_X,D0_X,tau_X,nu_X,K_X,piD0_X,pitau1_X,
               pitau2_X,pitau3_X,nutau_X]))
    if df=='dh_Dst':
        xrange=max(abs([PV_X,B_X,Dst_X,D0_X,tau_X,nu_X,K_X,piDst_X,
               piK_X,pitau1_X,pitau2_X,pitau3_X,nutau_X]))
    if df=='dh_2460':
        xrange=max(abs([Dst_X,Dstst_X,D0_X,tau_X,nu_X,K_X,piDst_X,piDstst_X,piK_X,pitau1_X,
               pitau2_X,pitau3_X,nutau_X]))
    elif df=='dh_2420':
        xrange=max(abs([Dst_X,Dstst_X,D0_X,tau_X,nu_X,K_X,piDst_X,piDstst_X,piK_X,pitau1_X,
               pitau2_X,pitau3_X,nutau_X]))
    return xrange

def calc_yrange(df):
    if df=='dh_D0':
        yrange=max(abs([PV_Y,B_X,D0_Y,tau_Y,nu_Y,K_Y,piD0_Y,pitau1_Y,
               pitau2_Y,pitau3_Y,nutau_Y]))
    if df=='dh_Dst':
        yrange=max(abs([PV_Y,B_Y,Dst_Y,D0_Y,tau_Y,nu_Y,K_Y,piDst_Y,
               piK_Y,pitau1_Y,pitau2_Y,pitau3_Y,nutau_Y]))
    if df=='dh_2460':
        yrange=max(abs([Dst_Y,Dstst_Y,D0_Y,tau_Y,nu_Y,K_Y,piDst_Y,piDstst_Y,piK_Y,pitau1_Y,
               pitau2_Y,pitau3_Y,nutau_Y]))
    elif df=='dh_2420':
        yrange=max(abs([Dst_Y,Dstst_Y,D0_Y,tau_Y,nu_Y,K_Y,piDst_Y,piDstst_Y,piK_Y,pitau1_Y,
               pitau2_Y,pitau3_Y,nutau_Y]))
        
    return yrange
def calc_zrange(df):
    if df=='dh_D0':
        zrange=max(abs([PV_Z,B_X,D0_Z,tau_Z,nu_Z,K_Z,piD0_Z,pitau1_Z,
               pitau2_Z,pitau3_Z,nutau_Z]))
    if df=='dh_Dst':
        zrange=max(abs([PV_Z,B_Z,Dst_Z,D0_Z,tau_Z,nu_Z,K_Z,piDst_Z,
               piK_Z,pitau1_Z,pitau2_Z,pitau3_Z,nutau_Z]))
    if df=='dh_2460' :
        zrange=max(abs([Dst_Z,Dstst_Z,D0_Z,tau_Z,nu_Z,K_Z,piDst_Z,piDstst_Z,piK_Z,pitau1_Z,
               pitau2_Z,pitau3_Z,nutau_Z]))
    elif df=='dh_2420':
        zrange=max(abs([Dst_Z,Dstst_Z,D0_Z,tau_Z,nu_Z,K_Z,piDst_Z,piDstst_Z,piK_Z,pitau1_Z,
               pitau2_Z,pitau3_Z,nutau_Z]))
    return zrange
############################################################################

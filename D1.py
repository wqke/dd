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
from functions import change_frame,calc_angles,particles,calc_xrange,calc_yrange,calc_zrange,com,coordinates



styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}



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
dh_2420=dht_2420.head(100)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)
app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/EQZeaW.css'})
server = app.server
#############################################
app.layout = html.Div(children=[
    html.Div([
        html.Div([
            html.Div([
                html.H2('B decay visualisation',
                                style={
                                'position': 'relative',
                                'top': '0px',
                                'left': '10px',
                                'font-family': 'Dosis',
                                'display': 'inline',
                                'font-size':'5.0rem',
                                'color': '#4D637F'
                                 }),
                        ]),
              
                    
                    
                dcc.Dropdown(
                    options=[{'label': 'D0', 'value': 'dh_D0'},
                                {'label': 'D*-', 'value': 'dh_Dst'},
                                {'label': 'D1(2420)0', 'value': 'dh_2420'},
                                {'label': 'D2*(2460)0', 'value': 'dh_2460'}],

                     value='dh_Dst',
                     id='dropdown-file'
                ),
                    
                    
                    
                html.Div([
                html.Br(),
                html.P('Choose ranges :',
                                style={
                                'font-family': 'Dosis',
                                'display': 'inline',
                                'font-size': '3rem',
                                'color': '#4D637F'
                                 })
                                ],style={'marginLeft':0}),
                html.Br(),
                html.P('costheta* : ',
                                style={
                                        'display':'inline-block',
                                        'verticalAlign': 'top',
                                        'marginRight': '10px'
                                }
                         ),
                html.Div([     
                dcc.RangeSlider(
                    id='choose-thetast',
                    min=-1,
                    max=1,
                    value=[-1,1], step=0.1
                    ),
                    html.Div(id='output-range-thetast')
                ],style={'width':300,'display':'inline-block','marginBottom':10,'marginLeft':8}),
                html.Br(),
                html.P('costheta_l :',
                            style={
                                    'display':'inline-block',
                                    'verticalAlign': 'top',
                                    'marginRight': '10px'
                            }
                ),
                html.Div([
                    dcc.RangeSlider(
                            id='choose-thetal',
                            min=-1, max=1, value=[-1,1], step=0.1,
                    ),
                    html.Div(id='output-range-thetal')
                ], style={'width':300, 'display':'inline-block', 'marginBottom':11}),

                html.Br(),
                html.P('chi : ',
                            style={
                                    'display':'inline-block',
                                    'verticalAlign': 'top',
                                    'marginRight': '10px'
                            }
                ),
                html.Div([

                    dcc.RangeSlider(
                            id='choose-chi',
                            min=-3.14, max=3.14, value=[-3.14,3.14], step=0.1,
                    ),
                    html.Div(id='output-range-chi')
                        ], style={'width':300, 'display':'inline-block', 'marginBottom':10,'marginLeft':46}),

                        ], style={'margin':0} ),

                html.P('Phase space of selected ranges',
                        style={
                    'font-family': 'Dosis',
                    'display': 'inline',
                    'font-size': '3.0rem',
                    'color': '#4D637F'
                }
                ),

                dcc.Graph(id = 'phase-space'),
                html.Div([
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-result', style=styles['pre']),
            ], className='six columns'),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        html.P('Show angles',
                 style={
                        'font-family': 'Dosis',
                        'display': 'inline',
                        'font-size': '3rem',
                        'color': '#4D637F'}),

        html.Br(),
        html.P('All flight distances are fixed to 1.',
                         style={
                        'font-family': 'Dosis',
                        'display': 'inline',
                        'font-size': '1.5rem',
                        'color': '#4D637F'}),
        dcc.Dropdown(
                    options=[{'label': 'Show theta* ', 'value': 'D0'},
                                        {'label': 'Show theta_l ', 'value': 'W'},
                                        {'label': 'Show theta* and theta_l', 'value': 'two'}],
                        value='D0',
                        id='dropdown-angle'
                ),


        dcc.Graph(id='show-angle'),
        ], className='six columns', style={'margin':0}),
        html.Div([
                dcc.RadioItems(
                    options=[
                        {'label':'3D one event','value':'3D'},
                        {'label':'XY projection','value':'XY'},
                        {'label':'YZ projection','value':'YZ'},
                        {'label':'ZX projection','value':'ZX'}
                    ],
                    value='3D',
                    id='which-D'
                ),
                html.Br(),
                html.P('Select frame:',
                 style={
                    'font-family': 'Dosis',
                    'display': 'inline',
                    'font-size': '3rem',
                    'color': '#4D637F'
                }),

                dcc.Dropdown(
                    options=[{'label': 'Lab', 'value': 'lab'},
                                        {'label': 'COM of B', 'value': 'B'},
                                        {'label': 'COM of (B-W)', 'value': 'D0'},
                                        {'label': 'COM of W', 'value': 'W'}],

                        value='lab',
                        id='dropdown-frame'
                ),

                dcc.Graph(id='frame-graph',style={'height':500,'width':500}),
                html.Br(),
                html.P('The line color in the 3D plot stands for the magnitude of the momentum in the LAB frame, darker color means larger momentum .',
                         style={
                        'font-family': 'Dosis',
                        'display': 'inline',
                        'font-size': '1.5rem',
                        'color': '#4D637F'}),
                html.Br(),
                html.P('Histograms',
                         style={
                        'font-family': 'Dosis',

                        'display': 'inline',
                        'font-size': '3rem',
                        'color': '#4D637F'})
                ], className='six columns', style={'margin':0}),
        html.Br(),
        html.Div([
                html.Div([

                        dcc.Dropdown(
                               options=[{'label': 'chi', 'value': 'chid'},
                                        {'label': 'costheta*', 'value': 'std'},
                                        {'label': 'costhetal', 'value': 'ld'},
                                        {'label': 'q2', 'value': 'q2'}],


                                        value='chid',
                                        id='dropdown-dist'
                                        ),
                        dcc.Graph(id='histog'),
                ], className='six columns', style={'margin':0})]),

        ])

#--------------------------------------------------------------
#Callback for the showangle graph, clickData and Dropdown Menu |
#--------------------------------------------------------------

@app.callback(
Output('show-angle', 'figure'),
[Input('phase-space', 'clickData'),
Input('dropdown-angle','value'),
Input('dropdown-file','value')])

def drawangle(selection,choice,filename):

    if filename=='dh_D0':
        df=dh_D0
        particle_list=particles(dh_D0)
        [B,W,D0,tau,nuB,K,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(dh_D0)
        [costhetast,costhetal,chi,nouvpi,nouvK,nouvD0]=angles

    elif filename=='dh_Dst':
        df=dh_Dst
        [B,W,Dst,D0,tau,nuB,K,piDst,piK,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(dh_Dst)
        [costhetast,costhetal,chi,nouvpi,nouvD0,nouvDst]=angles

    elif filename=='dh_2420':
        df=dh_2420
        particle_list=particles(dh_2420)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(dh_2420)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles

    elif filename=='dh_2460':
        df=dh_2460
        particle_list=particles(dh_2460)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(dh_2460)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles 
        

    if selection is None:
        return {}
    else:
        i=selection['points'][0]['pointNumber']

        if choice=='two':
            angleBx,angleBy,angleBz=(0,0,0)
            angleD0x,angleD0y,angleD0z=(nouvD0.z[i]/nouvD0.p[i],nouvD0.x[i]/nouvD0.p[i],nouvD0.y[i]/nouvD0.p[i])
            angleKx,angleKy,angleKz=(nouvK.z[i]/nouvK.p[i]+angleD0x,nouvK.x[i]/nouvK.p[i]+angleD0y,nouvK.y[i]/nouvK.p[i]+angleD0z)
            anglepix,anglepiy,anglepiz=(-nouvpi.z[i]/nouvpi.p[i]+angleD0x,-nouvpi.x[i]/nouvpi.p[i]+angleD0y,-nouvpi.y[i]/nouvpi.p[i]+angleD0z)
            angleQx,angleQy,angleQz=(-angleD0x,-angleD0y,-angleD0z)
            angletaux,angletauy,angletauz=(angleQx+nouvtau.z[i]/nouvtau.p[i],angleQy+nouvtau.x[i]/nouvtau.p[i],angleQz+nouvtau.y[i]/nouvtau.p[i])
            anglenux,anglenuy,anglenuz=(angleQx-nouvtau.z[i]/nouvtau.p[i],angleQy-nouvtau.x[i]/nouvtau.p[i],angleQz-nouvtau.y[i]/nouvtau.p[i])
            d1x,d1y,d1z=(2*angleQx,2*angleQy,2*angleQz)
            d2x,d2y,d2z=(2*angleD0x,2*angleD0y,2*angleD0z)
            dash1=go.Scatter3d(x=[angleQx,d1x],y=[angleQy,d1y],z=[angleQz,d1z],mode='lines',line = dict(color = ('rgb(20, 20, 255)'),width = 3,dash='dot'))
            dash2=go.Scatter3d(x=[angleD0x,d2x],y=[angleD0y,d2y],z=[angleD0z,d2z],mode='lines',line = dict(color = ('rgb(20, 20, 255)'),width = 3,dash='dot'))
            mesh1 = go.Mesh3d(x=[d1x,angletaux,angleQx],y=[d1y,angletauy,angleQy],z=[d1z,angletauz,angleQz],
                opacity=0.4,
                    color='#3E3A3A')

            mesh2 = go.Mesh3d(x=[d2x,angleKx,angleD0x],y=[d2y,angleKy,angleD0y],z=[d2z,angleKz,angleD0z],
                opacity=0.4,
                color='#3E3A3A')
            traceQ=go.Scatter3d(x=[angleBx,angleQx],y=[angleBy,angleQy],z=[angleBz,angleQz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['B', 'W'],textposition='top right',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

            traceD0=go.Scatter3d(x=[angleBx,angleD0x],y=[angleBy,angleD0y],z=[angleBz,angleD0z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'D0'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracenu=go.Scatter3d(x=[angleQx,anglenux],y=[angleQy,anglenuy],z=[angleQz,anglenuz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

            tracetau=go.Scatter3d(x=[angleQx,angletaux],y=[angleQy,angletauy],z=[angleQz,angletauz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

            traceK=go.Scatter3d(x=[angleD0x,angleKx],y=[angleD0y,angleKy],z=[angleD0z,angleKz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

            tracepi=go.Scatter3d(x=[angleD0x,anglepix],y=[angleD0y,anglepiy],z=[angleD0z,anglepiz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            dataangle=[traceQ,traceD0,tracenu,tracetau,traceK,tracepi,dash1,dash2,mesh1,mesh2]

        else:
            COM=LorentzVector(df[choice+'_PX_TRUE'],df[choice+'_PY_TRUE'],df[choice+'_PZ_TRUE'],df[choice+'_E_TRUE'])
            liste_part=change_frame(COM)    
            if filename=='dh_D0':
                [newB,newW,newD0,newtau,newnuB,newK,newpiD0,newpitau1,newpitau2,newpitau3,newnutau]=liste_part

            elif filename=='dh_Dst':
                [newB,newW,newDst,newD0,newtau,newnuB,newK,newpiDst,newpiK,newpitau1,newpitau2,newpitau3,newnutau]=liste_part

            elif filename=='dh_2420':

                [newB,newW,newD0,newtau,newnuB,newK,newpiD0,newpitau1,newpitau2,newpitau3,newnutau]=liste_part

            elif filename=='dh_2460':

                [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=liste_part
    
            

        if choice=='D0':
            angleD0x,angleD0y,angleD0z=(0,0,0)
            angleBx,angleBy,angleBz=(newB.z[i]/newB.p[i],newB.x[i]/newB.p[i],newB.y[i]/newB.p[i])
            angleQx,angleQy,angleQz=(newW.z[i]/newW.p[i]+angleBx,newW.x[i]/newW.p[i]+angleBy,newW.y[i]/newW.p[i]+angleBz)

            d1x,d1y,d1z=(-angleBx,-angleBy,-angleBz)
            angleKx,angleKy,angleKz=(newK.z[i]/newK.p[i]+angleD0x,newK.x[i]/newK.p[i]+angleD0y,newK.y[i]/newK.p[i]+angleD0z)
            anglepix,anglepiy,anglepiz=(newpiD0.z[i]/newpiD0.p[i]+angleD0x,newpiD0.x[i]/newpiD0.p[i]+angleD0y,newpiD0.y[i]/newpiD0.p[i]+angleD0z)
            angletaux,angletauy,angletauz=(angleQx+newtau.z[i]/newtau.p[i],angleQy+newtau.x[i]/newtau.p[i],angleQz+newtau.y[i]/newtau.p[i])
            anglenux,anglenuy,anglenuz=(angleQx+newnuB.z[i]/newnuB.p[i],angleQy+newnuB.x[i]/newnuB.p[i],angleQz+newnuB.y[i]/newnuB.p[i])
            traceQ=go.Scatter3d(x=[angleBx,angleQx],y=[angleBy,angleQy],z=[angleBz,angleQz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['B', 'W'],textposition='top right',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

            traceD0=go.Scatter3d(x=[angleBx,angleD0x],y=[angleBy,angleD0y],z=[angleBz,angleD0z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'D0'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracenu=go.Scatter3d(x=[angleQx,anglenux],y=[angleQy,anglenuy],z=[angleQz,anglenuz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracetau=go.Scatter3d(x=[angleQx,angletaux],y=[angleQy,angletauy],z=[angleQz,angletauz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            traceK=go.Scatter3d(x=[angleD0x,angleKx],y=[angleD0y,angleKy],z=[angleD0z,angleKz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracepi=go.Scatter3d(x=[angleD0x,anglepix],y=[angleD0y,anglepiy],z=[angleD0z,anglepiz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))


            dash1=go.Scatter3d(x=[angleD0x,d1x],y=[angleD0y,d1y],z=[angleD0z,d1z],mode='lines',line = dict(color = ('rgb(20, 20, 255)'),width = 3,dash='dot'))
            mesh1 = go.Mesh3d(x=[d1x,0,angleKx],y=[d1y,0,angleKy],z=[d1z,0,angleKz],
            opacity=0.4,
            color='#3E3A3A')
            dataangle=[traceQ,traceD0,tracenu,tracetau,traceK,tracepi,dash1,mesh1]


        if choice=='W':
            angleQx,angleQy,angleQz=(0,0,0)
            angleBx,angleBy,angleBz=(newB.z[i]/newB.p[i],newB.x[i]/newB.p[i],newB.y[i]/newB.p[i])
            angleD0x,angleD0y,angleD0z=(newD0.z[i]/newD0.p[i]+angleBx,newD0.x[i]/newD0.p[i]+angleBy,newD0.y[i]/newD0.p[i]+angleBz)
            angleKx,angleKy,angleKz=(newK.z[i]/newK.p[i]+angleD0x,newK.x[i]/newK.p[i]+angleD0y,newK.y[i]/newK.p[i]+angleD0z)

            anglepix,anglepiy,anglepiz=(newpiD0.z[i]/newpiD0.p[i]+angleD0x,newpiD0.x[i]/newpiD0.p[i]+angleD0y,newpiD0.y[i]/newpiD0.p[i]+angleD0z)
            angletaux,angletauy,angletauz=(angleQx+newtau.z[i]/newtau.p[i],angleQy+newtau.x[i]/newtau.p[i],angleQz+newtau.y[i]/newtau.p[i])
            anglenux,anglenuy,anglenuz=(angleQx+newnuB.z[i]/newnuB.p[i],angleQy+newnuB.x[i]/newnuB.p[i],angleQz+newnuB.y[i]/newnuB.p[i])
            d1x,d1y,d1z=(-angleBx,-angleBy,-angleBz)
            dash1=go.Scatter3d(x=[angleQx,d1x],y=[angleQy,d1y],z=[angleQz,d1z],mode='lines',line = dict(color = ('rgb(20, 20, 255)'),width = 3,dash='dot'))
            traceQ=go.Scatter3d(x=[angleBx,angleQx],y=[angleBy,angleQy],z=[angleBz,angleQz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['B', 'W'],textposition='top right',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

            traceD0=go.Scatter3d(x=[angleBx,angleD0x],y=[angleBy,angleD0y],z=[angleBz,angleD0z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'D0'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracenu=go.Scatter3d(x=[angleQx,anglenux],y=[angleQy,anglenuy],z=[angleQz,anglenuz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracetau=go.Scatter3d(x=[angleQx,angletaux],y=[angleQy,angletauy],z=[angleQz,angletauz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            traceK=go.Scatter3d(x=[angleD0x,angleKx],y=[angleD0y,angleKy],z=[angleD0z,angleKz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))
            tracepi=go.Scatter3d(x=[angleD0x,anglepix],y=[angleD0y,anglepiy],z=[angleD0z,anglepiz],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))


            mesh1 = go.Mesh3d(x=[d1x,0,angletaux],y=[d1y,0,angletauy],z=[d1z,0,angletauz],
            opacity=0.4,
            color='#3E3A3A')
            dataangle=[traceQ,traceD0,tracenu,tracetau,traceK,tracepi,dash1,mesh1]
       

            layoutangle = go.Layout(
                        showlegend=False,
                        paper_bgcolor = '#F4F4F8',
                        plot_bgcolor = '#F4F4F8',
                        autosize=True,
                        margin=dict(t=10, b=10, l=20, r=10),
                        scene=dict(
                        camera=dict(eye=dict(x=1.5,y=1.5,z=1.5)),

                        xaxis=dict(

                        title='Z',
                        titlefont=dict(
                        family='Arial, sans-serif',
                        size=18,
                        color='black'
                        ),
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=4,
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                        ),
                        yaxis=dict(

                        title='X',
                        titlefont=dict(
                        family='Arial, sans-serif',
                        size=18,
                        color='black'
                        ),
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=4,
                        showbackground=True,
                        backgroundcolor='rgb(230, 230, 230)'
                        ),
                        zaxis=dict(

                        title='Y',
                        titlefont=dict(
                        family='Arial, sans-serif',
                        size=18,
                        color='black'
                        ),
                gridcolor='#bdbdbd',
                gridwidth=2,
                zerolinecolor='#969696',
                zerolinewidth=4,
                linecolor='#636363',
                linewidth=4,

                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
                ),
                aspectratio = dict(x=1, y=1, z=0.7),
                aspectmode = 'manual'
                        )
                )

    return {'data':dataangle,'layout':layoutangle}


#--------------------------------
#                                |
#Callback for the range sliders  |
#                                |
#--------------------------------
@app.callback(
        dash.dependencies.Output('output-range-thetast', 'children'),
        [dash.dependencies.Input('choose-thetast', 'value')])

def update_output1(value):
    return 'You have selected "{}"'.format(value)



@app.callback(
        dash.dependencies.Output('output-range-thetal', 'children'),
        [dash.dependencies.Input('choose-thetal', 'value')])
def update_output2(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
        dash.dependencies.Output('output-range-chi', 'children'),
        [dash.dependencies.Input('choose-chi', 'value')])
def update_output3(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
        dash.dependencies.Output('phase-space', 'figure'),
        [Input('choose-thetast', 'value'),
        Input('choose-thetal', 'value'),
        Input('choose-chi', 'value'),
        Input('dropdown-file','value')])
def plot_phase_space(rangest,rangel,rangechi,filename):

    if filename=='dh_D0':
        df=dh_D0
        particle_list=particles(filename)
        [B,W,D0,tau,nuB,K,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvK,nouvD0]=angles

    elif filename=='dh_Dst':
        df=dh_Dst
        particle_list=particles(filename)
        [B,W,Dst,D0,tau,nuB,K,piDst,piK,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvD0,nouvDst]=angles

    elif filename=='dh_2420':
        df=dh_2420
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles

    elif filename=='dh_2460':
        df=dh_2460
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles 
        

    layout1=go.Layout(
        hovermode = 'closest',
        paper_bgcolor = '#F4F4F8',
        plot_bgcolor = '#F4F4F8',
        margin=dict(t=50, b=50, l=50, r=40),
        width=450,
        height=600,
        clickmode='event+select',
        scene=dict(camera=dict(eye=dict(x=2,y=2,z=2)),
                   xaxis=dict(title='costhetast',range=rangest),
                   yaxis=dict(title='costhetal',range=rangel),
                   zaxis=dict(title='chi',range=rangechi)
                                ))

    trace_phase=go.Scatter3d(
                x=costhetast,
                y=costhetal,
                z=chi,
                mode='markers',
                marker=dict(
                    size=5,
                    color=chi,
                    colorscale='Electric',
                    opacity=0.8,
                    colorbar=dict(thickness=20)
                )
    )
    return {'data': [trace_phase], 'layout':layout1}
#---------------------------
#                           |
#Callback for dropdown dist |
#                           |
#---------------------------

@app.callback(
Output('histog', 'figure'),
[Input('dropdown-dist','value'),
Input('dropdown-file','value')])

def drophist(choice,filename):
    if filename=='dh_D0':
        df=dh_D0
        particle_list=particles(filename)
        [B,W,D0,tau,nuB,K,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvK,nouvD0]=angles

    elif filename=='dh_Dst':
        df=dh_Dst
        particle_list=particles(filename)
        [B,W,Dst,D0,tau,nuB,K,piDst,piK,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvD0,nouvDst]=angles

    elif filename=='dh_2420':
        df=dh_2420
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles

    elif filename=='dh_2460':
        df=dh_2460
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles 
        
    if choice=='chid':
        return {'data':[go.Histogram(x=chi)],'layout':{'title':'chi','paper_bgcolor' : '#F4F4F8', 'plot_bgcolor' : '#F4F4F8'}}
    if choice=='std':
        return {'data':[go.Histogram(x=costhetast)],'layout':{'title':'costheta*','paper_bgcolor' : '#F4F4F8', 'plot_bgcolor' : '#F4F4F8'}}
    if choice=='ld':
        return {'data':[go.Histogram(x=costhetal)],'layout':{'title':'costhetal','paper_bgcolor' : '#F4F4F8', 'plot_bgcolor' : '#F4F4F8'}}
    if choice=='q2':
        q=W
        q2=q.mag2
        return {'data':[go.Histogram(x=q2)],'layout':{'title':'q2','paper_bgcolor' : '#F4F4F8', 'plot_bgcolor' : '#F4F4F8'}}

@app.callback(
    Output('click-result', 'children'),
    [Input('phase-space', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@app.callback(
Output('frame-graph', 'figure'),
[Input('phase-space', 'clickData'),
Input('which-D','value'),
Input('dropdown-frame','value'),
Input('dropdown-file','value')])
def drawevent(selection,radio,frame,filename):

    if filename=='dh_D0':
        df=dh_D0
        particle_list=particles(filename)
        [B,W,D0,tau,nuB,K,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvK,nouvD0]=angles

    elif filename=='dh_Dst':
        df=dh_Dst
        particle_list=particles(filename)
        [B,W,Dst,D0,tau,nuB,K,piDst,piK,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvD0,nouvDst]=angles

    elif filename=='dh_2420':
        df=dh_2420
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles

    elif filename=='dh_2460':
        df=dh_2460
        particle_list=particles(filename)
        [B,W,Dst,Dstst,D0,tau,nuB,K,piDst,piDstst,piD0,pitau1,pitau2,pitau3,nutau]=particle_list
        angles=calc_angles(filename)
        [costhetast,costhetal,chi,nouvpi,nouvDst,nouvDstst]=angles 
        

    if selection is None:
        return {}
    else:
        i=selection['points'][0]['pointNumber']

        dis=(df['B_FD_TRUE'][i]+df['D0_FD_TRUE'][i]+df['Tau_FD_TRUE'][i])/3                #the average flight distance
        cmin = 0
        cmax = B.p[i]

        def colorval(vec):              #determine the color
                val=vec.p[i]
                if val==cmax:
                    return 'rgb(0,0,0)'
                if val<cmax and val>cmax/4:
                    return 'rgb(0,0,83)'
                if val<=cmax/4 and val>cmax/16 :
                    return 'rgb(0,0,128)'
                if val<=cmax/16 and val>cmax/64:
                    return 'rgb(0,0,196)'
                else :
                    return 'rgb(0,0,255)'


                if frame=='lab':

                    coord=coordinates(filename)

                    if filename=='dh_D0':
                        [PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piD0_X,piD0_Y,piD0_Z,pitau1_X,pitau1_Y,pitau1_Z,
                        pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]=coord
                    if filename=='dh_Dst':
                        [PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,Dst_X,Dst_Y,Dst_Z,
                        D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piDst_X,piDst_Y,piDst_Z,
                        piK_X,piK_Y,piK_Z,pitau1_X,pitau1_Y,pitau1_Z,
                        pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]=coord

                    if filename=='dh_2420' :
                        [PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,Dst_X,Dst_Y,Dst_Z,Dstst_X,Dstst_Y,Dstst_Z,
                        D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piDst_X,piDst_Y,piDst_Z,
                        piDstst_X,piDstst_Y,piDstst_Z,
                        piK_X,piK_Y,piK_Z,pitau1_X,pitau1_Y,pitau1_Z,
                        pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]=coord
                    if filename=='dh_2460':
                        [PV_X,PV_Y,PV_Z,B_X,B_Y,B_Z,Dst_X,Dst_Y,Dst_Z,Dstst_X,Dstst_Y,Dstst_Z,
                        D0_X,D0_Y,D0_Z,tau_X,tau_Y,tau_Z,nu_X,nu_Y,nu_Z,K_X,K_Y,K_Z,piDst_X,piDst_Y,piDst_Z,
                        piDstst_X,piDstst_Y,piDstst_Z,
                        piK_X,piK_Y,piK_Z,pitau1_X,pitau1_Y,pitau1_Z,
                        pitau2_X,pitau2_Y,pitau2_Z,pitau3_X,pitau3_Y,pitau3_Z,nutau_X,nutau_Y,nutau_Z]=coord
                        
                        
                else:
                    COM=com(filename,frame)
                    liste_part=change_frame(COM,filename)
                    if filename=='dh_D0':
                        [newB,newW,newD0,newtau,newnuB,newK,newpiD0,newpitau1,newpitau2,newpitau3,newnutau]=liste_part

                    elif filename=='dh_Dst':
                        [newB,newW,newDst,newD0,newtau,newnuB,newK,newpiDst,newpiK,newpitau1,newpitau2,newpitau3,newnutau]=liste_part

                    elif filename=='dh_2420' : 
                        [newB,newW,newDst,newDstst,newD0,newtau,newnuB,newK,newpiDst,newpiDstst,newpiD0,newpitau1,newpitau2,newpitau3,newnutau]=liste_part
                    elif filename=='dh_2460':
                        [newB,newW,newDst,newDstst,newD0,newtau,newnuB,newK,newpiDst,newpiDstst,newpiD0,newpitau1,newpitau2,newpitau3,newnutau]=liste_part
                   
                    PV_X,PV_Y,PV_Z=(0,0,0)
                    if frame=='B':
                        B_X,B_Y,B_Z=(0,0,0)
                    else:
                        B_X,B_Y,B_Z=[newB.z[i]*df['B_FD_TRUE'][i]/newB.p[i],newB.x[i]*df['B_FD_TRUE'][i]/newB.p[i],newB.y[i]*df['B_FD_TRUE'][i]/newB.p[i]]


                    tau_X,tau_Y,tau_Z=[newtau.z[i]*df['Tau_FD_TRUE'][i]/newtau.p[i]+B_X,newtau.x[i]*df['Tau_FD_TRUE'][i]/newtau.p[i]+B_Y,newtau.y[i]*df['Tau_FD_TRUE'][i]/newtau.p[i]+B_Z]
                    nu_X=newnuB.z[i]*dis/newnuB.p[i]+B_X
                    nu_Y=newnuB.x[i]*dis/newnuB.p[i]+B_Y
                    nu_Z=newnuB.y[i]*dis/newnuB.p[i]+B_Z
                    pitau1_X=newpitau1.z[i]*dis/newpitau1.p[i]+tau_X
                    pitau1_Y=newpitau1.x[i]*dis/newpitau1.p[i]+tau_Y
                    pitau1_Z=newpitau1.y[i]*dis/newpitau1.p[i]+tau_Z

                    pitau2_X=newpitau2.z[i]*dis/newpitau2.p[i]+tau_X
                    pitau2_Y=newpitau2.x[i]*dis/newpitau2.p[i]+tau_Y
                    pitau2_Z=newpitau2.y[i]*dis/newpitau2.p[i]+tau_Z

                    pitau3_X=newpitau3.z[i]*dis/newpitau3.p[i]+tau_X
                    pitau3_Y=newpitau3.x[i]*dis/newpitau3.p[i]+tau_Y
                    pitau3_Z=newpitau3.y[i]*dis/newpitau3.p[i]+tau_Z

                    nutau_X=newnutau.z[i]*dis/newnutau.p[i]+tau_X
                    nutau_Y=newnutau.x[i]*dis/newnutau.p[i]+tau_Y
                    nutau_Z=newnutau.y[i]*dis/newnutau.p[i]+tau_Z
                    K_X=newK.z[i]*dis/newK.p[i]+D0_X
                    K_Y=newK.x[i]*dis/newK.p[i]+D0_Y
                    K_Z=newK.y[i]*dis/newK.p[i]+D0_Z            
                    piD0_X=newpiD0.z[i]*dis/newpiD0.p[i]+D0_X
                    piD0_Y=newpiD0.x[i]*dis/newpiD0.p[i]+D0_Y
                    piD0_Z=newpiD0.y[i]*dis/newpiD0.p[i]+D0_Z

                    if filename=='dh_D0':                    
                        D0_X,D0_Y,D0_Z=[newD0.z[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+B_X,newD0.x[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+B_Y,newD0.y[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+B_Z]

                    if filename=='dh_Dst':                    
                        Dst_X,Dst_Y,Dst_Z=[newDst.z[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+B_X,newDst.x[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+B_Y,newDst.y[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+B_Z]
                        piDst_X=newpiDst.z[i]*dis/newpiDst.p[i]+Dst_X
                        piDst_Y=newpiDst.x[i]*dis/newpiDst.p[i]+Dst_Y
                        piDst_Z=newpiDst.y[i]*dis/newpiDst.p[i]+Dst_Z
                    
                    if filename=='dh_2460' :
                        Dstst_X,Dstst_Y,Dstst_Z=[newDstst.z[i]*df['Dstst_FD_TRUE'][i]/newDstst.p[i]+B_X,newDstst.x[i]*df['Dstst_FD_TRUE'][i]/newDstst.p[i]+B_Y,newDstst.y[i]*df['Dstst_FD_TRUE'][i]/newDstst.p[i]+B_Z]
                        Dst_X,Dst_Y,Dst_Z=[newDst.z[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+Dstst_X,newDst.x[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+Dstst_Y,newDst.y[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+Dstst_Z]
                        piDst_X=newpiDst.z[i]*dis/newpiDst.p[i]+Dst_X
                        piDst_Y=newpiDst.x[i]*dis/newpiDst.p[i]+Dst_Y
                        piDst_Z=newpiDst.y[i]*dis/newpiDst.p[i]+Dst_Z
                        D0_X,D0_Y,D0_Z=[newD0.z[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+Dst_X,newD0.x[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+Dst_Y,newD0.y[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+Dst_Z]
                                               
                        piDstst_X=newpiDstst.z[i]*dis/newpiDstst.p[i]+Dstst_X
                        piDstst_Y=newpiDstst.x[i]*dis/newpiDstst.p[i]+Dstst_Y
                        piDstst_Z=newpiDstst.y[i]*dis/newpiDstst.p[i]+Dstst_Z
                    if filename=='dh_2420':
                        Dstst_X,Dstst_Y,Dstst_Z=[newDstst.z[i]*df['Dstst_FD_TRUE'][i]/newDstst.p[i]+B_X,newDstst.x[i]*df['Dstst_FD_TRUE'][i]/newDstst.p[i]+B_Y,newDstst.y[i]*df['Dstst_FD_TRUE'][i]/newDstst.p[i]+B_Z]
                        Dst_X,Dst_Y,Dst_Z=[newDst.z[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+Dstst_X,newDst.x[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+Dstst_Y,newDst.y[i]*df['Dst_FD_TRUE'][i]/newDst.p[i]+Dstst_Z]
                        piDst_X=newpiDst.z[i]*dis/newpiDst.p[i]+Dst_X
                        piDst_Y=newpiDst.x[i]*dis/newpiDst.p[i]+Dst_Y
                        piDst_Z=newpiDst.y[i]*dis/newpiDst.p[i]+Dst_Z
                        D0_X,D0_Y,D0_Z=[newD0.z[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+Dst_X,newD0.x[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+Dst_Y,newD0.y[i]*df['D0_FD_TRUE'][i]/newD0.p[i]+Dst_Z]
                                               
                        piDstst_X=newpiDstst.z[i]*dis/newpiDstst.p[i]+Dstst_X
                        piDstst_Y=newpiDstst.x[i]*dis/newpiDstst.p[i]+Dstst_Y
                        piDstst_Z=newpiDstst.y[i]*dis/newpiDstst.p[i]+Dstst_Z
                    
                xrange=calc_xrange(filename)
                yrange=calc_yrange(filename)
                zrange=calc_zrange(filename)
                
                if radio=='3D':
                    
                    if filename=='dh_D0':
                        traceB=go.Scatter3d(x=[PV_X,B_X],y=[PV_Y,B_Y],z=[PV_Z,B_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['PV', ''],textposition='top left',line = dict(color =colorval(B),width=3))
                        tracetau=go.Scatter3d(x=[B_X,tau_X],y=[B_Y,tau_Y],z=[B_Z,tau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = colorval(tau),width=3))
                        tracenuB=go.Scatter3d(x=[B_X,nu_X],y=[B_Y,nu_Y],z=[B_Z,nu_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)",opacity=0.8),text=['', 'nuB'],textposition='top left',line = dict(color = colorval(nuB),width=3))
                        tracepitau1=go.Scatter3d(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],z=[tau_Z,pitau1_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau1),width = 3))
                        tracepitau2=go.Scatter3d(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],z=[tau_Z,pitau2_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau2),width = 3))
                        tracepitau3=go.Scatter3d(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],z=[tau_Z,pitau3_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau3),width = 3))
                        tracenutau=go.Scatter3d(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],z=[tau_Z,nutau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = colorval(nutau),width = 3))
                        traceK=go.Scatter3d(x=[D0_X,K_X],y=[D0_Y,K_Y],z=[D0_Z,K_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color =colorval(K),width=3))
                        tracepiD0=go.Scatter3d(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],z=[D0_Z,piD0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color=colorval(piD0),width=3))

                        traceD0=go.Scatter3d(x=[B_X,D0_X],y=[B_Y,D0_Y],z=[B_Z,D0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D0'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                        
                    if filename=='dh_Dst':
                        traceB=go.Scatter3d(x=[PV_X,B_X],y=[PV_Y,B_Y],z=[PV_Z,B_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['PV', ''],textposition='top left',line = dict(color =colorval(B),width=3))
                        tracetau=go.Scatter3d(x=[B_X,tau_X],y=[B_Y,tau_Y],z=[B_Z,tau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = colorval(tau),width=3))
                        tracenuB=go.Scatter3d(x=[B_X,nu_X],y=[B_Y,nu_Y],z=[B_Z,nu_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)",opacity=0.8),text=['', 'nuB'],textposition='top left',line = dict(color = colorval(nuB),width=3))
                        tracepitau1=go.Scatter3d(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],z=[tau_Z,pitau1_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau1),width = 3))
                        tracepitau2=go.Scatter3d(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],z=[tau_Z,pitau2_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau2),width = 3))
                        tracepitau3=go.Scatter3d(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],z=[tau_Z,pitau3_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau3),width = 3))
                        tracenutau=go.Scatter3d(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],z=[tau_Z,nutau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = colorval(nutau),width = 3))
                        traceK=go.Scatter3d(x=[D0_X,K_X],y=[D0_Y,K_Y],z=[D0_Z,K_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color =colorval(K),width=3))
                        tracepiD0=go.Scatter3d(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],z=[D0_Z,piD0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color=colorval(piD0),width=3))

                        tracepiK=go.Scatter3d(x=[D0_X,piK_X],y=[D0_Y,piK_Y],z=[D0_Z,piK_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color=colorval(piK),width = 3))
                        tracepiDst=go.Scatter3d(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],z=[Dst_Z,piDst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(piDst),width = 3))
                        traceDst=go.Scatter3d(x=[B_X,Dst_X],y=[B_Y,Dst_Y],z=[B_Z,Dst_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D*'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                        traceD0=go.Scatter3d(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],z=[Dst_Z,D0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D0'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                    if filename=='dh_2420' :
                        traceB=go.Scatter3d(x=[PV_X,B_X],y=[PV_Y,B_Y],z=[PV_Z,B_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['PV', ''],textposition='top left',line = dict(color =colorval(B),width=3))
                        tracetau=go.Scatter3d(x=[B_X,tau_X],y=[B_Y,tau_Y],z=[B_Z,tau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = colorval(tau),width=3))
                        tracenuB=go.Scatter3d(x=[B_X,nu_X],y=[B_Y,nu_Y],z=[B_Z,nu_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)",opacity=0.8),text=['', 'nuB'],textposition='top left',line = dict(color = colorval(nuB),width=3))
                        tracepitau1=go.Scatter3d(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],z=[tau_Z,pitau1_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau1),width = 3))
                        tracepitau2=go.Scatter3d(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],z=[tau_Z,pitau2_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau2),width = 3))
                        tracepitau3=go.Scatter3d(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],z=[tau_Z,pitau3_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau3),width = 3))
                        tracenutau=go.Scatter3d(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],z=[tau_Z,nutau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = colorval(nutau),width = 3))
                        traceK=go.Scatter3d(x=[D0_X,K_X],y=[D0_Y,K_Y],z=[D0_Z,K_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color =colorval(K),width=3))
                        tracepiD0=go.Scatter3d(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],z=[D0_Z,piD0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color=colorval(piD0),width=3))
                        
                        tracepiDstst=go.Scatter3d(x=[Dstst_X,piDstst_X],y=[Dstst_Y,piDstst_Y],z=[Dstst_Z,piDstst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(piDst),width = 3))
                        traceDstst=go.Scatter3d(x=[B_X,Dstst_X],y=[B_Y,Dstst_Y],z=[B_Z,Dstst_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D**'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                        tracepiDst=go.Scatter3d(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],z=[Dst_Z,piDst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(piDst),width = 3))
                        traceDst=go.Scatter3d(x=[Dstst_X,Dst_X],y=[Dstst_Y,Dst_Y],z=[Dstst_Z,Dst_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D*'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                        traceD0=go.Scatter3d(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],z=[Dst_Z,D0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D0'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                    if filename=='dh_2460':
                        traceB=go.Scatter3d(x=[PV_X,B_X],y=[PV_Y,B_Y],z=[PV_Z,B_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['PV', ''],textposition='top left',line = dict(color =colorval(B),width=3))
                        tracetau=go.Scatter3d(x=[B_X,tau_X],y=[B_Y,tau_Y],z=[B_Z,tau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = colorval(tau),width=3))
                        tracenuB=go.Scatter3d(x=[B_X,nu_X],y=[B_Y,nu_Y],z=[B_Z,nu_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)",opacity=0.8),text=['', 'nuB'],textposition='top left',line = dict(color = colorval(nuB),width=3))
                        tracepitau1=go.Scatter3d(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],z=[tau_Z,pitau1_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau1),width = 3))
                        tracepitau2=go.Scatter3d(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],z=[tau_Z,pitau2_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau2),width = 3))
                        tracepitau3=go.Scatter3d(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],z=[tau_Z,pitau3_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(pitau3),width = 3))
                        tracenutau=go.Scatter3d(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],z=[tau_Z,nutau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = colorval(nutau),width = 3))
                        traceK=go.Scatter3d(x=[D0_X,K_X],y=[D0_Y,K_Y],z=[D0_Z,K_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color =colorval(K),width=3))
                        tracepiD0=go.Scatter3d(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],z=[D0_Z,piD0_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color=colorval(piD0),width=3))

                        tracepiDstst=go.Scatter3d(x=[Dstst_X,piDstst_X],y=[Dstst_Y,piDstst_Y],z=[Dstst_Z,piDstst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(piDst),width = 3))
                        traceDstst=go.Scatter3d(x=[B_X,Dstst_X],y=[B_Y,Dstst_Y],z=[B_Z,Dstst_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D**'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                        tracepiDst=go.Scatter3d(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],z=[Dst_Z,piDst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = colorval(piDst),width = 3))
                        traceDst=go.Scatter3d(x=[Dstst_X,Dst_X],y=[Dstst_Y,Dst_Y],z=[Dstst_Z,Dst_Z],mode='lines+markers+text',marker=dict(size=5,color="rgb(5,200,5)",opacity=0.8,cmin=cmin,cmax=cmax),text=['', 'D*'],textposition='top left',line = dict(width=3,color=colorval(D0)))
                    layout_event = go.Layout(
                            showlegend=False,
                            #width=400,
                            #height=400,
                            paper_bgcolor = '#F4F4F8',
                            plot_bgcolor = '#F4F4F8',
                            autosize=True,
                            margin=dict(t=10, b=10, l=20, r=10),
                            scene=dict(
                            camera=dict(eye=dict(x=1.5,y=1.5,z=1.5)),
                            xaxis=dict(
                            range=[-abs(xrange), abs(xrange)],
                            title='Z direction [nm]',
                            titlefont=dict(
                            family='Arial, sans-serif',
                            size=18,
                            color='black'
                            ),
                            gridcolor='#bdbdbd',
                            gridwidth=2,
                            zerolinecolor='#969696',
                            zerolinewidth=4,
                            linecolor='#636363',
                            linewidth=4,
                            showbackground=True,
                            backgroundcolor='rgb(230, 230,230)'
                            ),
                            yaxis=dict(
                            range=[-abs(yrange), abs(yrange)],
                            title='X direction [nm]',
                            titlefont=dict(
                            family='Arial, sans-serif',
                            size=18,
                            color='black'
                            ),
                            gridcolor='#bdbdbd',
                            gridwidth=2,
                            zerolinecolor='#969696',
                            zerolinewidth=4,
                            linecolor='#636363',
                            linewidth=4,
                            showbackground=True,
                            backgroundcolor='rgb(230, 230, 230)'
                            ),
                            zaxis=dict(
                            range=[-abs(zrange), abs(zrange)],
                            title='Y direction [nm]',
                            titlefont=dict(
                            family='Arial, sans-serif',
                            size=18,
                            color='black'
                            ),
                    gridcolor='#bdbdbd',
                    gridwidth=2,
                    zerolinecolor='#969696',
                    zerolinewidth=4,
                    linecolor='#636363',
                    linewidth=4,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                    ),
                    aspectratio = dict(x=1, y=1, z=0.7),
                    aspectmode = 'manual'
                            )
                    )



                elif radio=='ZX':
                    if filename=='dh_D0':
                        traceB=go.Scatter(x=[PV_X,B_X],y=[PV_Y,B_Y],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[B_X,D0_X],y=[B_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_X,tau_X],y=[B_Y,tau_Y],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_X,nu_X],y=[B_Y,nu_Y],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_X,K_X],y=[D0_Y,K_Y],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracepiD0=go.Scatter(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenutau=go.Scatter(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        traceD0=go.Scatter(x=[B_X,D0_X],y=[B_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                    if filename=='dh_Dst':
                        traceB=go.Scatter(x=[PV_X,B_X],y=[PV_Y,B_Y],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[B_X,D0_X],y=[B_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_X,tau_X],y=[B_Y,tau_Y],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_X,nu_X],y=[B_Y,nu_Y],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_X,K_X],y=[D0_Y,K_Y],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracepiD0=go.Scatter(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenutau=go.Scatter(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        tracepiK=go.Scatter(x=[D0_X,piK_X],y=[D0_Y,piK_Y],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[B_X,Dst_X],y=[B_Y,Dst_Y],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                    if filename=='dh_2420' :
                        traceB=go.Scatter(x=[PV_X,B_X],y=[PV_Y,B_Y],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[B_X,D0_X],y=[B_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_X,tau_X],y=[B_Y,tau_Y],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_X,nu_X],y=[B_Y,nu_Y],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_X,K_X],y=[D0_Y,K_Y],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracepiD0=go.Scatter(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenutau=go.Scatter(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        tracepiDstst=go.Scatter(x=[Dstst_X,piDstst_X],y=[Dstst_Y,piDstst_Y],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDstst=go.Scatter(x=[B_X,Dstst_X],y=[B_Y,Dstst_Y],mode='lines+markers+text',text=['','D**'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[Dstst_X,Dst_X],y=[Dstst_Y,Dst_Y],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                    if filename=='dh_2460':
                        traceB=go.Scatter(x=[PV_X,B_X],y=[PV_Y,B_Y],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[B_X,D0_X],y=[B_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_X,tau_X],y=[B_Y,tau_Y],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_X,nu_X],y=[B_Y,nu_Y],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_X,K_X],y=[D0_Y,K_Y],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracepiD0=go.Scatter(x=[D0_X,piD0_X],y=[D0_Y,piD0_Y],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenutau=go.Scatter(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        tracepiDstst=go.Scatter(x=[Dstst_X,piDstst_X],y=[Dstst_Y,piDstst_Y],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDstst=go.Scatter(x=[B_X,Dstst_X],y=[B_Y,Dstst_Y],mode='lines+markers+text',text=['','D**'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[Dstst_X,Dst_X],y=[Dstst_Y,Dst_Y],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                
                    layout_event = go.Layout(
                    showlegend=False,
                    paper_bgcolor = '#F4F4F8',
                    plot_bgcolor = '#F4F4F8',
                    xaxis=dict(
                        title='Z direction [nm]',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        mirror='ticks',
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=6
                    ),
                    yaxis=dict(
                        title='X direction [nm]',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        mirror='ticks',
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=6
                        )
                    )


                elif radio=='XY':

                    if filename=='dh_D0':
                        traceB=go.Scatter(x=[PV_Y,B_Y],y=[PV_Z,B_Z],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Y,tau_Y],y=[B_Z,tau_Z],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Y,nu_Y],y=[B_Z,nu_Z],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Y,pitau1_Y],y=[tau_Z,pitau1_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Y,pitau2_Y],y=[tau_Z,pitau2_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Y,pitau3_Y],y=[tau_Z,pitau3_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Y,K_Y],y=[D0_Z,K_Z],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Y,piD0_Y],y=[D0_Z,piD0_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Y,nutau_Y],y=[tau_Z,nutau_Z],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        traceD0=go.Scatter(x=[B_X,D0_X],y=[B_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        
                    if filename=='dh_Dst':
                        traceB=go.Scatter(x=[PV_Y,B_Y],y=[PV_Z,B_Z],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Y,tau_Y],y=[B_Z,tau_Z],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Y,nu_Y],y=[B_Z,nu_Z],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Y,pitau1_Y],y=[tau_Z,pitau1_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Y,pitau2_Y],y=[tau_Z,pitau2_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Y,pitau3_Y],y=[tau_Z,pitau3_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Y,K_Y],y=[D0_Z,K_Z],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Y,piD0_Y],y=[D0_Z,piD0_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Y,nutau_Y],y=[tau_Z,nutau_Z],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        tracepiK=go.Scatter(x=[D0_Y,piK_Y],y=[D0_Z,piK_Z],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_Y,piDst_Y],y=[Dst_Z,piDst_Z],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[B_Y,Dst_Y],y=[B_Z,Dst_Z],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
   
                    if filename=='dh_2420' :
                        traceB=go.Scatter(x=[PV_Y,B_Y],y=[PV_Z,B_Z],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Y,tau_Y],y=[B_Z,tau_Z],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Y,nu_Y],y=[B_Z,nu_Z],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Y,pitau1_Y],y=[tau_Z,pitau1_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Y,pitau2_Y],y=[tau_Z,pitau2_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Y,pitau3_Y],y=[tau_Z,pitau3_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Y,K_Y],y=[D0_Z,K_Z],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Y,piD0_Y],y=[D0_Z,piD0_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Y,nutau_Y],y=[tau_Z,nutau_Z],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        tracepiDstst=go.Scatter(x=[Dstst_Y,piDstst_Y],y=[Dstst_Z,piDstst_Z],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDstst=go.Scatter(x=[B_Y,Dstst_Y],y=[B_Z,Dstst_Z],mode='lines+markers+text',text=['','D**'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_Y,piDst_Y],y=[Dst_Z,piDst_Z],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[Dstst_Y,Dst_Y],y=[Dstst_Z,Dst_Z],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                    
                    if filename=='dh_2460':
                        traceB=go.Scatter(x=[PV_Y,B_Y],y=[PV_Z,B_Z],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Y,tau_Y],y=[B_Z,tau_Z],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Y,nu_Y],y=[B_Z,nu_Z],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Y,pitau1_Y],y=[tau_Z,pitau1_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Y,pitau2_Y],y=[tau_Z,pitau2_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Y,pitau3_Y],y=[tau_Z,pitau3_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Y,K_Y],y=[D0_Z,K_Z],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Y,piD0_Y],y=[D0_Z,piD0_Z],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Y,nutau_Y],y=[tau_Z,nutau_Z],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDstst=go.Scatter(x=[Dstst_Y,piDstst_Y],y=[Dstst_Z,piDstst_Z],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDstst=go.Scatter(x=[B_Y,Dstst_Y],y=[B_Z,Dstst_Z],mode='lines+markers+text',text=['','D**'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_Y,piDst_Y],y=[Dst_Z,piDst_Z],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceD0=go.Scatter(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[Dstst_Y,Dst_Y],y=[Dstst_Z,Dst_Z],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                    
                    layout_event = go.Layout(
                        showlegend=False,
                        paper_bgcolor = '#F4F4F8',
                        plot_bgcolor = '#F4F4F8',
                        xaxis=dict(
                        title='X direction [nm]',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        mirror='ticks',
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=6
                        ),
                        yaxis=dict(
                        title='Y direction [nm]',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        mirror='ticks',
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=6
                        )
                )

                elif radio=='YZ':
                                        
                    if filename=='dh_D0':
                        traceB=go.Scatter(x=[PV_Z,B_Z],y=[PV_X,B_X],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Z,tau_Z],y=[B_X,tau_X],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Z,nu_Z],y=[B_X,nu_X],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Z,pitau1_Z],y=[tau_X,pitau1_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Z,pitau2_Z],y=[tau_X,pitau2_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Z,pitau3_Z],y=[tau_X,pitau3_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Z,K_Z],y=[D0_X,K_X],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Z,piD0_Z],y=[D0_X,piD0_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Z,nutau_Z],y=[tau_X,nutau_X],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        traceD0=go.Scatter(x=[B_Z,D0_Z],y=[B_X,D0_X],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        
                    if filename=='dh_Dst':
                        traceB=go.Scatter(x=[PV_Z,B_Z],y=[PV_X,B_X],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Z,tau_Z],y=[B_X,tau_X],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Z,nu_Z],y=[B_X,nu_X],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Z,pitau1_Z],y=[tau_X,pitau1_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Z,pitau2_Z],y=[tau_X,pitau2_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Z,pitau3_Z],y=[tau_X,pitau3_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Z,K_Z],y=[D0_X,K_X],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Z,piD0_Z],y=[D0_X,piD0_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Z,nutau_Z],y=[tau_X,nutau_X],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        traceD0=go.Scatter(x=[Dst_Z,D0_Z],y=[Dst_X,D0_X],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracepiK=go.Scatter(x=[D0_Z,piK_Z],y=[D0_X,piK_X],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_Z,piDst_Z],y=[Dst_X,piDst_X],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[B_Z,Dst_Z],y=[B_X,Dst_X],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                        
                    if filename=='dh_2420' : 
                        traceB=go.Scatter(x=[PV_Z,B_Z],y=[PV_X,B_X],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Z,tau_Z],y=[B_X,tau_X],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Z,nu_Z],y=[B_X,nu_X],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Z,pitau1_Z],y=[tau_X,pitau1_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Z,pitau2_Z],y=[tau_X,pitau2_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Z,pitau3_Z],y=[tau_X,pitau3_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Z,K_Z],y=[D0_X,K_X],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Z,piD0_Z],y=[D0_X,piD0_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Z,nutau_Z],y=[tau_X,nutau_X],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        traceD0=go.Scatter(x=[Dst_Z,D0_Z],y=[Dst_X,D0_X],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[Dstst_Z,Dst_Z],y=[Dstst_X,Dst_X],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDstst=go.Scatter(x=[Dstst_Z,piDstst_Z],y=[Dstst_X,piDstst_X],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDstst=go.Scatter(x=[B_Z,Dstst_Z],y=[B_X,Dstst_X],mode='lines+markers+text',text=['','D**'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_Z,piDst_Z],y=[Dst_X,piDst_X],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                    if filename=='dh_2460':
                        traceB=go.Scatter(x=[PV_Z,B_Z],y=[PV_X,B_X],mode='lines+markers+text',text=['PV','B'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracetau=go.Scatter(x=[B_Z,tau_Z],y=[B_X,tau_X],mode='lines+markers+text',text=['','tau'],textposition='top center',line=dict(color='darkblue',width=2))

                        tracenuB=go.Scatter(x=[B_Z,nu_Z],y=[B_X,nu_X],mode='lines+markers+text',text=['','nu'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau1=go.Scatter(x=[tau_Z,pitau1_Z],y=[tau_X,pitau1_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau2=go.Scatter(x=[tau_Z,pitau2_Z],y=[tau_X,pitau2_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepitau3=go.Scatter(x=[tau_Z,pitau3_Z],y=[tau_X,pitau3_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))

                        traceK=go.Scatter(x=[D0_Z,K_Z],y=[D0_X,K_X],mode='lines+markers+text',text=['','K'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiD0=go.Scatter(x=[D0_Z,piD0_Z],y=[D0_X,piD0_X],mode='lines+markers+text',text=['', 'pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracenutau=go.Scatter(x=[tau_Z,nutau_Z],y=[tau_X,nutau_X],mode='lines+markers+text',text=['', 'nu'],textposition='top center',line=dict(color='darkblue',width=2))


                        traceD0=go.Scatter(x=[Dst_Z,D0_Z],y=[Dst_X,D0_X],mode='lines+markers+text',text=['','D0'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDst=go.Scatter(x=[Dstst_Z,Dst_Z],y=[Dstst_X,Dst_X],mode='lines+markers+text',text=['','D*'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDstst=go.Scatter(x=[Dstst_Z,piDstst_Z],y=[Dstst_X,piDstst_X],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                        traceDstst=go.Scatter(x=[B_Z,Dstst_Z],y=[B_X,Dstst_X],mode='lines+markers+text',text=['','D**'],textposition='top center',line=dict(color='darkblue',width=2))
                        tracepiDst=go.Scatter(x=[Dst_Z,piDst_Z],y=[Dst_X,piDst_X],mode='lines+markers+text',text=['','pi'],textposition='top center',line=dict(color='darkblue',width=2))
                   
                        
                    layout_event = go.Layout(
                        showlegend=False,
                        paper_bgcolor = '#F4F4F8',
                        plot_bgcolor = '#F4F4F8',
                        xaxis=dict(
                        title='Y direction [nm]',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        mirror='ticks',
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=6
                        ),
                        yaxis=dict(
                        title='Z direction [nm]',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        mirror='ticks',
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=6
                        )
                        )

    if filename=='dh_D0':
        data_event=[traceB,tracetau,traceD0,tracenuB,traceK,tracepiD0,tracepitau1,tracepitau2,tracepitau3,tracenutau]
    if filename=='dh_Dst':
        data_event=[traceB,tracetau,traceD0,tracenuB,traceK,tracepiD0,tracepitau1,tracepitau2,tracepitau3,tracenutau,traceDst,tracepiDst,tracepiK]
    if filename=='dh_2420':          
        data_event=[traceDstst,tracepiDstst,traceB,tracetau,traceD0,tracenuB,traceK,tracepiD0,tracepitau1,tracepitau2,tracepitau3,tracenutau,traceDst,tracepiDst,tracepiK]
    if filename=='dh_2460':
        data_event=[traceDstst,tracepiDstst,traceB,tracetau,traceD0,tracenuB,traceK,tracepiD0,tracepitau1,tracepitau2,tracepitau3,tracenutau,traceDst,tracepiDst,tracepiK]
    
    return {'data': data_event, 'layout':layout_event}












if __name__ == '__main__':
    app.run_server(debug=True)






                                                                                                                                                                                                                                                                     
                                                                                                                                    
                                                                                                                                    

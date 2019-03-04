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


# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 05:44:33 2021

@author: Dnis
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:16:42 2021

@author: Dnis
"""


import csv
from matplotlib.widgets import Cursor
import random
from heapq import nsmallest
from sklearn.neighbors import KernelDensity
import warnings
import scipy.stats as stats
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import pandas as pd
import seaborn as sb
from matplotlib import pyplot
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde
from tkinter.messagebox import *
from PIL import ImageTk, Image
from collections import Counter

root = tk.Tk()
global selector
selector = tk.StringVar()

root.title('Violin Analysis')
root.geometry('550x500')
root.resizable(False, False)

frame = tk.Frame(root)
options = {'padx': 5, 'pady': 5}


# =============================================================================


def SelectFichero():
    global name
    name = fd.askopenfilename(title="Select a file to analysis", filetypes=(
        ("csv files", "*.csv"), ("All files", "*.*")))
    ruta1_label = ttk.Label(text=name)
    ruta1_label.place(x=40, y=50)

    print(name)
    return name


def SelectFicheroLimites():
    global nameLimits
    nameLimits = fd.askopenfilename(title="Select Alert - Alarm  file ", filetypes=(
        ("excel files", "*.xlsx"), ("excel files", "*.xls"), ("All files", "*.*")))
    ruta1_labelLimits = ttk.Label(text=nameLimits)
    ruta1_labelLimits.place(x=40, y=270)

    print(nameLimits)
    seleccionar_descripores_malos()
    return nameLimits

    # metodos para seleccionar ficheros


def seleccionar_parametros():

    global datos
    global descVibra
    datos = pd.read_csv(name)

    ruta1_label = ttk.Label(frame, text=name)
    ruta1_label.grid(column=0, row=1, sticky='W', **options)

    descvibra = datos.loc[:, datos.columns.str.startswith('g')]
    descVibra = descvibra.columns.values.tolist()

    parametros = ttk.Combobox(root, values=descVibra,
                              textvariable=selector, state="readonly")
    parametros.place(x=40, y=120, width=275)
    parametros.current(0)

    parametros.bind("<<ComboboxSelected>>", callbackFunc)

    return

def seleccionar_descripores_malos():
    todos=datos.columns.tolist()
    cuantosTodos=len(todos)
    limites = pd.read_excel(nameLimits)
    listaDisparo=[]
    puntoDisparo=[]
    for i in range(5, cuantosTodos):
        descrip= todos[i]   
        valor=datos[descrip].max()
        disparo=alarma_disparo(descrip, 2)[1]
        
        if float(valor)>disparo:
                listaDisparo.append(descrip)
                if 'gbx' in descrip:
                    if 'gearbox' not in puntoDisparo:
                        puntoDisparo.append('gearbox')
                elif 'gnde' in descrip:
                    if 'generator' not in puntoDisparo:
                      puntoDisparo.append('generator')
            
    ruta_decritores_malos = ttk.Label(text=listaDisparo)
    ruta_decritores_malos.place(x=40, y=310)
    ruta_decritores_malos2 = ttk.Label(text=puntoDisparo)
    ruta_decritores_malos2.place(x=40, y=350)
    
    print(listaDisparo)
    print(puntoDisparo)
    return listaDisparo,puntoDisparo 


def callbackFunc(event):

    global descripselect
    global descriptor

    print('selector.get()', selector.get())

    descriptor = selector.get()

    descripselect = selector.get()

    return


def KDE_scipy(x, x_grid, bandwidth, **kwargs):

    global kde

    """ Kernel density estimation with scipy"""

    # Requiere de importar

    # scipy.stats import gaussian_kde

    # x : valores de la variable
    # x_grid : valores para los cuales se calcula kernel.
    #           si se desea construir un histograma deben estar igualmente espaciados
    #           para generar x_grid utilizar numpy =>
    #           x_grid= np.linspace(minVal, maxval, count=len(x))

    kde = gaussian_kde(x, bw_method=bandwidth/np.std(valor), **kwargs)

    #print ('KDE.evaluate')

    return kde.evaluate(x_grid)


def calculos_estadisticos():

    global final
    global primerValorDisparo
    global valorDisparo
    global disparo
    global alarma
    global valor
    global moda
    global x_grid
    global x1, x2, y1, y2, y2Calc, Variacion

    valor = []
    valor = datos[selector.get()].tolist()

    if valor[0] < 0:
        valor[0] = 0.001
    for i in range(1, len(valor)):
        if valor[i] < 0:
            valor[i] = valor[i-1]

    # *************************************************************************

    #           ASUMIR DISTRIBUCION NORMAL ES UN ERROR

    global CantMed
    CantMed = np.count_nonzero(valor)
    alarma_disparo(descriptor, Bin)
    

    # calcular grid para un histograma

    valor_min = np.amin(valor)
    valor_max = np.amax(valor)

    x_grid = np.linspace(valor_min, valor_max, len(valor)//100)

    Rango = CantMed//5

    if Bin == 0:

        all_value()

    else:
        bines()

    return


def Calcular_Y2paraX1(x1, y1, x2, y2, y2Calc, Variacion):

    # for j in range(0, len(x2)):
    #    print (j, x2[j],y2[j])

    M = 0
    y1max_OldItems = 0

    #print("comienzo interpolacion")

    for i in range(0, len(x1)):
        if x1[i] < x2[0]:
            y2Calc[i] = 0.0
        B = 0

    # for j in range(0, len(x2)):
    #    print (j, x2[j],y2[j])

    M = 0
    for i in range(0, len(x1)):
        if y1max_OldItems < y1[i]:
            y1max_OldItems = y1[i]

        # valores a interpolar

        for j in range(M, len(x2)-1):

            #print(i, x1[i], y1[i],j, x2[j],x2[j+1], y2[j])

            if x2[j] <= x1[i] and x2[j+1] > x1[i]:
                m_pendiente = (y2[j+1]-y2[j])/(x2[j+1]-x2[j])

                b = y2[j]-m_pendiente*x2[j]

                y2Calc[i] = m_pendiente*x1[i] + b
                #print (j, x2[j],y2[j])
                #print (i ,j , M, x1[i], y1[i], x2[j], y2[j], y2Calc[i],( y2[j]-y2Calc[i]))
                M = j
                j = len(x2)-1

        #print ("salida for j")

    #print("ymax_OldItems",y1max_OldItems, "len x1", len(x1))

    for i in range(0, len(x1)):

        Variacion[i] = 100*(y2Calc[i]-y1[i])/y1max_OldItems
        #print (x1[i], y1[i], y2[i],y2Calc[i], Variacion[i])

    # print("terminado")

    return


def seleccionar_bin(event):

    global Bin
    selected_bin = Combobox_bin.get()
    Bin = selected_bin

    if Bin == 'Bin1':
        Bin = 1
    elif Bin == 'Bin2':
        Bin = 2
    elif Bin == 'Bin3':
        Bin = 3
    elif Bin == 'Bin4':
        Bin = 4
    elif Bin == 'Bin5':
        Bin = 5
    else:
        Bin = 0

    return Bin


def alarma_disparo(descriptor, Bin):

    global disparo
    global alarma

    if Bin == 1:
        Bin = 'Bin1'
    elif Bin == 2:
        Bin = 'Bin2'
    elif Bin == 3:
        Bin = 'Bin3'
    elif Bin == 4:
        Bin = 'Bin4'
    elif Bin == 5:
        Bin = 'Bin5'
    else:
        Bin = 0

    descriptor1 = str(descriptor)
    limites = pd.read_excel(nameLimits)
    descriptor_limite = limites['DescriptorName'].tolist()
    descriptor_alarma = limites['Alarm values']
    descriptor_advertencia = limites['Alert values']
    Bin1 = limites['DescriptorBin']

    for i in range(len(descriptor_limite)):
        descriptor_limite[i] = descriptor_limite[i].lower()

    Cal_limites = pd.DataFrame()
    Columns: ['descriptores', 'alarma', 'disparo', 'bin']
    Cal_limites['descriptores'] = descriptor_limite
    Cal_limites['alarma'] = descriptor_advertencia
    Cal_limites['disparo'] = descriptor_alarma
    Cal_limites['bin'] = Bin1
    
    global valoresLimites 
    if '_' in descriptor:
        print(descriptor)
        descriptor=descriptor.replace('_', '.')
       
    
    
    valoresLimites = Cal_limites[Cal_limites.descriptores == descriptor]
    limitefinal = valoresLimites[valoresLimites.bin == Bin]

    if len(limitefinal) == 0:
        alarma = valoresLimites.alarma.max()
        disparo = valoresLimites.disparo.max()
    else:
        alarma = limitefinal.alarma.tolist()[0]
        disparo = limitefinal.disparo.tolist()[0]
   
    return alarma, disparo



def bines():
    Rango = CantMed//5
    bines = datos.bin
    datos4 = pd.DataFrame()
    Columns: ['Parametro', 'Bin']
    datos4['Parametro'] = valor
    datos4['Bin'] = bines
    datos4.set_index('Bin', inplace=True)
    
    valores_bin = datos4.loc[float(Bin)]#aparece un erro cuando no hay mediciones para un bin
    valores = []
    global valores_bines
    valores_bines = valores_bin['Parametro'].tolist()
    global moda_bines
    moda_bines = stats.mode(valores_bin, axis=None)
    
    CantMed_bines = np.count_nonzero(valores_bines)
    Rango_bines = CantMed_bines//5

    df = pd.DataFrame()
    Columns: ['valor']
    df['valor'] = valores_bines
    global pdf_bines
    pdf_bines = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df)

    # grafico #1

    fig, ax = pyplot.subplots(2, 2, figsize=[9, 6])
    ax[0, 0].grid()
    ax[0, 0].set_title("Violin analysis for bin "+str(Bin))
    ax[0, 0].set_ylabel("Descriptor " + descripselect)

    sb.violinplot(ax=ax[0, 0], y=valores_bines, widths=0.4, color='lightgreen')

    # Grafico # 2

    ax[0, 1].grid()
    ax[0, 1].set_title("Violin analysis for bin "+str(Bin))
    ax[0, 0].set_ylabel("Descriptor " + descripselect)
    sb.violinplot(ax=ax[0, 1], y=valores_bines[0:Rango],
                  widths=0.4, color='skyblue')

    # grafico #3

    ax[1, 0].grid()
    ax[1, 0].set_title("Kernel density estimation for bin"+str(Bin))
    ax[1, 0].set_xlabel("Descriptor " + descripselect)

    sb.kdeplot(ax=ax[1, 0], data=valores_bines, color="green")

    # grafico #4

    ax[1, 1].grid()
    ax[1, 1].set_title("Kernel density estimation for bin"+str(Bin))
    ax[1, 1].set_xlabel("Descriptor " + descripselect)

    sb.kdeplot(ax=ax[1, 1], data=valores_bines[0:Rango], color="blue")

    kde_curve = ax[1, 1].lines[0]

    global x_bines
    global y_bines
    global ymax_bines
    global xmax_bines
    global xMax_bines
    global xMin_bines

    x_bines = kde_curve.get_xdata()
    y_bines = kde_curve.get_ydata()

    ymax_bines = y_bines.max()
    xmax_bines = x_bines[np.argmax(y_bines)]
    xMax_bines = x_bines.max()
    xMin_bines = x_bines.min()


# pagina No 2

    fig, ax = pyplot.subplots(2, 2, sharex=True, figsize=[9, 6])

    # grafico #1

    ax[0, 0].grid()
    ax[0, 0].set_title("Analysis  for bin all bin in range" + str(Rango))
    ax[0, 0].set_ylabel("Kernel (kde) / Ocurrence ")

    ax[0, 0].hist(valores_bines[Rango_bines: CantMed_bines -1], 50, facecolor='lightgreen')
    sb.kdeplot(ax=ax[0, 0], data=valores_bines[Rango_bines: CantMed_bines -1],
               color='green')   # esta repetido

    #  capturara datos  del kernel

    kde_curve = ax[0, 0].lines[0]
    global x1
    global y1
    x1 = kde_curve.get_xdata()
    y1 = kde_curve.get_ydata()

    # annot_max(x1,y1)

    # grafico #2'

    ax[0, 1].grid()
    ax[0, 1].set_title("Analysis of recents " + str(Rango_bines) + " items")
    ax[0, 1].set_ylabel("Kernel (kde) / Ocurrence ")

    ax[0, 1].hist(valores_bines[0:Rango_bines], 50, facecolor='skyblue')
    sb.kdeplot(ax=ax[0, 1], data=valores_bines[0:Rango_bines],
               color='blue')   # ultimas mediciones

    #  capturara datos  del kernel

    kde_curve = ax[0, 1].lines[0]
    x2 = kde_curve.get_xdata()
    y2 = kde_curve.get_ydata()

    # Interpolar valores de y2 para las x1

    y2Calc = np.zeros(len(x1))  # la cantidad de elementos es compatible con x1
    Variacion = np.zeros(len(x1))

    Calcular_Y2paraX1(x1, y1, x2, y2, y2Calc, Variacion)

    # grafico #3

    ax[1, 0].grid()
    ax[1, 0].set_title("Comparison of KDE for older and recents items")
    ax[1, 0].set_xlabel("Descriptor " + descripselect)
    ax[1, 0].set_ylabel("Kernel density estimator (KDE) ")
    sb.kdeplot(ax=ax[1, 0], data=valores_bines[Rango_bines: CantMed_bines-1],
               color='green')   # esta repetido
    sb.kdeplot(ax=ax[1, 0], data=valores_bines[0:Rango_bines],
               color='blue')   # ultimas mediciones
    # ax[1,0].plot(x1,Variacion)

    # grafico #4

    ax[1, 1].grid()
    ax[1, 1].set_title("Kernel variability  Index (*) ")
    ax[1, 1].set_xlabel("Descriptor " + descripselect)
    ax[1, 1].set_ylabel("Kernel variability index ( % ) ")
    ax[1, 1].plot(x1, Variacion, linewidth=2,  color='black')

    # PLOTEAR

    pyplot.suptitle(
        "Kernel and histogram historical and recents items behavior. File:" + name)

    pyplot.show()

    # -----------------------------------------------------------------------------
    # pagina #3

    fig, ax = pyplot.subplots(figsize=(9, 7))

    ax .grid()
    ax .set_title(
        "KVI - Kernel variability Index (*) = (kde_recents - kde_Old)/kde_Old_max")
    ax .set_xlabel("Descriptor " + descripselect)
    ax .set_ylabel("KVI - Kernel variability index ( % ) ")
    ax .plot(x1, Variacion, linewidth=2, color='black',
             label='KVI - Kernel Variability Index')

    # ---------------------------------------------------

    # graficar los limites a utilizar

    # limites de probabilidad
    # 20% variabilidad de KDE significativa
    # 50% Variabilidad de KDE - nivel de alerta
    # 100% Variabilidad de KDE - Nivel alarma

    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x1_mediana = (x1_max + x1_min)/2

    y1_min = np.min(Variacion)
    Alerta_Sustituto = alarma
    Alarma_Sustituto = disparo

    # inicializar x, y

    x = [0, 0]
    y = [0, 0]

    # linea de variabilidad no significativa

    x[0] = x1_min
    x[1] = x1_max
    y[0] = 20
    y[1] = 20
    pyplot.plot(x, y, color='lime', linestyle='--',
                label='KVI - Significative value', linewidth=3)

    # linea de variabilidad - nivel alerta

    x[0] = x1_mediana
    x[1] = Alerta_Sustituto
    y[0] = 50
    y[1] = 50
    pyplot.plot(x, y, color='gold', linestyle='--',
                label='KVI - Alert level', linewidth=3)

    # linea de variabilidad - nivel alarma

    x[0] = x1_mediana
    x[1] = Alarma_Sustituto
    y[0] = 100
    y[1] = 100
    pyplot.plot(x, y, color='red',  linestyle='--',
                label='KVI - Alarm level', linewidth=3)

    # limites de los valores alarma y disparo

    x[0] = Alerta_Sustituto
    x[1] = Alerta_Sustituto
    y[0] = 50
    y[1] = y1_min

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    x[0] = Alarma_Sustituto
    x[1] = Alarma_Sustituto
    y[0] = 100
    y[1] = y1_min

    pyplot.plot(x, y, color='red', label='Descriptor - Alarm value',
                linestyle='--', linewidth=3)

    # PLOTEAR

    pyplot.suptitle("KVI - Kerner variability index analysis. File:" + name)
    pyplot.legend()
    pyplot.suptitle(
        "Violin diagram and analisys. File:" + name)   
    pyplot.show()

    # pagina No 5

    fig, ax = pyplot.subplots(figsize=(9, 7))
    pyplot.title("Violin analysis for Bin" + str(Bin))
    ax.set_xlabel("Probability density")
    ax.set_ylabel("Descriptor " + descripselect)
    ax.grid()
    sb.violinplot(ax=ax, y=valores_bines,  widths=0.4, color='lightgreen')

    # limites de los valores alarma y disparo

    y[0] = Alerta_Sustituto
    y[1] = Alerta_Sustituto
    x[0] = 0.5
    x[1] = -0.5

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    y[0] = Alarma_Sustituto
    y[1] = Alarma_Sustituto
    x[0] = 0.5
    x[1] = -0.5

    pyplot.plot(x, y, color='red', label='Descriptor - Alarm value',
                linestyle='--', linewidth=3)
    pyplot.legend()

    # PLOTEAR

    eva1 = str(evaluacion1(xmax_bines))
    eva3 = str(evaluacion3(moda_bines))
    eva2 = str(evaluacion2(moda_bines, alarma, disparo))
    eva4 = str(evaluacion4(xmax_bines, alarma, disparo))

 
    eva9 = str(evaluacion9(alarma, disparo, valores_bines))
    eva0 = str(evaluacion0(disparo, alarma, valores_bines))
    

    # Annotation

    ax.annotate('Most probable value: '+eva1,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -5), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate('Common value: '+eva3,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -25), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -45), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva4,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -65), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Actual value evaluation: '+eva9,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -90), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate('Descriptor Evaluation: '+eva0,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -110), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    # ------------------------------------------------------------------
    pyplot.suptitle("Violin diagram and evaluation. File:" + name)
    pyplot.show()

    # pagina 6
    # grafico6
    # cdf
    fig, ax = pyplot.subplots(figsize=(9, 7))
    ax.grid()
    pyplot.title("CDF analysis for bin"+str(Bin))
    ax.set_xlabel("Descriptor " + descripselect)
    ax.set_ylabel("Cumulative Probability Estimation")
    sb.kdeplot(ax=ax, data=valores_bines, cumulative=True, color='blue')

    # limites de los valores alarma y disparo

    x[0] = Alerta_Sustituto
    x[1] = Alerta_Sustituto
    y[0] = 1
    y[1] = 0

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    x[0] = Alarma_Sustituto
    x[1] = Alarma_Sustituto
    y[0] = 1
    y[1] = 0

    pyplot.plot(x, y, color='red', label='Descriptor - Alarm value',
                linestyle='--', linewidth=3)
    pyplot.suptitle("KVI - Kerner variability index analysis. File:" + name)
    pyplot.legend()
    # obteniendo datos para el calculo de probabilidad
    ax = sb.kdeplot(ax=ax, data=valores_bines, cumulative=True, color='blue')
    kde_curve = ax.lines[0]

    x = kde_curve.get_xdata()
    y = kde_curve.get_ydata()
  
    cdf_disparo = y[np.where(x>=disparo)]
    cdf_alarma = y[np.where(x>=alarma)]
    
    if len(cdf_disparo)==0:
        cdf_disparo=0
    else:
        cdf_disparo=str(cdf_disparo[0])
        
    if len(cdf_alarma)==0:
        cdf_alarma=0
    else:
        cdf_alarma =str(cdf_alarma[0])

    probabilidad_aumente=str(abs(1-float(cdf_disparo))) 
    probabilidad_aumente2=str(abs(1-float(cdf_alarma))) 
    eva01=str(evaluacion01(probabilidad_aumente, probabilidad_aumente))
  
    eva02=str(consecutivo(valores_bines))
    ax.annotate('Probability of inreasing again beyond alarm value:'+probabilidad_aumente2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -330), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate('Probability of inreasing again beyond warning value'+probabilidad_aumente2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -310), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Probabilty to be equal to warning : '+ str(cdf_alarma),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -290), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Probabilty to be equal to alarm : '+  str(cdf_disparo),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -270), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva01,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -250), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate(eva02,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -350), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    

    pyplot.show()
    pyplot.suptitle(
        "CDF: Cumulative density estimation, and evaluation. File:" + name)

    return


def all_value():

    Rango = CantMed//5
    global moda
    moda = stats.mode(valor, axis=None)
    
 

    fig, ax = pyplot.subplots(2, 2, figsize=[9, 6])
    ax[0, 0].grid()
    ax[0, 0].set_title("Violin analysis. All items :" + str(CantMed))
    ax[0, 0].set_ylabel("Descriptor " + descripselect)

    sb.violinplot(ax=ax[0, 0], y=valor, widths=0.4, color='lightgreen')

    # Grafico # 2

    ax[0, 1].grid()
    ax[0, 1].set_title("Violin analysis for the recents " +
                       str(Rango)+" items (20%)")
    sb.violinplot(ax=ax[0, 1], y=valor[0:Rango], widths=0.4, color='skyblue')

    # grafico #3

    ax[1, 0].grid()
    ax[1, 0].set_title("Kernel density estimation. All items")
    ax[1, 0].set_xlabel("Descriptor " + descripselect)

    sb.kdeplot(ax=ax[1, 0], data=valor, color="green")

    # grafico #4

    ax[1, 1].grid()
    ax[1, 1].set_title(
        "Kernel density estimation for the recents " + str(Rango)+" items")
    ax[1, 1].set_xlabel("Descriptor " + descripselect)

    sb.kdeplot(ax=ax[1, 1], data=valor[0:Rango],
               color="blue")   # ultimas mediciones

    # calculando valores maximos, y kde

    df = pd.DataFrame()
    Columns: ['valor']
    df['valor'] = valor
    pdf = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df)

    kde_curve = ax[1, 0].lines[0]
    global x
    global y
    x = kde_curve.get_xdata()
    y = kde_curve.get_ydata()

    global ymax
    global ymin
    global xmax
    global xmin
    global xMin
    global xMax
    ymax = y.max()
    ymin = y.min()
    xMax = x.max()
    xMin = x.min()
    xmax = x[np.argmax(y)]
    xmin = x[np.argmin(y)]

    # evaluacion para toda la data

    # evaluacion para los valores recientes
    global valor_reciente
    valor_reciente = []
    valor_reciente = valor[0:Rango]
    moda_reciente = stats.mode(valor_reciente, axis=None)
    df = pd.DataFrame()
    Columns: ['valor']
    df['valor'] = valor_reciente
    pdf_reciente = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df)

    kde_curve = ax[1, 1].lines[0]
    x_reciente = kde_curve.get_xdata()
    y_reciente = kde_curve.get_ydata()
    ymax_reciente = y_reciente.max()
    xmax_reciente = x_reciente[np.argmax(y_reciente)]
    xMax_reciente = x_reciente.max()
    xMin_reciente = x_reciente.min()

    # titulo del grafico

    pyplot.suptitle(
        "Violin diagram for recent measaruments. File:" + name)

    pyplot.show()

    # -------------------------------------------------

    # pagina No 2

    fig, ax = pyplot.subplots(2, 2, sharex=True, figsize=[9, 6])

    # grafico #1

    ax[0, 0].grid()
    ax[0, 0].set_title("Analysis of items older than recents " + str(Rango))
    ax[0, 0].set_ylabel("Kernel (kde) / Ocurrence ")

    ax[0, 0].hist(valor[Rango: CantMed-1], 50, facecolor='lightgreen')
    sb.kdeplot(ax=ax[0, 0], data=valor[Rango: CantMed-1],
               color='green')   # esta repetido

    #  capturara datos  del kernel

    kde_curve = ax[0, 0].lines[0]
    global x1
    global y1
    x1 = kde_curve.get_xdata()
    y1 = kde_curve.get_ydata()

    # annot_max(x1,y1)

    # grafico #2'

    ax[0, 1].grid()
    ax[0, 1].set_title("Analysis of recents " + str(Rango) + " items")
    ax[0, 1].set_ylabel("Kernel (kde) / Ocurrence ")

    ax[0, 1].hist(valor[0:Rango], 50, facecolor='skyblue')
    sb.kdeplot(ax=ax[0, 1], data=valor[0:Rango],
               color='blue')   # ultimas mediciones

    #  capturara datos  del kernel

    kde_curve = ax[0, 1].lines[0]
    x2 = kde_curve.get_xdata()
    y2 = kde_curve.get_ydata()

    # Interpolar valores de y2 para las x1

    y2Calc = np.zeros(len(x1))  # la cantidad de elementos es compatible con x1
    Variacion = np.zeros(len(x1))

    Calcular_Y2paraX1(x1, y1, x2, y2, y2Calc, Variacion)

    # grafico #3

    ax[1, 0].grid()
    ax[1, 0].set_title("Comparison of KDE for older and recents items")
    ax[1, 0].set_xlabel("Descriptor " + descripselect)
    ax[1, 0].set_ylabel("Kernel density estimator (KDE) ")
    sb.kdeplot(ax=ax[1, 0], data=valor[Rango: CantMed-1],
               color='green')   # esta repetido
    sb.kdeplot(ax=ax[1, 0], data=valor[0:Rango],
               color='blue')   # ultimas mediciones
    # ax[1,0].plot(x1,Variacion)

    # grafico #4

    ax[1, 1].grid()
    ax[1, 1].set_title("Kernel variability  Index (*) ")
    ax[1, 1].set_xlabel("Descriptor " + descripselect)
    ax[1, 1].set_ylabel("Kernel variability index ( % ) ")
    ax[1, 1].plot(x1, Variacion, linewidth=2,  color='black')

    # PLOTEAR

    pyplot.suptitle(
        "Kernel density estimator for recent mesaruments. File:" + name)

    pyplot.show()

    # -----------------------------------------------------------------------------
    # pagina #3

    fig, ax = pyplot.subplots(figsize=(9, 7))

    ax .grid()
    ax .set_title(
        "KVI - Kernel variability Index (*) = (kde_recents - kde_Old)/kde_Old_max")
    ax .set_xlabel("Descriptor " + descripselect)
    ax .set_ylabel("KVI - Kernel variability index ( % ) ")
    ax .plot(x1, Variacion, linewidth=2, color='black',
             label='KVI - Kernel Variability Index')

    # ---------------------------------------------------

    # graficar los limites a utilizar

    # limites de probabilidad
    # 20% variabilidad de KDE significativa
    # 50% Variabilidad de KDE - nivel de alerta
    # 100% Variabilidad de KDE - Nivel alarma

    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x1_mediana = (x1_max + x1_min)/2

    y1_min = np.min(Variacion)
    Alerta_Sustituto = alarma
    Alarma_Sustituto = disparo

    # inicializar x, y

    x = [0, 0]
    y = [0, 0]

    # linea de variabilidad no significativa

    x[0] = x1_min
    x[1] = x1_max
    y[0] = 20
    y[1] = 20
    pyplot.plot(x, y, color='lime', linestyle='--',
                label='KVI - Significative value', linewidth=3)

    # linea de variabilidad - nivel alerta

    x[0] = x1_mediana
    x[1] = Alerta_Sustituto
    y[0] = 50
    y[1] = 50
    pyplot.plot(x, y, color='gold', linestyle='--',
                label='KVI - Alert level', linewidth=3)

    # linea de variabilidad - nivel alarma

    x[0] = x1_mediana
    x[1] = Alarma_Sustituto
    y[0] = 100
    y[1] = 100
    pyplot.plot(x, y, color='red',  linestyle='--',
                label='KVI - Alarm level', linewidth=3)

    # limites de los valores alarma y disparo

    x[0] = Alerta_Sustituto
    x[1] = Alerta_Sustituto
    y[0] = 50
    y[1] = y1_min

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    x[0] = Alarma_Sustituto
    x[1] = Alarma_Sustituto
    y[0] = 100
    y[1] = y1_min

    pyplot.plot(x, y, color='red', label='Descriptor - Alarm value',
                linestyle='--', linewidth=3)

    # PLOTEAR

    pyplot.suptitle("KVI - Kerner variability index analysis. File:" + name)
    pyplot.show()

    # --------------------------------------------------------------------------------

    # pagina No 4

    # Graficos independientes

    fig, ax = pyplot.subplots(figsize=(9, 7))
    ax.grid()
    pyplot.title("Violin analysis. All items. (" +
                 str(CantMed)+" items)")
    ax.set_xlabel("Probability density")
    ax.set_ylabel("Descriptor " + descripselect)
    sb.violinplot(ax=ax, y=valor, widths=0.4, color='lightgreen')

    # limites de los valores alarma y disparo

    y[0] = Alerta_Sustituto
    y[1] = Alerta_Sustituto
    x[0] = 0.5
    x[1] = -0.5

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    y[0] = Alarma_Sustituto
    y[1] = Alarma_Sustituto
    x[0] = 0.5
    x[1] = -0.5

    pyplot.plot(x, y, color='red', label='Descriptor - Alarm value',
                linestyle='--', linewidth=3)
    
    eva1 = str(evaluacion1(xmax))
    eva3 = str(evaluacion3(moda))
    eva2 = str(evaluacion2(moda, alarma, disparo))
    eva4 = str(evaluacion4(xmax, alarma, disparo))

 
    eva9 = str(evaluacion9(alarma, disparo, valor))
    eva0 = str(evaluacion0(disparo, alarma, valor))
    eva10=str(total(valor))
   

    # Annotation

    ax.annotate('Most probable value: '+eva1,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -5), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate('Common value: '+eva3,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -25), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -50), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva4,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -70), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Last value evaluation: '+eva9,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -90), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate('Descriptor Evaluation: '+eva0,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -110), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate(eva10,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -130), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    pyplot.suptitle("Violin diagram and evaluations. File:" + name)
    pyplot.show()


    # pagina 7
    # grafico6
    # cdf
    fig, ax = pyplot.subplots(figsize=(9, 7))
    ax.grid()
    pyplot.title("CDF analysis. All items")
    ax.set_xlabel("Descriptor " + descripselect)
    ax.set_ylabel("Cumulative Probability Estimation")
    sb.kdeplot(ax=ax, data=valor, cumulative=True, color='blue')

    # limites de los valores alarma y disparo

    x[0] = Alerta_Sustituto
    x[1] = Alerta_Sustituto
    y[0] = 1
    y[1] = 0

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    x[0] = Alarma_Sustituto
    x[1] = Alarma_Sustituto
    y[0] = 1
    y[1] = 0
    

    pyplot.plot(x, y, color='red',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)

    # obteniendo datos para el calculo de probabilidad
    ax = sb.kdeplot(ax=ax, data=valor, cumulative=True, color='blue')
    kde_curve = ax.lines[0]

    x = kde_curve.get_xdata()
    y = kde_curve.get_ydata()
  
    cdf_disparo = y[np.where(x>=disparo)]
    cdf_alarma = y[np.where(x>=alarma)]
    
    if len(cdf_disparo)==0:
        cdf_disparo=0
    else:
        cdf_disparo=str(cdf_disparo[0])
        
    if len(cdf_alarma)==0:
        cdf_alarma=0
    else:
        cdf_alarma =str(cdf_alarma[0])

    probabilidad_aumente=str(abs(1-float(cdf_disparo))) 
    probabilidad_aumente2=str(abs(1-float(cdf_alarma))) 
    
    eva01=str(evaluacion01(probabilidad_aumente, probabilidad_aumente))
    eva02=str(consecutivo(valor))
  
    
    ax.annotate('Probability of inreasing again beyond alarm value:'+probabilidad_aumente2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -330), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate('Probability of inreasing again beyond warning value'+probabilidad_aumente2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -310), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Probabilty to be equal to warning : '+ str(cdf_alarma),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -290), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Probabilty to be equal to alarm : '+  str(cdf_disparo),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -270), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva01,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -250), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate(eva02,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -350), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    

    pyplot.suptitle("CDF - Cumulative frequency analysis, and evaluations. File:" + name)
    pyplot.show()
    
    
    
    
    # pagina No 6

    # Graficos independientes

    fig, ax = pyplot.subplots(figsize=(9, 7))
    ax.grid()

    
    pyplot.title("Violin analysis for the last measurement.(" +
                 str(len(valor_reciente))+" items ")
    ax.set_xlabel("Probability density")
    ax.set_ylabel("Descriptor " + descripselect)
    sb.violinplot(ax=ax, y=valor_reciente, widths=0.4, color='lightgreen')
    
        # inicializar x, y

    x = [-0.5, 0.5]
    y = [disparo, disparo]


    pyplot.plot(x, y, color='red',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)
    
    
    
    x = [-0.5, 0.5]
    y = [alarma, alarma]

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)


    


    eva1 = str(evaluacion1(xmax_reciente))
    eva3 = str(evaluacion3(moda_reciente))
    eva2 = str(evaluacion2(moda_reciente, alarma, disparo))
    eva4 = str(evaluacion4(xmax_reciente, alarma, disparo))

 
    eva9 = str(evaluacion9(alarma, disparo, valor_reciente))
    eva0 = str(evaluacion0(disparo, alarma, valor_reciente))
    eva10= str(total(valor_reciente))
   
    # Annotation

    ax.annotate('Most probable value: '+eva1,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -5), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate('Common value: '+eva3,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -20), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -40), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva4,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -50), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('last value evaluation: '+eva9,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -70), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate('Descriptor Evaluation: '+eva0,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -90), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate(eva10,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -110), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    pyplot.suptitle("Violin diagram and evaluations. File:" + name)
    pyplot.show()



     # pagina 9
    # grafico9
    # cdf
    fig, ax = pyplot.subplots(figsize=(9, 7))
    ax.grid()
    pyplot.title("CDF analysis for the last mesurements")
    ax.set_xlabel("Descriptor " + descripselect)
    ax.set_ylabel("Cumulative Probability Estimation")
    sb.kdeplot(ax=ax, data=valor_reciente, cumulative=True, color='blue')
    
    # inicializar x, y

    x = [disparo, disparo]
    y = [0, 1]


    pyplot.plot(x, y, color='red',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)
    
    
    
    x = [alarma, alarma]
    y = [0, 1]

    pyplot.plot(x, y, color='gold',  linestyle='--',
                label='Descriptor - Alert value', linewidth=3)


    ax = sb.kdeplot(ax=ax, data=valor_reciente, cumulative=True, color='blue')
    kde_curve = ax.lines[0]

    x = kde_curve.get_xdata()
    y = kde_curve.get_ydata()
  
    cdf_disparo = y[np.where(x>=disparo)]
    cdf_alarma = y[np.where(x>=alarma)]
    
    if len(cdf_disparo)==0:
        cdf_disparo=0
    else:
        cdf_disparo=str(cdf_disparo[0])
        
    if len(cdf_alarma)==0:
        cdf_alarma=0
    else:
        cdf_alarma =str(cdf_alarma[0])

    probabilidad_aumente=str(abs(1-float(cdf_disparo))) 
    probabilidad_aumente2=str(abs(1-float(cdf_alarma))) 
    
    eva01=str(evaluacion01(probabilidad_aumente, probabilidad_aumente))
    eva02=str(consecutivo(valor_reciente))
  
    
    ax.annotate('Probability of inreasing again beyond alarm value:'+probabilidad_aumente2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -330), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.annotate('Probability of inreasing again beyond warning value'+probabilidad_aumente2,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -310), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Probabilty to be equal to the warning: '+ str(cdf_alarma),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -290), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate('Probabilty to be equal to the alarm: '+  str(cdf_disparo),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -270), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')

    ax.annotate(eva01,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -250), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    ax.annotate(eva02,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(200, -350), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    
    pyplot.suptitle("CDF - Cumulative density estimation, and evaluation. File:" + name)
    
  
    pyplot.show()
    consecutivo(valor)
    
    
    


# =============================================================================
# Evaluacion
# ===============================================================================




def evaluacion01(prob_alarma, prob_disparo):
    
    prob_alarma=float(prob_alarma)
    prob_disparo=float(prob_disparo)
    
    if prob_alarma>=0.8 and prob_disparo>=0.9:
        evaluacion=' non-hazardous condition'
    else:
          evaluacion= 'Hazardous condition'
    return evaluacion
    
    
    
def evaluacion0(disparo, alarma,valores):
    global evaluacion
    valores_alarma =[]
    valores_disparo=[]
    
    for i in valores:
        if i>=alarma:
            valores_alarma.append(i)
    
    for i in valores:
        if i>=disparo:
            valores_disparo.append(i)
    
    if len(valores_alarma)==0:
        cantidad_alarma=0
    else:
        cantidad_alarma=len(valores_alarma)
        
    if len(valores_disparo)==0:
        cantidad_disparos=0
    else:
        cantidad_disparos=len(valores_disparo)
            

    cantidad_valores=len(valores)
   
    porcentaje_alarma=cantidad_alarma*100/cantidad_valores
    porcentaje_disparo=cantidad_disparos*100/cantidad_valores
  
    
    if porcentaje_alarma == 0.04 and porcentaje_disparo <=0.01:
        evaluacion='ok. Alarm values: '+str(cantidad_disparos)+ 'and warning values: '+str(cantidad_alarma)
    elif porcentaje_alarma <= 0.7 and porcentaje_disparo <=0.02:
        evaluacion='Warning. Alarm values: '+str(cantidad_disparos)+' and warning values: '+str(cantidad_alarma)
    elif porcentaje_alarma >= 0.10 and porcentaje_disparo >=0.5:
        evaluacion='Danger. Alarm values: '+str(cantidad_disparos)+' and warning values: '+str(cantidad_alarma)
        
    return evaluacion
    
def evaluacion1(amplitud):

    return amplitud


def evaluacion3(moda):
    evaluacion=moda.mode[0]

    return evaluacion


def evaluacion2(moda, alarma, disparo):
    global evaluacion
    evaluacion = 'evaluacion'
    if int(moda.mode[0]) < alarma:
        evaluacion = 'the moda value is ok'
    elif moda.mode[0] > alarma and moda.mode[0] < disparo:

        evaluacion = ' the moda value is in warning'

    elif int(moda.mode[0]) > disparo:

        evaluacion = 'the moda value is in alarm'

    return evaluacion


def evaluacion4(amplitud, alarma, disparo):

    #alarma = int(alarma)
    global evaluacion
    evaluacion = 'evaluacion'
    
    if amplitud < alarma:

        evaluacion = 'The most probable value is ok'

    elif amplitud > alarma and amplitud < disparo:

        evaluacion = 'The most probable value warning'
    else:

        evaluacion = 'The most probable value is excessive'
    return evaluacion


def evaluacion5(moda, amplitud, descriptor, disparo):
    print('DESCRIPTOR CONDITION ASSESSMENT')
    print(disparo)
    print(moda.mode[0])
    global evaluacion

    if float(moda.mode[0]) > disparo and xmax > disparo:

        evaluacion = 'The health of the descriptor is compromised'
    else:

        evaluacion = 'The health of the descriptor it´s not compromised'

    return evaluacion


def evaluacion9(alarma, disparo, valor):
    valor0 = valor[0]
    global evaluacion
    if valor0 < alarma:
        evaluacion = 'ok'
    elif valor0 > alarma and valor0 < disparo:
        evaluacion = 'warning'
    elif valor0 > disparo:
        evaluacion = 'alarm'

    return evaluacion, valor0



def total(valor):
    total="Total measurement"+str(len(valor))
    return total

def consecutivo(valor):
    
    pocision_disparo=[]
    for i in valor:
        if (i>=disparo):
            pocision_disparo.append(valor.index(i))
    
    pocision_alarma=[]
    for i in valor:
        if (i>=alarma):
            pocision_alarma.append(valor.index(i))
    
    cuantos=[]
    for i in range(0, len(pocision_disparo)):
        cuantos.append(pocision_disparo.count(pocision_disparo[i]))
    
    cuantos1=[]
    for i in range(0, len(pocision_alarma)):
        cuantos1.append(pocision_disparo.count(pocision_alarma[i]))
    
    global maxDisparo
    global maxAlarma
    if cuantos==[]:
        maxDisparo=0
    else:
        maxDisparo=max(cuantos)
    
    if cuantos1==[]:
        maxAlarma=0
    else:
        maxAlarma=max(cuantos1)
    
    
    por_max_disparo=(maxDisparo*100)/len(valor)
    por_max_alarma=(maxAlarma*100)/len(valor)
    
    global evaluacion
    
    if por_max_disparo>=0.005 and por_max_alarma>=0.005:
        evaluacion='Danger. Exist repetibility in alarm and warning values'
    else:
        evaluacion='ok. Poor repetibility for alarma an warning value in the data'

    return evaluacion
    
   
    

# =============================================================================

# INTERFACE

# =============================================================================


ruta_label = ttk.Label(text='Select file to analysis')
ruta_label.place(x=40, y=20)

ruta1_label = ttk.Label(
    text="____________________________________________________________________________________________________")
ruta1_label.place(x=40, y=50)

button_seleccionar = ttk.Button(text='CSV File')
button_seleccionar.configure(command=SelectFichero)
button_seleccionar.place(x=240, y=20)

# seleccionar descriptor

ruta2_label = ttk.Label(text='Select Descriptor to analysis')
ruta2_label.place(x=40, y=90)


button_seleccionar1 = ttk.Button(text='Descriptor')
button_seleccionar1.configure(command=seleccionar_parametros)
button_seleccionar1.place(x=240, y=90)

try:

    datos = pd.read_csv(name)

    ruta1_label = ttk.Label(frame, text=name)
    ruta1_label.grid(column=0, row=1, sticky='W', **options)

    descvibra = datos.loc[:, datos.columns.str.startswith('g')]
    descVibra = descvibra.columns.values.tolist()

    parametros = ttk.Combobox(root, values=descVibra,
                              texstvariable=selector, state="readonly")
    parametros.place(x=40, y=120, width=275)
    parametros.current(0)

except:

    descVibra = ['Descriptors name']
    parametros = ttk.Combobox(root, values=descVibra,
                              textvariable=selector, state="disable")
    parametros.place(x=40, y=120, width=275)
    parametros.current(0)


# seleccionar bin

ruta2_label = ttk.Label(text='Select measurement bin to taking into account')
ruta2_label.place(x=40, y=150)


selected_bin = tk.StringVar()
Bin_Option = ['All Bins', 'Bin1', 'Bin2', 'Bin3', 'Bin4', 'Bin5']
Combobox_bin = ttk.Combobox(
    root, textvariable=selected_bin, width=40)   # Construir combobox
Combobox_bin['values'] = Bin_Option
Combobox_bin.bind('<<ComboboxSelected>>', seleccionar_bin)
Combobox_bin.place(x=40, y=190, width=275)

#Conbobox_bin.bind('<<ComboboxSelected>>', PointMed_Changed)
Combobox_bin.set("All bins")
selected_bin = 0

# seleccionar fihero con alertas y limites

ruta_label = ttk.Label(text='Select Alert - Alarm file ')
ruta_label.place(x=40, y=240)

ruta1_labelLimits = ttk.Label(
    text="____________________________________________________________________________________________________")
ruta1_labelLimits.place(x=40, y=270)

button_seleccionar = ttk.Button(text='Excel File')
button_seleccionar.configure(command=SelectFicheroLimites)
button_seleccionar.place(x=240, y=240)

ruta_decritores_malos = ttk.Label(text='bad descriptor')
ruta_decritores_malos.place(x=40, y=310)


ruta_decritores_malos2 = ttk.Label(text='fail element')
ruta_decritores_malos2.place(x=40, y=350)
# calcular graficar

button_calcular = ttk.Button(text='Calcular')
button_calcular.configure(command=calculos_estadisticos)
button_calcular.place(x=40, y=400)


root.mainloop()
 
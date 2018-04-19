#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:50:03 2018

@author: liangyangxiao
"""

import tkinter as tk
import sentiment_analysis as sa

def sentiment_analysis(str):
    sa.sentiment_test(str)
    var.set(sa.high_information_features())
    #var.set(sa.high_information_bigram_features())

def submit_text():
    var=Input_entry.get()
    sentiment_analysis(var)
    
windows = tk.Tk()
windows.title('Sentiment Analysis Demo')
windows.geometry('450x300')
var = tk.StringVar()

    
Input_label=tk.Label(windows,text='INPUT REVIEW',font=('Arial',12),width=12,height=2)
Input_label.place(x=170,y=50)

Input_entry=tk.Entry(windows,show=None)
Input_entry.place(x=115,y=75)

Output_label=tk.Label(windows,textvariable=var,bg='black',fg='white',font=('Arial',12),width=10,height=2)
Output_label.place(x=170,y=120)

Sumbit_button=tk.Button(windows,text='SUBMIT',width=10,height=2,command=submit_text)
Sumbit_button.place(x=162,y=170)

windows.mainloop()
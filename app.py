# Library
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# Judul
st.title('OPTIMASI SELEKSI KARYAWAN')

#Imputasi
def preprocessing(df):
    columns_to_impute = [col for col in df.columns if col not in ['Nama', 'Pendidikan']]
    preprocessor = ColumnTransformer([
        ('imputasi', SimpleImputer(strategy='constant', fill_value=0), columns_to_impute)],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessor.fit(df)
    df = preprocessor.transform(df)
    df = pd.DataFrame(df, columns=preprocessor.get_feature_names_out())
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(-1))
    df = df[cols]
    return df

#Converter
def convert_df(df):
    required_columns = ['Nama','Keterampilan','Pengalaman','Kepribadian','Motivasi','Fleksibilitas','Gaji']
    for col in required_columns:
        if col not in df.columns:
            st.error(f'Missing required columns: {col}')
            return
        
    for i in df.columns:
        if i not in ['Nama', 'Pendidikan']:
            df[i] = df[i].astype(int)
            
    df.columns = [col.replace(' ','_') for col in df.columns]
    return df
            
#Nilai
def nilai(df):
    df['Nilai'] = df[['Keterampilan', 'Pengalaman', 'Kepribadian', 'Motivasi', 'Fleksibilitas']].dot([0.2, 0.2, 0.2, 0.2, 0.2])

#Pendidikan
def pendidikan():
    options = ['S1', 'SMA', 'SMP']
    pendidikan = st.selectbox('Pilih beberapa opsi:', options)
    mapping = {'SMP': 1, 'SMA': 2, 'S1': 3}
    df['Pendidikan'] = [mapping[p] for p in df['Pendidikan']]
    pendidikan_angka = [mapping[pendidikan]]
    return pendidikan_angka

#Optimasi
def optimization(df,kuota,pengalaman,gaji,pendidikan_angka):
    #Model
    model = pyo.ConcreteModel()
    model.karyawan = RangeSet(1, len(df))
    
    # Parameter
    model.nama = Param(model.karyawan, initialize={k: df.iloc[k-1]['Nama'] for k in range(1, len(df) + 1)})
    model.gaji = Param(model.karyawan, initialize={k: df.iloc[k-1]['Gaji'] for k in range(1, len(df) + 1)})
    model.pengalaman = Param(model.karyawan, initialize={k: df.iloc[k-1]['Pengalaman'] for k in range(1, len(df) + 1)})
    model.nilai = Param(model.karyawan, initialize={k: df.iloc[k-1]['Nilai'] for k in range(1, len(df) + 1)})    
    model.pendidikan = Param(model.karyawan, initialize={k: df.iloc[k-1]['Pendidikan'] for k in range(1, len(df) + 1)})
    
    #Variable
    model.x = Var(model.karyawan, domain=Binary)
    
    # Kendala gaji
    model.kendala_gaji = pyo.ConstraintList()
    for indeks in model.karyawan:
        model.kendala_gaji.add(expr = model.gaji[indeks] * model.x[indeks] <= gaji)
    
    # Kendala pengalaman
    model.kendala_pengalaman = pyo.ConstraintList()
    for indeks in model.karyawan:
        model.kendala_pengalaman.add(expr = model.pengalaman[indeks] * model.x[indeks] >= pengalaman * model.x[indeks])
    
    # Kendala kuota
    model.kendala_kuota = Constraint(expr=sum(model.x[k] for k in model.karyawan) <= kuota)

    # Kendala pendidikan
    model.kendala_pendidikan = pyo.ConstraintList()
    for indeks in model.karyawan:
        model.kendala_pendidikan.add(expr = model.pendidikan[indeks] * model.x[indeks] == pendidikan_angka[0] * model.x[indeks])
        
    # Objektif
    model.obj = Objective(expr=sum(model.nilai[k] * model.x[k] for k in model.karyawan), sense=maximize)

    #Solver
    opt = SolverFactory('glpk')
    results = opt.solve(model, tee=True) # tee=True untuk menampilkan output solver di konsol
    
    #Cek hasil solusi
    if results.solver.status != SolverStatus.ok or results.solver.termination_condition != TerminationCondition.optimal:
        st.error(f"Solusi tidak ditemukan! Status solver: {results.solver.status}, Termination condition: {results.solver.termination_condition}")
        return
    

    st.markdown('---'*10)
    
    #Hasil optimasi
    st.write("\nHasil Seleksi Karyawan:")
    mapping = {1: 'SMP', 2: 'SMA', 3: 'S1'}
    df['Pendidikan'] = [mapping[p] for p in df['Pendidikan']]
    model.pengalaman = Param(model.karyawan, initialize={k: df.iloc[k-1]['Pengalaman'] for k in range(1, len(df) + 1)})
    for k in model.karyawan:
        if model.x[k].value == 1:
            st.write(f"Karyawan {model.nama[k]}: Dipilih (Pendidikan: {model.pendidikan[k]}, Pengalaman: {model.pengalaman[k]}, Gaji: {model.gaji[k]}, Nilai: {model.nilai[k]})")

#Upload File 
uploaded_file = st.file_uploader("Upload Excel Master Data", type=["xlsx"])

#Upload
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df = preprocessing(df)
        df = convert_df(df)
        nilai(df)
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index('Nama')))
        df = df[cols]
        st.write(df)
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")

    #Input Capacity
    kuota = st.number_input("Kuota:", min_value=0)
    pengalaman = st.number_input("Nilai Minimal Pengalaman:", min_value=0)
    gaji = st.number_input("Gaji Maksimal:", min_value=0)
    pendidikan_angka = pendidikan()
    
    if st.button("Calculate"):
        try:
            optimization(df,kuota,pengalaman,gaji,pendidikan_angka)
        except Exception as e:
            st.error(f"Error : {e}")


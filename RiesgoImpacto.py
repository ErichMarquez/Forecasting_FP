import os
import pandas as pd
from google.colab import drive
drive.mount("/content/drive")

#Carpeta principal
Proyecto="/content/drive/MyDrive/Forecasting-V2"

#Seleccion de la sucursal
indice_sucursal=2
Rutas={
    "sucursal": f"{Proyecto}/Suc{indice_sucursal}"
  }

#Subcarpetas
Subrutas={
    "modelos": f"{Rutas["sucursal"]}/modelos",
    "pronosticos": f"{Rutas["sucursal"]}/pronosticos",
    "metadatos": f"{Rutas["sucursal"]}/metadatos",
    "validacion": f"{Rutas["sucursal"]}/validacion",
    "datos": f"{Rutas["sucursal"]}/datos",
    "validacion_ventas_reales": f"{Rutas["sucursal"]}/validacion_ventas_reales"
}

for ruta in Subrutas.values():
    os.makedirs(ruta, exist_ok=True)

print("Carpetas listas en Drive")

#Parámetros a usar
import pandas as pd
import numpy as np

year=[2026]
orden_meses={"Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,"Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12}
n_meses_a_validar=6
mes="Octubre"

#Riesgo
Peso_variabilidad=0.5
Peso_error=1-Peso_variabilidad

#Error y sesgo
Decay=0.85
Min_Periodos_Validados=3
Umbral_sesgo=0.15

#Variabilidad
Ventas_semanas=52
Min_Hist=24
horizonte_std=4

#Tendencia
Umbral_Tendencia=0.05
Q_Bajo=0.4
Q_Alto=0.75
EPS=1e-6

nombre_mes={v: k for k, v in orden_meses.items()}

def normalizar_sku(serie):
    return serie.astype(str).str.lstrip("0")

def historico_semanal(Subrutas,indice_sucursal):
    archivos={f"Intermitentes sucursal {indice_sucursal}":"Intermitente",f"ML sucursal {indice_sucursal}":"MLForecast"}
    frames=[]
    for archivo,modelo in archivos.items():
        df=pd.read_csv(f"{Subrutas['datos']}/{archivo}.csv")
        df["Modelo"]=modelo
        frames.append(df)
    df=pd.concat(frames,ignore_index=True)
    df["SKU"]=normalizar_sku(df["SKU"])
    df["fecha"]=pd.to_datetime(df["fecha"])
    df["Sucursal"]=indice_sucursal
    return df.rename(columns={"ventas":"VentaReal"})

def construir_grupos(df_historico_semanal):
    grupos=df_historico_semanal[["SKU","Sucursal","Modelo"]].drop_duplicates().reset_index(drop=True)
    return grupos

def pronostico_actual(Subrutas,indice_sucursal,mes):
    df=pd.read_excel(f"{Subrutas['pronosticos']}/Reporte de pronósticos {mes}.xlsx").rename(columns={"Pronóstico de Ventas":"Pronóstico"})
    df["SKU"]=normalizar_sku(df["SKU"])
    df["Sucursal"]=indice_sucursal
    df=df[["SKU","Sucursal","Producto","Pronóstico"]]
    return df

meses_excluidos=[(2026, "Enero"),(2026, "Febrero")]

def meses_a_validar(mes,year,n=n_meses_a_validar,exclusiones=None):
    excluir=exclusiones or []
    idx=year[0]*12+(orden_meses[mes]-1)
    periodos=[]
    for k in range(n,0,-1):
        a,m0=divmod(idx-k,12)
        periodo=(a,nombre_mes[m0+1])
        if periodo not in excluir:
            periodos.append(periodo)
    return periodos

def cargar_validaciones(Subrutas,indice_sucursal,periodos):
    frames,faltantes=[],[]
    for year,mes in periodos:
        path=f"{Subrutas['validacion_ventas_reales']}/Validacion_{mes}_{year}.xlsx"
        if not os.path.exists(path):
            faltantes.append(f"{mes},{year}")
            continue
        df=pd.read_excel(path).rename(columns={"ventas":"VentaReal","Pronóstico de Ventas":"Pronóstico"})
        df["SKU"]=normalizar_sku(df["SKU"])
        df["Sucursal"]=indice_sucursal
        df["Mes"]=f"{mes} {year}"
        df["OrdenMes"]=year*12+orden_meses[mes]
        frames.append(df)
    if faltantes:
        print("Sin archivo de validación para:", ", ".join(faltantes))
    if not frames:
        raise FileNotFoundError("No se encontraron archivos de validación")
    df=pd.concat(frames,ignore_index=True).sort_values(by=["SKU","OrdenMes"])
    return df

periodos_validados=meses_a_validar(mes,year,exclusiones=meses_excluidos)
print("Ventana de validación:", periodos_validados)

hist_semanal=historico_semanal(Subrutas,indice_sucursal)
grupos=construir_grupos(hist_semanal)
df_val=cargar_validaciones(Subrutas,indice_sucursal,periodos_validados)
df_pronostico=pronostico_actual(Subrutas,indice_sucursal,mes)

print("Historico semanal:",hist_semanal.shape)
print(hist_semanal.groupby("Modelo")["SKU"].nunique())
print("Validaciones:",df_val.shape,"|meses encontrados ",df_val["Mes"].unique().tolist())
print("Pronóstico actual:",df_pronostico.shape)

def calibrar_cortes_modelo(df_historico_semanal,columna,grupo_col="Modelo",q_bajo=Q_Bajo,q_alto=Q_Alto):
    return(
        df_historico_semanal.groupby(grupo_col)[columna]
        .quantile(q=[q_bajo,q_alto])
        .unstack()
        .rename(columns={q_bajo:"corte_bajo",q_alto:"corte_alto"})
        .reset_index()
    )

def clasificacion_corte(valor,corte_bajo,corte_alto):
    if pd.isna(valor) or pd.isna(corte_bajo) or pd.isna(corte_alto):
        return "Sin datos"
    if valor<=corte_bajo:
        return "Bajo"
    if valor>=corte_alto:
        return "Alto"
    return "Medio"

#Variabilidad y promedio mensual
def calcular_variabilidad(df_historico_semanal,ventana=Ventas_semanas,min_semanas=Min_Hist):
    df=(df_historico_semanal.sort_values("fecha").groupby(["SKU","Sucursal"]).tail(ventana))
    agg=(df.groupby(["SKU","Sucursal"])["VentaReal"].agg(PromedioHistSem="mean",DesvStdHist="std",n_sem="count").reset_index())
    agg["CV"]=agg["DesvStdHist"]/agg["PromedioHistSem"].replace(0,np.nan)
    agg.loc[agg["n_sem"]<min_semanas,"CV"]=np.nan
    return agg

def calcular_promedio_periodo(df_variabilidad,semanas_por_periodo=horizonte_std):
    out=df_variabilidad[["SKU","Sucursal"]].copy()
    out["PromedioHist"]=(df_variabilidad["PromedioHistSem"]*semanas_por_periodo).clip(lower=EPS)
    return out

variabilidad=calcular_variabilidad(hist_semanal)
promedio_hist=calcular_promedio_periodo(variabilidad)
print(variabilidad.merge(grupos,on=["SKU","Sucursal"],how="left").groupby("Modelo")["CV"].describe())

#Error histórico ponderado
def calcular_error_historico(df_val,promedio_hist,decay=Decay,minimo=Min_Periodos_Validados):
    df=df_val.merge(promedio_hist,on=["SKU","Sucursal"],how="left")
    df["ErrorRel"]=(df["VentaReal"]-df["Pronóstico"]).abs()/df["PromedioHist"]

    df["meses_atras"]=(
        df.groupby(["SKU","Sucursal"])["OrdenMes"].rank(method="first",ascending=False)
    )
    df["peso"]=(decay**df["meses_atras"]).where(df["ErrorRel"].notna(),0.0)
    df["_num"]=df["ErrorRel"].fillna(0)*df["peso"]

    res=(
        df.groupby(["SKU","Sucursal"])
        .agg(_num=("_num","sum"),_den=("peso","sum"),N_val=("ErrorRel","count"))
        .reset_index()
    )
    res["ErrorHistRel"]=res["_num"]/res["_den"].replace(0,np.nan)
    res.loc[res["N_val"]<minimo,"ErrorHistRel"]=np.nan
    return res[["SKU","Sucursal","ErrorHistRel"]]

error=calcular_error_historico(df_val,promedio_hist)
print(error.merge(grupos,on=["SKU","Sucursal"],how="left").groupby("Modelo")["ErrorHistRel"].describe())

#Calculo de Confiabilidad
def calcular_confiabilidad(df_val,promedio_hist,cortes_cv_error):
    df=df_val.merge(promedio_hist,on=["SKU","Sucursal"],how="left")
    df["ErrorRel"]=(df["VentaReal"]-df["Pronóstico"]).abs()/df["PromedioHist"]

    agg=(
        df.groupby(["SKU","Sucursal"])["ErrorRel"]
        .agg(media="mean",std="std",N_val="count")
        .reset_index()
    )

    agg["CV_error"]=agg["std"]/agg["media"].replace(0,np.nan)
    agg.loc[(agg["media"]==0)&(agg["N_val"]>=Min_Periodos_Validados),"CV_error"]

    corte_bajo=cortes_cv_error["corte_bajo"].iloc[0]
    corte_alto=cortes_cv_error["corte_alto"].iloc[0]

    def _clasificar(x):
        if x["N_val"]<Min_Periodos_Validados or pd.isna(x["CV_error"]):
            return "Baja"
        if x["CV_error"]<=corte_bajo:
            return "Alta"
        if x["CV_error"]>=corte_alto:
            return "Baja"
        return "Media"

    agg["Confiabilidad"]=agg.apply(_clasificar,axis=1)
    return agg

#Calculo de Sesgo Direccional
def calcular_sesgo(df_val,promedio_hist,decay=Decay,umbral=Umbral_sesgo):
    df=df_val.merge(promedio_hist,on=["SKU","Sucursal"],how="left")
    df["Bias"]=(df["VentaReal"]-df["Pronóstico"])

    df["meses_atras"]=(
        df.groupby(["SKU","Sucursal"])["OrdenMes"].rank(method="first",ascending=False)
    )
    df["peso"]=(decay**df["meses_atras"]).where(df["Bias"].notna(),0.0)
    df["_num"]=df["Bias"].fillna(0)*df["peso"]

    res=(
        df.groupby(["SKU","Sucursal"])
        .agg(_num=("_num","sum"),_den=("peso","sum"),N=("Bias","count"),PromedioHist=("PromedioHist","first"))
        .reset_index()
    )
    res["BiasPct"]=(res["_num"]/res["_den"].replace(0,np.nan))/res["PromedioHist"]
    res.loc[res["N"]<Min_Periodos_Validados,"BiasPct"]=np.nan

    def _etiqueta(x):
        if pd.isna(x): return "Sin Datos"
        if x>umbral: return "Por debajo de lo esperado"
        if x<-umbral: return "Por encima de lo esperado"
        return "Sin sesgo"
    res["Sesgo"]=res["BiasPct"].apply(_etiqueta)
    return res[["SKU","Sucursal","BiasPct","Sesgo"]]

sesgo=calcular_sesgo(df_val,promedio_hist)
print(sesgo["Sesgo"].value_counts())

#Calibración de cortes
#Nota, estos solo se calcularán una vez cada 6 meses

def calibrar_todos_cortes(df_historico_semanal,df_val,q_bajo=Q_Bajo,q_alto=Q_Alto):
    grupos=construir_grupos(df_historico_semanal)
    variabilidad=calcular_variabilidad(df_historico_semanal)
    promedio_hist=calcular_promedio_periodo(variabilidad)

    #CV de ventas desde el histórico
    cv=variabilidad.merge(grupos,on=["SKU","Sucursal"]).dropna(subset=["CV"])
    cortes_cv=calibrar_cortes_modelo(cv,"CV",q_bajo=q_bajo,q_alto=q_alto)

    #Error relativo desde validaciones
    err=(calcular_error_historico(df_val,promedio_hist)
    .dropna(subset=["ErrorHistRel"])
    .merge(grupos,on=["SKU","Sucursal"])
    )
    cortes_error=calibrar_cortes_modelo(err,"ErrorHistRel",q_bajo=q_bajo,q_alto=q_alto)

    #CV error global
    sin_cortes=pd.DataFrame({"corte_bajo":[np.nan],"corte_alto":[np.nan]})
    conf=calcular_confiabilidad(df_val,promedio_hist,sin_cortes)
    cv_err=conf.loc[conf["N_val"]>=Min_Periodos_Validados,"CV_error"].dropna()
    cortes_cv_error=pd.DataFrame({"corte_bajo":[cv_err.quantile(q_bajo)],"corte_alto":[cv_err.quantile(q_alto)]})
    return cortes_cv, cortes_error, cortes_cv_error

#cortes_cv,cortes_error,cortes_cv_error=calibrar_todos_cortes(hist_semanal,df_val)
#print(cortes_cv,cortes_error,cortes_cv_error,sep="\n\n")

#cortes_cv.to_parquet(f"{Subrutas['metadatos']}/cortes_cv.parquet",index=False)
#cortes_error.to_parquet(f"{Subrutas['metadatos']}/cortes_error.parquet",index=False)
#cortes_cv_error.to_parquet(f"{Subrutas['metadatos']}/cortes_cv_error.parquet",index=False)

cortes_cv=pd.read_parquet(f"{Subrutas['metadatos']}/cortes_cv.parquet")
cortes_error=pd.read_parquet(f"{Subrutas['metadatos']}/cortes_error.parquet")
cortes_cv_error=pd.read_parquet(f"{Subrutas['metadatos']}/cortes_cv_error.parquet")

#Riesgo y tendencia
def calcular_riesgo(df_variabilidad,df_error,cortes_cv,cortes_error,grupos,peso_var=Peso_variabilidad,peso_err=Peso_error):
    cortes_cv=cortes_cv.rename(columns={"corte_bajo":"corte_bajo_cv","corte_alto":"corte_alto_cv"})
    cortes_error=cortes_error.rename(columns={"corte_bajo":"corte_bajo_err","corte_alto":"corte_alto_err"})

    df=(
        grupos
        .merge(df_variabilidad,on=["SKU","Sucursal"],how="left")
        .merge(df_error,on=["SKU","Sucursal"],how="left")
        .merge(cortes_cv,on="Modelo",how="left")
        .merge(cortes_error,on="Modelo",how="left")
    )

    df["Variabilidad_Modelo"]=df.apply(
        lambda x: clasificacion_corte(x["CV"],x["corte_bajo_cv"],x["corte_alto_cv"]),
        axis=1)
    df["Error_Modelo"]=df.apply(
        lambda x: clasificacion_corte(x["ErrorHistRel"],x["corte_bajo_err"],x["corte_alto_err"]),
        axis=1)

    mapa_num={"Bajo":0.0,"Medio":0.5,"Alto":1.0}
    df["Riesgo_Score"]=(peso_var*df["Variabilidad_Modelo"].map(mapa_num)+peso_err*df["Error_Modelo"].map(mapa_num))
    df["Nivel_Riesgo"]=pd.cut(df["Riesgo_Score"],bins=[-0.01,0.33,0.66,1.0],labels=["Bajo","Medio","Alto"])
    return df.drop(columns=[c for c in df.columns if c.startswith("corte")])

def calcular_tendencia(df_riesgo_actual,historico_riesgo,n_reportes=3,umbral=Umbral_Tendencia):
      previos=(
          historico_riesgo.sort_values("Fecha_reporte")
          .groupby(["SKU","Sucursal"])
          .tail(n_reportes)
          .groupby(["SKU","Sucursal"])["Riesgo_Score"].mean()
          .reset_index().rename(columns={"Riesgo_Score":"Riesgo_Score_prev"})
          )

      df=df_riesgo_actual.merge(previos,on=["SKU","Sucursal"],how="left")
      delta=df["Riesgo_Score"]-df["Riesgo_Score_prev"]

      def _clasif(y):
        if pd.isna(y): return "Sin Histórico"
        if y>umbral: return "Empeorando"
        if y<-umbral: return "Mejorando"
        return "Estable"

      df["Tendencia"]=delta.apply(_clasif)
      return df

#Calculo de Impacto
def calcular_impacto(df_pronostico_actual):
    df=df_pronostico_actual.sort_values(["Sucursal","Pronóstico"],ascending=[True,False]).copy()
    df["PctAcumSucursal"]=(
        df.groupby("Sucursal")["Pronóstico"].cumsum()/df.groupby("Sucursal")["Pronóstico"].transform("sum")
    )
    df["Impacto"]=df["PctAcumSucursal"].apply(
        lambda p:"Alto" if p<=0.80 else("Medio" if p<=0.90 else "Bajo")
    )
    return df

#Calculo de Prioridad
def calcular_prioridad(df):
    def _prioridad(z):
        if z["Confiabilidad"]=="Baja" or pd.isna(z["Nivel_Riesgo"]):
            return "Falta de Datos para estimación"
        if z["Nivel_Riesgo"]=="Alto" and z["Impacto"] in ("Alto"):
            base="Crítico"
        elif z["Nivel_Riesgo"]=="Medio" or (z["Nivel_Riesgo"]=="Alto" and z["Impacto"] in ("Medio","Bajo")):
            base="Monitorear"
        else:
            base="Normal"

        if base in ("Crítico","Monitorear"):
            if z["Sesgo"]=="Por debajo de lo esperado":
                return f"{base} (con riesgo de desabasto)"
            if z["Sesgo"]=="Por encima de lo esperado":
                return f"{base} (con riesgo de sobreinventario)"
        return base

    df["Prioridad"]=df.apply(_prioridad,axis=1)
    return df

def generar_reporte_riesgo_impacto(df_historico_semanal,df_val,df_pronostico_actual,cortes_cv,cortes_error,cortes_cv_error,historico_riesgo=None):
    grupos=construir_grupos(df_historico_semanal)
    variabilidad=calcular_variabilidad(df_historico_semanal)
    promedio_hist=calcular_promedio_periodo(variabilidad)

    error=calcular_error_historico(df_val,promedio_hist)
    riesgo=calcular_riesgo(variabilidad,error,cortes_cv,cortes_error,grupos)
    if historico_riesgo is not None and len(historico_riesgo)>0:
        historico_riesgo["Sucursal"]=historico_riesgo["Sucursal"].astype(int)
        riesgo=calcular_tendencia(riesgo,historico_riesgo)
    else:
        riesgo["Tendencia"]="Sin Histórico"

    impacto=calcular_impacto(df_pronostico_actual)
    confiabilidad=calcular_confiabilidad(df_val,promedio_hist,cortes_cv_error)
    sesgo=calcular_sesgo(df_val,promedio_hist)

    reporte=(
        impacto
        .merge(riesgo,on=["SKU","Sucursal"],how="left")
        .merge(confiabilidad[["SKU","Sucursal","CV_error","Confiabilidad"]],on=["SKU","Sucursal"],how="left")
        .merge(sesgo,on=["SKU","Sucursal"],how="left")
    )
    reporte["Tendencia"]=reporte["Tendencia"].fillna("Sin Histórico")
    reporte["Sesgo"]=reporte["Sesgo"].fillna("Sin Datos")
    reporte["Confiabilidad"]=reporte["Confiabilidad"].fillna("Baja")
    reporte=calcular_prioridad(reporte)

    columnas=["SKU","Sucursal","Producto","Pronóstico","Riesgo_Score","Nivel_Riesgo","Tendencia","Confiabilidad","Sesgo","Impacto","Prioridad"]
    return reporte[[c for c in columnas if c in reporte.columns]]

ruta_hist_riesgo=f"{Subrutas['metadatos']}/historico_riesgo.csv"
periodo_actual=f"{year[0]}-{orden_meses[mes]:02d}"

historico_riesgo=None
if os.path.exists(ruta_hist_riesgo):
    historico_riesgo=pd.read_csv(ruta_hist_riesgo,dtype={"SKU":str,"Sucursal":str})
    historico_riesgo=historico_riesgo[historico_riesgo["Fecha_reporte"]<periodo_actual]

reporte=generar_reporte_riesgo_impacto(hist_semanal,df_val,df_pronostico,cortes_cv,cortes_error,cortes_cv_error,historico_riesgo)
reporte.to_excel(f"Reporte De Pronosticos {mes} {year[0]} Sucursal {indice_sucursal}.xlsx",index=False)

nuevo=reporte[["SKU","Sucursal","Riesgo_Score"]].dropna(subset=["Riesgo_Score"]).copy()
nuevo["Fecha_reporte"]=periodo_actual

if os.path.exists(ruta_hist_riesgo):
    previo=pd.read_csv(ruta_hist_riesgo,dtype={"SKU":str,"Sucursal":str})
    previo=previo[previo["Fecha_reporte"]!=periodo_actual]
    historico_riesgo=pd.concat([previo,nuevo],ignore_index=True)
else:
    historico_riesgo=nuevo
historico_riesgo.to_csv(ruta_hist_riesgo,index=False)

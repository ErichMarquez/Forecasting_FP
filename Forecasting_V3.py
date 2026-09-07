#En mi caso estoy realizando todo en Drive para poder mandar automaticamente la información sin necesidad de almacenarla localmente
import os
from google.colab import drive
import pandas as pd
drive.mount("/content/drive")

#Carpeta principal
Proyecto="/content/drive/MyDrive/Forecasting-V2"

#Seleccion de la sucursal
Inicio_Reporte=pd.Timestamp("2026-08-03") #El lunes siguiente al último domingo que se exluye
Fin_Reporte=pd.Timestamp("2026-08-30") #Último Domingo que abarca el reporte

indice_sucursal=4
mes="Agosto"
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

import numpy as np
import joblib
import warnings
from datetime import datetime

#Rutas de Entrada para Entrenamiento
ruta_Intermitente=f"{Subrutas["datos"]}/Intermitentes sucursal {indice_sucursal}.csv"
ruta_ML=f"{Subrutas["datos"]}/ML sucursal {indice_sucursal}.csv"

#Calculo de Horizonte de pronóstico automático:
def calcular_horizonte(
    ruta_intermitente: str=ruta_Intermitente,
    ruta_ML: str=ruta_ML,
    fin_reporte: pd.Timestamp=Fin_Reporte,
    freq: str=frecuencia
 ) ->int:
    fecha_max_intermitente=pd.to_datetime(pd.read_csv(ruta_intermitente,usecols=["fecha"])["fecha"]).max()
    fecha_max_ML=pd.to_datetime(pd.read_csv(ruta_ML,usecols=["fecha"])["fecha"]).max()
    fecha_final_entrenamiento=max(fecha_max_intermitente,fecha_max_ML)

    pasos=pd.date_range(start=fecha_final_entrenamiento,end=fin_reporte,freq=freq)
    horizonte=len(pasos)-1

    if horizonte<=0:
        raise ValueError(
            f"Horizonte calculando inválido ({horizonte})."
            f"Última fecha de histórico para entrenamiento: {fecha_final_entrenamiento.date()}, "
            f"Fecha final de Pronóstico: {fin_reporte.date()}"
        )

    return horizonte

horizonte=calcular_horizonte(ruta_Intermitente,ruta_ML,Fin_Reporte)
print(f"Horizonte calculado automáticamente: {horizonte} semanas")

print(f"\nEsperando archivos en:")
print(f"{ruta_Intermitente}")
print(f"{ruta_ML}")

#Carga de los CSV
def cargar_csv(
    ruta_intermitente: str=ruta_Intermitente,
    ruta_ML: str=ruta_ML
) ->tuple[pd.DataFrame,pd.DataFrame]:
    def ord_datos(df: pd.DataFrame) -> pd.DataFrame:
      df=df.copy()
      df=df.rename(columns={
          "SKU":"unique_id",
          "ventas":"y",
          "fecha":"ds"
      })

      df["ds"]=pd.to_datetime(df["ds"],format="%Y-%m-%d")
      df["y"]=pd.to_numeric(df["y"],errors="coerce")
      df["unique_id"]=df["unique_id"].astype(str)
      df["id_prom"]=df["id_prom"].fillna(0).astype(int)
      df["fin_prom"]=pd.to_datetime(df["fin_prom"],format="%Y-%m-%d",errors="coerce")
      df["semanas_restantes_promo"]=(
          (df["fin_prom"]-df["ds"]).dt.days
          .clip(lower=0)
          .fillna(0)
          .div(7)
          .round(0)
          .astype(int)
      )
      return df.sort_values(by=["unique_id","ds"]).reset_index(drop=True)

    #Se agregan los features a futuro para los continuos
    def features_estacionalidad(df: pd.DataFrame)->pd.DataFrame:
        df=df.copy().sort_values(by=["unique_id","ds"]).reset_index(drop=True)
        semana_corr=df["ds"].dt.strftime('%U').astype(int)
        df["week"]=np.where(semana_corr==0,53,semana_corr)
        df["month"]=df["ds"].dt.month.astype(int)
        df["quarter"]=df["ds"].dt.quarter.astype(int)
        df["year"]=df["ds"].dt.year.astype(int)

        df["y_prom_semana"]=(
            df.groupby(["unique_id","week"])["y"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_semana"]=df["y_prom_semana"].fillna(
            df.groupby(["unique_id"])["y"].transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_mes"]=(
            df.groupby(["unique_id","month"])["y"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_mes"]=df["y_prom_mes"].fillna(
            df.groupby(["unique_id"])["y"].transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_trimestre"]=(
            df.groupby(["unique_id","quarter"])["y"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_trimestre"]=df["y_prom_trimestre"].fillna(
            df.groupby(["unique_id"])["y"].transform(lambda x: x.shift(1).expanding().mean())
        )
        
        coef_var=df.groupby("unique_id")["y"].transform(lambda x: x.std()/x.mean())
        df["coef_var"]=coef_var
        return df

    df_intermitente=ord_datos(pd.read_csv(ruta_intermitente))
    df_ML_sin_ft=ord_datos(pd.read_csv(ruta_ML))
    df_ML=features_estacionalidad(df_ML_sin_ft)

    df_ML.to_csv(f"{Subrutas['datos']}/ML_con_ft.csv",index=False)
    print(f"CSV Intermitente cargado SKUs: {len(df_intermitente['unique_id'].unique()):,}")
    print(f"CSV ML cargado SKUs: {len(df_ML['unique_id'].unique()):,}")

    return df_intermitente,df_ML
    
!pip install mlforecast lightgbm

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import lightgbm as lgb
from lightgbm import LGBMRegressor

#Configuración de features por sucursal de Intermitentes
lags_int_grupo_A=[4,8,12,52]
lags_int_grupo_B=[4,8,13,26,52]
lags_int_grupo_C=[4,8,13,52]

lag_transforms_int_grupo_A={
    4: [RollingMean(window_size=4),RollingStd(window_size=4)],
    8: [RollingMean(window_size=8)],
    12: [RollingMean(window_size=12)],
}

lag_transforms_int_grupo_B={
    4: [RollingMean(window_size=4),RollingStd(window_size=4)],
    8: [RollingMean(window_size=8)],
    13: [RollingMean(window_size=13)]
}

config_sucursal_int={
    1:{"lags_select":lags_int_grupo_C,"lag_transforms_corto":lag_transforms_int_grupo_B},
    2:{"lags_select":lags_int_grupo_C,"lag_transforms_corto":lag_transforms_int_grupo_B},
    3:{"lags_select":lags_int_grupo_C,"lag_transforms_corto":lag_transforms_int_grupo_B},
    4:{"lags_select":lags_int_grupo_C,"lag_transforms_corto":lag_transforms_int_grupo_B},
    6:{"lags_select":lags_int_grupo_A,"lag_transforms_corto":lag_transforms_int_grupo_A},
    7:{"lags_select":lags_int_grupo_A,"lag_transforms_corto":lag_transforms_int_grupo_A}
}

#Inicio de entrenamiento y pronóstico de Intermitentes
def entrenar_adida(
    df_intermitente: pd.DataFrame,
    horizonte: int=horizonte,
) ->MLForecast:
    
    cfg_int=config_sucursal_int[indice_sucursal]
    df_model=df_intermitente[["unique_id","ds","y"]].copy()
    df_model["unique_id"]=df_model["unique_id"].astype(int)

    #Features específicas para demanda intermitente
    df_model["prop_ventas"]=(
        df_model.groupby("unique_id")["y"]
        .transform(lambda x:(x.shift(1)>0).expanding().mean())
    )

    #Semanas desde la última demanda positiva
    def semanas_desde_ultima_venta(serie):
        result=[]
        contador=0
        for val in serie.shift(1).fillna(0):
            if val>0:
                contador=0
            else:
                contador+=1
            result.append(contador)
        return pd.Series(result, index=serie.index)

    df_model["semanas_sin_venta"] = (
        df_model.groupby("unique_id")["y"]
        .transform(semanas_desde_ultima_venta)
    )

    if horizonte<=8:
        lags=cfg_int["lags_select"]
        lag_transforms=cfg_int["lag_transforms_corto"]
        tvp=1.65
    else:
        df_model["quarter"]=df_model["ds"].dt.quarter.astype(int)

        lags=[13,26,52]
        lag_transforms={
            13: [RollingMean(window_size=13)]
            }
        tvp=1.35

    #Primera parte: Encontrar n_optimo con función objetivo compuesta
    def composite_metric(y_pred,dataset):
        y_true=dataset.get_label()
        mae=np.abs(y_pred-y_true).mean()
        rmse=np.sqrt(((y_pred-y_true)**2).mean())
        mean_y=np.abs(y_true).mean()

        if mean_y==0:
          return "composite_error",float("inf"),False

        mae_norm=mae/mean_y
        rmse_norm=rmse/mean_y
        naive_error=np.abs(np.diff(y_true)).mean()
        mase=(mae/naive_error) if naive_error>0 else float("inf")

        alpha=0.6
        beta=0.3
        gamma=0.1
        score=alpha*mae_norm+beta*rmse_norm+gamma*mase
        return "composite_error",score,False #False = Más pequeño mejor

    mlf_prep=MLForecast(
        models={"lgbm":LGBMRegressor()},
        freq=frecuencia,
        lags=lags,
        lag_transforms=lag_transforms
    )

    features_df=mlf_prep.preprocess(
        df_model,
        static_features=["unique_id"]
        )

    x_columns=[c for c in features_df.columns if c not in ["unique_id","ds","y"]]
    features_df=features_df.dropna(subset=x_columns)

    fecha_max=features_df["ds"].max()
    fecha_corte_features=fecha_max-pd.tseries.frequencies.to_offset(frecuencia)*horizonte*4

    x_train=features_df[features_df["ds"]<=fecha_corte_features][x_columns]
    y_train=features_df[features_df["ds"]<=fecha_corte_features]["y"]
    x_val=features_df[features_df["ds"]>fecha_corte_features][x_columns]
    y_val=features_df[features_df["ds"]>fecha_corte_features]["y"]

    lgb_train=lgb.Dataset(x_train,y_train)
    lgb_val=lgb.Dataset(x_val,y_val,reference=lgb_train)

    params={
        "objective":"tweedie",
        "tweedie_variance_power":tvp,
        "metric":"None",
        "learning_rate":0.05,
        "num_leaves":31,
        "max_depth":8,
        "min_child_samples":20,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "n_jobs":-1,
        "random_state":42,
        "verbose":-1
    }

    callbacks=[
        lgb.early_stopping(stopping_rounds=50,verbose=False),
        lgb.log_evaluation(period=-1)
    ]

    booster=lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        feval=composite_metric,
        callbacks=callbacks
    )

    n_optimo=booster.best_iteration
    print(f"Estimators óptimos encontrados: {n_optimo}")

    #Parte 2: Definición del modelo
    mlf = MLForecast(
        models={
            "lgbm_intermitente": LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=tvp,
                n_estimators=n_optimo,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        },
        freq=frecuencia,
        lags=lags,
        lag_transforms=lag_transforms
        )
    
    if horizonte<=8:
      print("Ejecutando cross-validation")
      cv_df=mlf.cross_validation(
          df=df_model,
          h=horizonte,
          n_windows=4,
          step_size=horizonte,
          fitted=True,
          static_features=["unique_id"]
          )
      mae=(cv_df["lgbm_intermitente"]-cv_df["y"]).abs().mean()
      rmse=np.sqrt(((cv_df["lgbm_intermitente"]-cv_df["y"])**2).mean())
      metricas_df=pd.DataFrame([{
          "modelo":"LightGBM_Intermitente",
          "MAE":round(mae,6),
          "RMSE":round(rmse,6)
          }])
      cv_por_sku=(
          cv_df.groupby("unique_id")
          .apply(lambda g: pd.Series({
             "MAE": (g["lgbm_intermitente"]-g["y"]).abs().mean(),
             "RMSE": np.sqrt(((g["lgbm_intermitente"]-g["y"])**2).mean()),
             "bias": (g["lgbm_intermitente"]-g["y"]).mean()
         }),include_groups=False)
         .reset_index()
         )
        
      skus_riesgosos=cv_por_sku[cv_por_sku["MAE"]>mae*1.5]
      if len(skus_riesgosos)>0:
          print(f"Cantidad de SKUs con riesgo en el pronóstico de intermitentes: {len(skus_riesgosos)}, con un error 1.5 veces mayor al promedio ({mae:.2f})")
          print(skus_riesgosos.sort_values("MAE", ascending=False).to_string())
          skus_riesgosos.to_excel(f"{Subrutas['validacion']}/skus_riesgosos_Intermitentes_{mes}.xlsx",index=False)

      metricas_df.to_parquet(f"{Subrutas['validacion']}/cv_metricas_intermitente_{mes}.parquet", index=False)
      cv_df.to_parquet(f"{Subrutas['validacion']}/cv_detalle_intermitente_{mes}.parquet", index=False)
      cv_por_sku.to_parquet(f"{Subrutas['validacion']}/cv_metricas_por_sku_intermitente_{mes}.parquet", index=False)
      
    else:
      print("Debido a la longuitud de horizonte no se realizará CV")

    mlf.fit(
        df_model,
        static_features=["unique_id"],
        fitted=True
        )
    joblib.dump(mlf,f"{Subrutas['modelos']}/mlf_intermitentes_{mes}.joblib")
    df_model.to_parquet(f"{Subrutas['modelos']}/df_intermitentes_{mes}.parquet",index=False)

    metada={
        "fecha_entrenamiento":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_SKUs":df_intermitente["unique_id"].nunique(),
        "horizonte":horizonte,
        "freq":frecuencia,
        "lags":lags,
        "tvp":tvp,
        "skus":df_intermitente["unique_id"].unique().tolist()
    }
    
    if horizonte<=8:
      metada["metricas_cv"]=metricas_df.to_dict(orient="records")

    joblib.dump(metada,f"{Subrutas['metadatos']}/metadatos_intermitentes_{mes}.joblib")
    
    print("Modelo Intermitente ML guardado en Drive")
    return mlf
    
#Pronóstico de Intermitentes
def pronosticar_adida(
    df_intermitente: pd.DataFrame,
    horizonte: int=horizonte
) ->pd.DataFrame:

  mlf=joblib.load(f"{Subrutas['modelos']}/mlf_intermitentes_{mes}.joblib")

  if df_intermitente is None:
    df_model=pd.read_parquet(f"{Subrutas['modelos']}/df_intermitentes_{mes}.parquet")
  else:
    df_model=df_intermitente[["unique_id","ds","y"]].copy()

  df_model["unique_id"]=df_model["unique_id"].astype(int)
  df_futuro=mlf.make_future_dataframe(h=horizonte)
  df_futuro["unique_id"]=df_futuro["unique_id"].astype(int)

  ultima_prop=(
      df_model.assign(
          prop_ventas=df_model.groupby("unique_id")["y"].transform(lambda x: (x.shift(1)>0).expanding().mean())
      )
      .groupby("unique_id")["prop_ventas"].last().reset_index()
      )

  def semanas_sin_venta_actual(grupo):
      y=grupo.sort_values("ds")["y"].values
      contador=0
      for val in reversed(y):
          if val>0:
            break
          contador+=1
      return contador

  ultimas_semanas=(
      df_model.groupby("unique_id")
      .apply(semanas_sin_venta_actual,include_groups=False)
      .reset_index()
      .rename(columns={0:"semanas_sin_venta"})
  )

  df_futuro=(df_futuro
             .merge(ultima_prop,on="unique_id",how="left")
             .merge(ultimas_semanas,on="unique_id",how="left")
             )
  if horizonte>8:
      df_futuro["quarter"]=df_futuro["ds"].dt.quarter.astype(int)
  
  forecast_df=mlf.predict(h=horizonte,X_df=df_futuro)
  forecast_df=forecast_df.rename(columns={"lgbm_intermitente":"y_pred"})
  forecast_df["y_pred"]=forecast_df["y_pred"].clip(lower=0)

  forecast_df.to_parquet(f"{Subrutas['pronosticos']}/forecast_intermitente_{mes}.parquet",index=False)

  return forecast_df
#Fin de entrenamiento y pronóstico de Intermitentes

#Configuración de features por sucursal de Continuos
features_base_ml=["y_prom_semana","y_prom_mes","y_prom_trimestre"]

lags_grupo_A=[4,8,12,52]
lags_grupo_B=[4,8,13,26,52]
lags_grupo_D=[4,8,12,13,52]

lag_transforms_grupo_A={
    4: [RollingMean(window_size=4),RollingStd(window_size=4)],
    12: [RollingMean(window_size=12)],
    52: [RollingMean(window_size=52)],
}

lag_transforms_grupo_B={
    4: [RollingMean(window_size=4),RollingStd(window_size=4)],
    8: [RollingMean(window_size=8)],
    13: [RollingMean(window_size=12)],
    26: [RollingMean(window_size=26)],
    52: [RollingMean(window_size=52)]
}

lag_transforms_grupo_C={
    4: [RollingMean(window_size=4),RollingStd(window_size=4)],
    8: [RollingMean(window_size=8)],
    12: [RollingMean(window_size=12)],
    52: [RollingMean(window_size=52)]
}

lag_transforms_grupo_D={
    4: [RollingMean(window_size=4),RollingStd(window_size=4)],
    8: [RollingMean(window_size=8),RollingStd(window_size=8)],
    12: [RollingMean(window_size=12)],
    13: [RollingMean(window_size=13)],
    52: [RollingMean(window_size=52)]
}

config_sucursal={
    1:{"lags_select":lags_grupo_D,"extra_features":["coef_var"],"lag_transforms_corto":lag_transforms_grupo_D},
    2:{"lags_select":lags_grupo_D,"extra_features":[],"lag_transforms_corto":lag_transforms_grupo_D},
    3:{"lags_select":lags_grupo_D,"extra_features":[],"lag_transforms_corto":lag_transforms_grupo_D},
    4:{"lags_select":lags_grupo_B,"extra_features":[],"lag_transforms_corto":lag_transforms_grupo_B},
    6:{"lags_select":lags_grupo_A,"extra_features":[],"lag_transforms_corto":lag_transforms_grupo_C},
    7:{"lags_select":lags_grupo_A,"extra_features":[],"lag_transforms_corto":lag_transforms_grupo_C}
}

#Inicio de entrenamiento y pronóstico de Continuos
def entrenar_mlforecast(
    df_ML: pd.DataFrame,
    horizonte: int=horizonte
) ->MLForecast:
    
    cfg=config_sucursal[indice_sucursal]
    df_model=df_ML[["unique_id","ds","y"]+features_base_ml+cfg["extra_features"]].copy()
    df_model["unique_id"]=df_model["unique_id"].astype(int)
    
    if horizonte<=8:
      lags=cfg["lags_select"]
      lag_transforms=cfg["lag_transforms_corto"]
    else:
      lags=[13,26,52]
      lag_transforms={
          13: [RollingMean(window_size=13)],
          }
    #Primera parte: Encontrar n_optimo con una función objetivo compuesta
    def composite_metric(y_pred,dataset):
        y_true=dataset.get_label()
        mae=np.abs(y_pred-y_true).mean()
        rmse=np.sqrt(((y_pred-y_true)**2).mean())
        mean_y=np.abs(y_true).mean()

        if mean_y==0:
          return "composite_error",float("inf"),False

        mae_norm=mae/mean_y
        rmse_norm=rmse/mean_y
        naive_error=np.abs(np.diff(y_true)).mean()
        mase=(mae/naive_error) if naive_error>0 else float("inf")

        alpha=1
        beta=0.0
        gamma=0.0
        score=alpha*mae_norm+beta*rmse_norm+gamma*mase
        return "composite_error",score,False #False = Más pequeño mejor

    mlf_prep=MLForecast(
        models={"lgbm":LGBMRegressor()},
        freq=frecuencia,
        lags=lags,
        lag_transforms=lag_transforms
    )

    features_df=mlf_prep.preprocess(
        df_model,
        static_features=["unique_id"]
        )

    x_columns=[c for c in features_df.columns if c not in ["unique_id","ds","y"]]
    features_df=features_df.dropna(subset=x_columns)

    fecha_max=features_df["ds"].max()
    fecha_corte_features=fecha_max-pd.tseries.frequencies.to_offset(frecuencia)*horizonte*2

    x_train=features_df[features_df["ds"]<=fecha_corte_features][x_columns]
    y_train=features_df[features_df["ds"]<=fecha_corte_features]["y"]
    x_val=features_df[features_df["ds"]>fecha_corte_features][x_columns]
    y_val=features_df[features_df["ds"]>fecha_corte_features]["y"]

    lgb_train=lgb.Dataset(x_train,y_train)
    lgb_val=lgb.Dataset(x_val,y_val,reference=lgb_train)

    params={
        "objective":"regression",
        "metric":"None",
        "learning_rate":0.03,
        "num_leaves":63,
        "max_depth":8,
        "min_child_samples":20,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "n_jobs":-1,
        "random_state":42,
        "verbose":-1
    }

    callbacks=[
        lgb.early_stopping(stopping_rounds=100,verbose=False,min_delta=0.0),
        lgb.log_evaluation(period=-1)
    ]

    booster=lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        feval=composite_metric,
        callbacks=callbacks
    )

    n_optimo=booster.best_iteration
    print(f"Estimators óptimos encontrados: {n_optimo}")

    

    #Parte 2: Definición del modelo
    mlf=MLForecast(
        models={
            "lgbm":LGBMRegressor(
                n_estimators=n_optimo,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        },
        freq=frecuencia,
        lags=lags,
        lag_transforms=lag_transforms
    )
#Parte 3: Cross_Validation en caso de ser aplicable
    if horizonte<=8:
      print("Ejecutando cross-validation")
      cv_df=mlf.cross_validation(
          df=df_model,
          h=horizonte,
          n_windows=4,
          step_size=horizonte,
          fitted=True,
          static_features=["unique_id"]
      )

      mae=(cv_df["lgbm"]-cv_df["y"]).abs().mean()
      rmse=np.sqrt(((cv_df["lgbm"]-cv_df["y"])**2).mean())

      print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

      metricas_df=pd.DataFrame([{
          "modelo":"LightGBM",
          "MAE":round(mae,6),
          "RMSE":round(rmse,6)
      }])

      cv_por_sku=(
        cv_df.groupby("unique_id")
        .apply(lambda g: pd.Series({
            "MAE": (g["lgbm"]-g["y"]).abs().mean(),
            "RMSE": np.sqrt(((g["lgbm"]-g["y"])**2).mean()),
            "bias": (g["lgbm"]-g["y"]).mean()
        }),include_groups=False)
        .reset_index()
        )

      #Skus con riesgo en el pronóstico con errores mayores al error promedio
      skus_riesgosos=cv_por_sku[cv_por_sku["MAE"]>mae*1.5]
      if len(skus_riesgosos)>0:
        print(f"Cantidad de SKUs con riesgo en el pronóstico: {len(skus_riesgosos)}, con un error 1.5 veces mayor al promedio ({mae:.2f})")
        skus_riesgosos.to_excel(f"{Subrutas['validacion']}/skus_riesgosos_ML_{mes}.xlsx",index=False)

      metricas_df.to_parquet(f"{Subrutas['validacion']}/cv_metricas_ML_{mes}.parquet",index=False)
      cv_df.to_parquet(f"{Subrutas['validacion']}/cv_detalle_ML_{mes}.parquet",index=False)
      cv_por_sku.to_parquet(f"{Subrutas['validacion']}/cv_metricas_por_sku_ML_{mes}.parquet",index=False)

    else:
      print("Debido a la longuitud de horizonte no se realizará CV")
        
    mlf.fit(
        df_model,
        static_features=["unique_id"],
        fitted=True
    )

    joblib.dump(mlf,f"{Subrutas['modelos']}/mlf_ML_{mes}.joblib")
    df_model.to_parquet(f"{Subrutas['modelos']}/df_ML_{mes}.parquet",index=False)

    metada={
        "fecha_entrenamiento":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_SKUs":df_ML["unique_id"].nunique(),
        "horizonte":horizonte,
        "freq":frecuencia,
        "lags":lags,
        "skus":df_ML["unique_id"].unique().tolist()
    }

    metada["features_exogeneas"]=features_base_ml+cfg["extra_features"]

    if horizonte<=8:
      metada["metricas_cv"]=metricas_df.to_dict(orient="records")
    joblib.dump(metada,f"{Subrutas['metadatos']}/metadatos_ML_{mes}.joblib")

    print("Modelo ML guardado en Drive")

    importancia=mlf.models_["lgbm"].feature_importances_
    features=mlf.ts.features_order_
    print(pd.Series(importancia, index=features).sort_values(ascending=False))
    return mlf
    
#Pronostico de ML
def pronosticar_mlforecast(
    df_ML: pd.DataFrame,
    horizonte: int=horizonte
) ->pd.DataFrame:

    mlf=joblib.load(f"{Subrutas['modelos']}/mlf_ML_{mes}.joblib")
    cfg=config_sucursal[indice_sucursal]
    df_ML=df_ML[["unique_id","ds","y","week","month","quarter"]+features_base_ml+cfg["extra_features"]].copy()

    df_ML["unique_id"]=df_ML["unique_id"].astype(int)
    df_ML["ds"]=pd.to_datetime(df_ML["ds"])
    
    df_futuro=mlf.make_future_dataframe(h=horizonte)
    df_futuro["unique_id"]=df_futuro["unique_id"].astype(int)

    semana_corr=df_futuro["ds"].dt.strftime('%U').astype(int)
    df_futuro["week"]=np.where(semana_corr==0,53,semana_corr)
    df_futuro["month"]=df_futuro["ds"].dt.month.astype(int)
    df_futuro["quarter"]=df_futuro["ds"].dt.quarter.astype(int)

    prom_sem=df_ML.groupby(["unique_id","week"])["y"].mean().reset_index().rename(columns={"y":"y_prom_semana"})

    df_futuro=pd.merge(
        df_futuro,
        prom_sem,
        on=["unique_id","week"],
        how="left"
    )

    prom_mes=df_ML.groupby(["unique_id","month"])["y"].mean().reset_index().rename(columns={"y":"y_prom_mes"})
    prom_trim=df_ML.groupby(["unique_id","quarter"])["y"].mean().reset_index().rename(columns={"y":"y_prom_trimestre"})

    df_futuro=(df_futuro
               .merge(prom_mes,on=["unique_id","month"],how="left")
               .merge(prom_trim,on=["unique_id","quarter"],how="left")
               )

    for feat in cfg["extra_features"]:
        feat_estatica=df_ML.groupby("unique_id")[feat].last().reset_index()
        df_futuro=df_futuro.merge(feat_estatica,on="unique_id",how="left")

    df_futuro=df_futuro.drop(columns=["week","month","quarter"],errors="ignore")

    forecast_df=mlf.predict(h=horizonte,X_df=df_futuro)
    forecast_df=forecast_df.rename(columns={"lgbm":"y_pred"})
    forecast_df["y_pred"]=forecast_df["y_pred"].clip(lower=0)

    forecast_df.to_parquet(f"{Subrutas['pronosticos']}/forecast_ML_{mes}.parquet",index=False)

    return forecast_df

#Pipeline principal
def pipeline_completo(
    ruta_intermitente: str=ruta_Intermitente,
    ruta_ML: str=ruta_ML,
    horizonte: int=horizonte,
    calendario_promo: pd.DataFrame=None,
    solo_pronosticar: bool=False,
    entrenar_intermitente: bool=True,
    entrenar_continuo: bool=True,
    pron_adida: bool=True
) ->pd.DataFrame:
    inicio=datetime.now()
    print(f"Pipeline Iniciado: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Frecuencia: {frecuencia}")
    print(f"Horizonte: {horizonte} semanas")

    print("\n [1/4] Cargando archivos desde Drive")
    df_intermitente,df_ML=cargar_csv(ruta_intermitente,ruta_ML)

    if not solo_pronosticar:
        print("\n [2/4] Entrenando modelos")
        if entrenar_intermitente:
            print("\n StatsForecast Intermitentes")
            entrenar_adida(df_intermitente,horizonte)
        else:
            print("\n Entrenamiento de StatsForecast Omitido Usando modelo guardado")

        if entrenar_continuo:
            print("\n MLForecast")
            entrenar_mlforecast(df_ML,horizonte)
        else:
            print("\n Entrenamiento de MLForecast Omitido Usando modelo guardado")
    else:
        print("\n [2/4] Cargando modelos desde Drive, solo se harán pronósticos")

    print("\n [3/4] Generando pronósticos")
    if pron_adida:
      print("\n Generando pronósticos de Intermitentes")
      forecast_intermitente=pronosticar_adida(df_intermitente, horizonte)
    else:
      print("\n No se generarán pronósticos nuevos de intermitentes, usando resultados guardados")
      forecast_intermitente=pd.read_parquet(f"{Subrutas['pronosticos']}/forecast_intermitente_{mes}.parquet")

    print("\n Generando pronósticos de MLForecast")
    forecast_ML=pronosticar_mlforecast(df_ML,horizonte)

    print(f"\n [4/4] Generando resultados")
    nombre_map=pd.concat([
        df_intermitente.drop_duplicates("unique_id")[["unique_id","producto"]],
        df_ML.drop_duplicates("unique_id")[["unique_id","producto"]]
        ]).drop_duplicates("unique_id").reset_index(drop=True)

    forecast_intermitente["Modelo"]="Intermitente"
    forecast_ML["Modelo"]="MLForecast"

    forecast_ML["unique_id"]=forecast_ML["unique_id"].astype(str)
    forecast_intermitente["unique_id"]=forecast_intermitente["unique_id"].astype(str)

    forecast_total=(
        pd.concat([forecast_intermitente,forecast_ML],ignore_index=True)
        .merge(nombre_map,on="unique_id",how="left")
    )

    forecast_total.to_parquet(f"{Subrutas['pronosticos']}/forecast_total_{mes}.parquet",index=False)
    forecast=forecast_total.copy()

    forecast["unique_id"]=pd.to_numeric(forecast["unique_id"],errors="coerce")

    forecast_total_trimed=forecast_total[
        (forecast_total["ds"]>=Inicio_Reporte) &
        (forecast_total["ds"]<=Fin_Reporte)
    ]

    forecast_total_trimed.to_parquet(f"{Subrutas['pronosticos']}/forecast_total_{mes}.parquet",index=False)

    forecast=forecast_total_trimed.copy()

    forecast["unique_id"]=pd.to_numeric(forecast["unique_id"],errors="coerce")

    reporte=(
        forecast
        .groupby(["unique_id"])
        .agg({"y_pred":"sum","Modelo":"first","producto":"first"})
        .reset_index()
        .rename(columns={"y_pred":"Pronóstico de Ventas","unique_id":"SKU","producto":"Producto"})
        .sort_values(by="SKU")
        .reset_index(drop=True)
    )

    reporte["Pronóstico de Ventas"]=np.round(reporte["Pronóstico de Ventas"])
    reporte.to_parquet(f"{Subrutas['pronosticos']}/reporte_total_{mes}.parquet",index=False)

    reporte_excel=reporte.drop(columns=["Modelo"]).copy()
    reporte_excel.to_excel(f"{Subrutas['pronosticos']}/Reporte de pronósticos {mes}.xlsx",index=False)

    duracion=(datetime.now()-inicio).seconds
    print(f"Pipeline Compleado Duración: {duracion} s")
    print(f"Archivo guardado en: {Subrutas['pronosticos']}/reporte_total.parquet")
    return forecast_total
    
#Aquí es donde selecciono el tipo de pronóstico
#Entrenamiento y pronóstico
resultado=pipeline_completo()

#Solo repronóstico
#resultado=pipeline_completo(solo_pronosticar=True)

#Entrenamiento de continuos sin intermitentes
#resultado=pipeline_completo(entrenar_intermitente=False,entrenar_continuo=True)

#Entrenamiento de intermitentes sin continuos
#resultado=pipeline_completo(entrenar_adida=True,entrenar_mlforecast=False)

#Entrenamiento de continuos y pronóstico de continuos
#resultado=pipeline_completo(entrenar_intermitente=False,entrenar_continuo=True,pron_adida=False)

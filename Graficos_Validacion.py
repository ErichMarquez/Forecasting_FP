indice_sucursal=4
mes="Agosto"

#Carga de ventas reales, Datos correspondientes a la fecha de la cual se comprobará el pronóstico
Ventas_Reales=pd.read_csv(f"{Subrutas["validacion_ventas_reales"]}/Ventas {mes} 2026.csv",header=None)

Ventas_Reales.columns=["SKU","ventas","sucursal","fecha"]

Ventas_Reales_sum=Ventas_Reales.groupby(["SKU","sucursal"])["ventas"].sum().reset_index()
Ventas_Reales_sum['ventas']=pd.to_numeric(Ventas_Reales_sum["ventas"],errors="coerce")

Ventas_Reales_Comp=(
    Ventas_Reales_sum
    .sort_values(by=["sucursal","SKU"])
    .reset_index(drop=True)
    )

cuenta_dup=Ventas_Reales_Comp.groupby(['sucursal','SKU']).size()
n_duplicados=(cuenta_dup>1).sum()
print(f"Cantidad de duplicados: {n_duplicados}")

Ventas_Reales_Suc=Ventas_Reales_Comp[Ventas_Reales_Comp["sucursal"]==indice_sucursal].drop(columns="sucursal").reset_index(drop=True)

#Carga de Pronósticos, este es el parquet que mencionó en la junta que hay que guardar.
pronostico_Mensual=pd.read_parquet(f"{Subrutas['pronosticos']}/reporte_total_{mes}.parquet")

Error=pd.merge(
    Ventas_Reales_Suc,
    pronostico_Mensual,
    on="SKU",
    how="left"
)
Error=Error.dropna().reset_index(drop=True)

Error["Diferencia"]=Error["ventas"]-Error["Pronóstico de Ventas"]
Error["Diferencia Absoluta"]=np.abs(Error["Diferencia"])

#Categorización de la respuesta del modelo
Error["Tipo de Estimación"]=np.select(
    [
        Error["Diferencia"]>0,
        Error["Diferencia"]<0
    ],
    ["Subestimación","Sobreestimación"],
    default="Estimación Exacta"
    )

Error["Magnitud del error"]=np.select(
    [
        Error["Diferencia Absoluta"]==1,
        (Error["Diferencia Absoluta"]>=2) & (Error["Diferencia Absoluta"]<=4),
        (Error["Diferencia Absoluta"]>=5) & (Error["Diferencia Absoluta"]<=10),
        Error["Diferencia Absoluta"]>=11
    ],
    ["Solo ±1 unidad","±2 a ±4 unidades","±5 a ±10 unidades","Más de ±10 unidades"],
    default="Sin error"
    )
Error=Error.drop(columns=["Diferencia"])

Error.to_parquet(f"{Subrutas['validacion_ventas_reales']}/Validacion_{mes}_2026.parquet",index=False)

Error.to_excel(f"{Subrutas['validacion_ventas_reales']}/Validacion_{mes}_2026.xlsx",index=False)

#Creacion de gráficos
fig,axes=plt.subplots(2,2,figsize=(16,22))
fig.suptitle(
    f"Evaluación del Pronóstico Total Sucursal {indice_sucursal}: {mes} 2026",
    fontsize=14, fontweight="bold", y=1.02
)

ax1,ax2,ax3,ax4=axes.flatten()
#Gráfico 1 Tipo de Estimación
orden_estimacion=["Sobreestimación","Subestimación","Estimación Exacta"]
colores_estimacion=["#C44E52","#4C72B0","#55A868"]
conteo_tipo=(
    Error["Tipo de Estimación"]
    .value_counts()
    .reindex(orden_estimacion)
    .fillna(0)
)

barras1=ax1.bar(
    conteo_tipo.index,
    conteo_tipo.values,
    color=colores_estimacion,
    edgecolor="white",
    linewidth=0.8,
    width=0.5
)

total=len(Error)
for barra,valor in zip(barras1,conteo_tipo.values):
    pct_total=valor/total*100
    ax1.text(
        barra.get_x()+barra.get_width()/2,
        barra.get_height()+total*0.01,
        f"{int(valor)}\n({pct_total:.1f}%)",
        ha="center",va="bottom",fontsize=10,fontweight="bold"
    )

ax1.set_title("Tipo de estimación que se presentó por SKU",fontsize=12,fontweight="bold",pad=12)
ax1.set_xlabel("Tipo de estimación",fontsize=10,fontweight="bold",style="italic")
ax1.set_ylabel("Número de SKUs",fontsize=10,fontweight="bold",style="italic")
ax1.set_ylim(0,conteo_tipo.max()*1.18)
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.spines[["top","right"]].set_visible(False)

#Gráfica 2 Magnitud
orden_magnitud=["Sin error","Solo ±1 unidad","±2 a ±4 unidades","±5 a ±10 unidades","Más de ±10 unidades"]

colores_magnitud=["#55A868", "#8BC34A", "#FFC107", "#FF7043", "#C44E52"]

conteo_mag=(
    Error["Magnitud del error"]
    .value_counts()
    .reindex(orden_magnitud)
    .fillna(0)
)

barras2=ax2.bar(
    range(len(orden_magnitud)),
    conteo_mag.values,
    color=colores_magnitud,
    edgecolor="white",
    linewidth=0.8,
    width=0.5
)

for barra,valor in zip(barras2,conteo_mag.values):
    pct_total=valor/total*100
    ax2.text(
        barra.get_x()+barra.get_width()/2,
        barra.get_height()+total*0.01,
        f"{int(valor)}\n({pct_total:.1f}%)",
        ha="center",va="bottom",fontsize=10,fontweight="bold"
    )

ax2.set_title("Magnitud de la diferencia por SKU",fontsize=12,fontweight="bold",pad=12)
ax2.set_xlabel("Rango de error",fontsize=10,fontweight="bold",style="italic")
ax2.set_ylabel("Número de SKUs",fontsize=10,fontweight="bold",style="italic")
ax2.set_xticks(range(len(orden_magnitud)))
ax2.set_xticklabels(orden_magnitud,fontsize=9,rotation=15,ha="right")
ax2.set_ylim(0,conteo_mag.max()*1.18)
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.spines[["top","right"]].set_visible(False)

#Gráfico 2 Magnitud de diferencia solo MLForecast
Error_ML=Error[Error["Modelo"]=="MLForecast"].reset_index(drop=True).copy()
total_ML=len(Error_ML)

conteo_mag_ML=(
    Error_ML["Magnitud del error"]
    .value_counts()
    .reindex(orden_magnitud)
    .fillna(0)
)

barras3=ax3.bar(
    range(len(orden_magnitud)),
    conteo_mag_ML.values,
    color=colores_magnitud,
    edgecolor="white",
    linewidth=0.8,
    width=0.5
)

for barra, valor in zip(barras3,conteo_mag_ML.values):
    pct_total=valor/total_ML*100
    ax3.text(
        barra.get_x()+barra.get_width()/2,
        barra.get_height()+total_ML*0.01,
        f"{int(valor)}\n({pct_total:.1f}%)",
        ha="center",va="bottom",fontsize=10,fontweight="bold"
    )

ax3.set_title(f"Magnitud de la diferencia - MLForecast ({mes} 2026)",fontsize=12,fontweight="bold",pad=12)
ax3.set_xlabel("Rango de error",fontsize=10,fontweight="bold",style="italic")
ax3.set_ylabel("Número de SKUs",fontsize=10,fontweight="bold",style="italic")
ax3.set_xticks(range(len(orden_magnitud)))
ax3.set_xticklabels(orden_magnitud,fontsize=9,rotation=15,ha="right")
ax3.set_ylim(0,conteo_mag_ML.max()*1.18)
ax3.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax3.spines[["top","right"]].set_visible(False)

#Gáfico 4:Magnitud de diferencia solo Intermitentes
Error_Int=Error[Error["Modelo"]!="MLForecast"].reset_index(drop=True).copy()
total_Int=len(Error_Int)

conteo_mag_Int=(
    Error_Int["Magnitud del error"]
    .value_counts()
    .reindex(orden_magnitud)
    .fillna(0)
)

barras4=ax4.bar(
    range(len(orden_magnitud)),
    conteo_mag_Int.values,
    color=colores_magnitud,
    edgecolor="white",
    linewidth=0.8,
    width=0.5
)

for barra, valor in zip(barras4,conteo_mag_Int.values):
    pct_total=valor/total_Int*100
    ax4.text(
        barra.get_x()+barra.get_width()/2,
        barra.get_height()+total_Int*0.01,
        f"{int(valor)}\n({pct_total:.1f}%)",
        ha="center",va="bottom",fontsize=10,fontweight="bold"
    )

ax4.set_title(f"Magnitud de la diferencia - Intermitentes ({mes} 2026)",fontsize=12,fontweight="bold",pad=12)
ax4.set_xlabel("Rango de error",fontsize=10,fontweight="bold",style="italic")
ax4.set_ylabel("Número de SKUs",fontsize=10,fontweight="bold",style="italic")
ax4.set_xticks(range(len(orden_magnitud)))
ax4.set_xticklabels(orden_magnitud,fontsize=9,rotation=15,ha="right")
ax4.set_ylim(0,conteo_mag_Int.max()*1.18)
ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax4.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{Subrutas['validacion_ventas_reales']}/evaluacion_pronostico_{mes}_2026.png",dpi=150,bbox_inches="tight", facecolor="white")

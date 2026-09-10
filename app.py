import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import matplotlib.pyplot as plt
import shap
import joblib


# CONFIGURACIÓN GENERAL DE LA APP


st.set_page_config(
    page_title="RenewAI",
    page_icon="🪴",
    layout="wide"
)


# CARGA DE DATOS


@st.cache_data
def cargar_datos():
    datos = pd.read_csv("data/datos_app.csv")

    # Compatibilidad con el nuevo dataset exportado desde el notebook
    if "fecha_hora" not in datos.columns and "fecha" in datos.columns:
        datos = datos.rename(columns={"fecha": "fecha_hora"})

    if "solar_real_mw" not in datos.columns and "solar_mw" in datos.columns:
        datos["solar_real_mw"] = datos["solar_mw"]

    if "eolica_real_mw" not in datos.columns and "eolica_onshore_mw" in datos.columns:
        datos["eolica_real_mw"] = datos["eolica_onshore_mw"]

    if "demanda_mw" not in datos.columns and "total_load_mw" in datos.columns:
        datos["demanda_mw"] = datos["total_load_mw"]

    # Convertir la columna temporal correctamente
    datos["fecha_hora"] = pd.to_datetime(
        datos["fecha_hora"],
        utc=True
    ).dt.tz_convert("Europe/Madrid")

    return datos


datos = cargar_datos()


# CARGA DE MODELOS Y VARIABLES


@st.cache_resource
def cargar_modelos():

    modelo_solar = joblib.load("models/modelo_solar.joblib")
    modelo_eolica = joblib.load("models/modelo_eolica.joblib")

    variables_solar = joblib.load("models/features_solar.joblib")
    variables_eolica = joblib.load("models/features_eolica.joblib")

    return (
        modelo_solar,
        modelo_eolica,
        variables_solar,
        variables_eolica
    )


modelo_solar, modelo_eolica, variables_solar, variables_eolica = cargar_modelos()

# Métricas finales exportadas desde el notebook (si están disponibles)
try:
    metricas_modelos = joblib.load("models/metricas_modelos.joblib")
except Exception:
    metricas_modelos = {
        "solar": {"MAE": 1015.88, "RMSE": 1786.94, "R2": 0.948},
        "eolica": {"MAE": 1201.27, "RMSE": 1649.55, "R2": 0.831},
        "cobertura": {"MAE": 6.55, "RMSE": 8.88, "R2": 0.875}
    }


# CABECERA PRINCIPAL


st.title("🪴 RenewAI")

st.subheader(
    "Predicción de generación solar y eólica en España"
)

st.write(
    "Plataforma interactiva basada en Machine Learning para el análisis "
    "de la generación solar y eólica en el sistema eléctrico español."
)


# BARRA LATERAL


with st.sidebar:

    # Título
    st.html(
        """
        <div style="margin-bottom: 25px;">
            <div style="
                font-size: 28px;
                font-weight: 700;
                color: #24252B;
                line-height: 1.2;
            ">
               🪴 RenewAI
            </div>

            <div style="
                font-size: 14px;
                color: #777777;
                margin-top: 6px;
            ">
                Predicción renovable · España
            </div>
        </div>
        """
    )

    # Navegación
    pagina = st.radio(
        "Navegación",
        [
            "📊 Panel principal",
            "🔬 Simulador What-if",
            "📈 Modelos y resultados",
            "💼 Impacto y alcance"
        ]
    )

    # Espacio entre navegación e información
    st.html(
        """
        <div style="height: 140px;"></div>
        """
    )

    # Información del proyecto
    st.html(
        """
        <div style="
            border-top: 1px solid #D9D9D9;
            padding-top: 20px;
        ">

            <div style="
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: #777777;
                margin-bottom: 7px;
            ">
                FUENTES DE DATOS
            </div>

            <div style="
                font-size: 13px;
                color: #777777;
                line-height: 1.7;
                margin-bottom: 22px;
            ">
                ENTSO-E · Open-Meteo<br>
                CAMS Copernicus · Ember
            </div>


            <div style="
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: #777777;
                margin-bottom: 7px;
            ">
                PERÍODO
            </div>

            <div style="
                font-size: 13px;
                color: #777777;
                line-height: 1.7;
                margin-bottom: 22px;
            ">
                Enero 2023 – Junio 2026<br>
                Datos horarios
            </div>


            <div style="
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: #777777;
                margin-bottom: 7px;
            ">
                MODELOS
            </div>

            <div style="
                font-size: 13px;
                color: #777777;
                line-height: 1.7;
            ">
                Gradient Boosting<br>
                Solar · Eólica
            </div>

        </div>
        """
    )


# PANEL PRINCIPAL


if pagina == "📊 Panel principal":

    st.divider()

    st.header("📊 Panel principal")

    st.write(
        "Selecciona una fecha y una hora del periodo de evaluación "
        "para consultar los resultados de RenewAI."
    )

    # Fechas disponibles


    fechas_disponibles = sorted(
        datos["fecha_hora"].dt.date.unique()
    )

    col_año, col_fecha, col_hora = st.columns(3)

    with col_año:
        año_seleccionado = st.selectbox(
            "📆 Año",
            [2025, 2026],
            key="año_panel"
        )

    if año_seleccionado == 2025:
        fecha_inicial = pd.Timestamp("2025-07-01").date()
        fecha_min = pd.Timestamp("2025-07-01").date()
        fecha_max = pd.Timestamp("2025-12-31").date()

    else:
        fecha_inicial = pd.Timestamp("2026-01-01").date()
        fecha_min = pd.Timestamp("2026-01-01").date()
        fecha_max = pd.Timestamp("2026-06-30").date()

    with col_fecha:
        fecha_seleccionada = st.date_input(
            "📅 Fecha",
            value=fecha_inicial,
            min_value=fecha_min,
            max_value=fecha_max,
            key="fecha_panel"
        )

    datos_fecha = datos[
        datos["fecha_hora"].dt.date == fecha_seleccionada
    ]

    horas_disponibles = sorted(
        datos_fecha["fecha_hora"].dt.hour.unique()
    )

    with col_hora:
        hora_seleccionada = st.selectbox(
            "🕐 Hora",
            horas_disponibles,
            format_func=lambda x: f"{x:02d}:00",
            key="hora_panel"
        )
    # Filtrar datos correspondientes a la fecha seleccionada


    datos_fecha = datos[
        datos["fecha_hora"].dt.date == fecha_seleccionada
    ]


    # Seleccionar la observación exacta


    observacion = datos_fecha[
        datos_fecha["fecha_hora"].dt.hour == hora_seleccionada
    ].iloc[0]


    # Mostrar momento seleccionado


    st.caption(
        f"Observación seleccionada: "
        f"{fecha_seleccionada.strftime('%d/%m/%Y')} "
        f"a las {hora_seleccionada:02d}:00"
    )

    # Indicadores principales


    st.subheader("Resultados")

    solar_pred = observacion["solar_pred_mw"]
    eolica_pred = observacion["eolica_pred_mw"]
    renovable_pred = observacion["renovable_pred_mw"]
    cobertura_pred = observacion["cobertura_pred_pct"]
    precio = observacion["precio_eur_mwh"]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="☀️ Solar",
            value=f"{solar_pred:,.0f} MW"
        )
        st.caption("Predicción")

    with col2:
        st.metric(
            label="🌬️ Eólica",
            value=f"{eolica_pred:,.0f} MW"
        )
        st.caption("Predicción")

    with col3:
        st.metric(
            label="⚡ Solar + eólica",
            value=f"{renovable_pred:,.0f} MW"
        )
        st.caption("Predicción")

    with col4:
        st.metric(
            label="🔋 Cobertura solar-eólica",
            value=f"{cobertura_pred:.1f} %",
            help=(
                "Cobertura estimada de la demanda eléctrica mediante generación "
                "solar y eólica onshore. Se calcula a partir de la generación "
                "predicha por RenewAI y la demanda observada para esta hora. "
                "No corresponde a una predicción directa de un modelo independiente."
            )
        )
        st.caption("Estimada")

    with col5:
        st.metric(
            label="💶 Precio mayorista",
            value=f"{precio:.2f} €/MWh",
            help=(
                "Precio mayorista de la electricidad observado históricamente "
                "para la fecha y hora seleccionadas. Se utiliza únicamente como "
                "información de contexto y no es una predicción de RenewAI."
            )
        )
        st.caption("Observado")
     
    # Evolución diaria: Predicho vs Real


    st.divider()
    st.subheader("Evolución durante el día seleccionado")

    # Ordenar las observaciones del día por hora
    datos_dia = datos_fecha.sort_values("fecha_hora").copy()

    
    # Gráfico solar
   

    fig_solar = go.Figure()

    fig_solar.add_trace(
        go.Scatter(
            x=datos_dia["fecha_hora"],
            y=datos_dia["solar_real_mw"],
            mode="lines",
            name="Real",
            line=dict(color="#1B5E20", width=3)
        )
    )

    fig_solar.add_trace(
        go.Scatter(
            x=datos_dia["fecha_hora"],
            y=datos_dia["solar_pred_mw"],
            mode="lines",
            name="Predicción",
            line=dict(color="#66BB6A", width=3)
        )
    )

    # Marcar la hora seleccionada
    momento_seleccionado = observacion["fecha_hora"]

    fig_solar.add_vline(
        x=momento_seleccionado.timestamp() * 1000,
        line_dash="dash"
    )

    fig_solar.update_layout(
        title="☀️ Generación solar — Real vs. Predicha",
        xaxis_title="Hora",
        yaxis_title="Generación (MW)",
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig_solar,
        use_container_width=True
    )

 
    # Gráfico eólico
  

    fig_eolica = go.Figure()

    fig_eolica.add_trace(
        go.Scatter(
            x=datos_dia["fecha_hora"],
            y=datos_dia["eolica_real_mw"],
            mode="lines",
            name="Real",
            line=dict(color="#1B5E20", width=3)
        )
    )

    fig_eolica.add_trace(
        go.Scatter(
            x=datos_dia["fecha_hora"],
            y=datos_dia["eolica_pred_mw"],
            mode="lines",
            name="Predicción",
            line=dict(color="#66BB6A", width=3)
        )
    )

    # Marcar la hora seleccionada
    fig_eolica.add_vline(
        x=momento_seleccionado.timestamp() * 1000,
        line_dash="dash"
    )

    fig_eolica.update_layout(
        title="🌬️ Generación eólica — Real vs. Predicha",
        xaxis_title="Hora",
        yaxis_title="Generación (MW)",
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig_eolica,
        use_container_width=True
    )

    
    # Cobertura de la demanda: Estimada vs Real
   

    st.divider()
    st.subheader("Cobertura de la demanda")

    fig_cobertura = go.Figure()

    fig_cobertura.add_trace(
        go.Scatter(
            x=datos_dia["fecha_hora"],
            y=datos_dia["cobertura_real_pct"],
            mode="lines",
            name="Real",
            line=dict(color="#1B5E20", width=3)
        )
    )

    fig_cobertura.add_trace(
        go.Scatter(
            x=datos_dia["fecha_hora"],
            y=datos_dia["cobertura_pred_pct"],
            mode="lines",
            name="Estimada",
            line=dict(color="#66BB6A", width=3)
        )
    )

    # Marcar la hora seleccionada
    fig_cobertura.add_vline(
        x=momento_seleccionado.timestamp() * 1000,
        line_dash="dash"
    )

    fig_cobertura.update_layout(
        title="🔋 Cobertura solar-eólica de la demanda — Real vs. Estimada",
        xaxis_title="Hora",
        yaxis_title="Cobertura de la demanda (%)",
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig_cobertura,
        use_container_width=True
    )

    st.caption(
        "La cobertura representa el porcentaje de la demanda eléctrica "
        "cubierto por la generación solar y eólica onshore. La cobertura "
        "estimada se calcula a partir de las predicciones de generación "
        "de RenewAI y la demanda observada."
    )



# SIMULADOR WHAT-IF

elif pagina == "🔬 Simulador What-if":


    st.divider()

    st.header("🔬 Simulador What-if")


    st.markdown(
        """
        Explora cómo podrían variar las predicciones de generación renovable
        ante cambios hipotéticos en las condiciones meteorológicas.
        """
    )

    st.html(
        """
        <div style="
            background-color: #F4FAF5;
            border: 1px solid #DDEEDF;
            border-radius: 12px;
            padding: 16px 18px;
            margin-top: 10px;
            margin-bottom: 25px;
            font-size: 14px;
            line-height: 1.6;
            color: #4F5B52;
        ">
            El simulador parte de una observación histórica y modifica determinadas
            variables meteorológicas manteniendo el resto de condiciones constantes.
            Los resultados representan escenarios hipotéticos basados en los modelos
            de RenewAI y no predicciones causales.
        </div>
        """
    )

  
    # Escenario de referencia


    st.subheader("1. Selecciona el escenario de referencia")

    st.write(
        "Selecciona una fecha y una hora del periodo de evaluación. "
        "Esta observación servirá como punto de partida para la simulación."
    )


    # Selección del escenario histórico de referencia


    fechas_disponibles_whatif = sorted(
        datos["fecha_hora"].dt.date.unique()
    )

    col_año_whatif, col_fecha_whatif, col_hora_whatif = st.columns(3)

    with col_año_whatif:
        año_whatif = st.selectbox(
            "📆 Año",
            [2025, 2026],
            key="año_whatif"
        )

    if año_whatif == 2025:
        fecha_inicial_whatif = pd.Timestamp("2025-07-01").date()
        fecha_min_whatif = pd.Timestamp("2025-07-01").date()
        fecha_max_whatif = pd.Timestamp("2025-12-31").date()

    else:
        fecha_inicial_whatif = pd.Timestamp("2026-01-01").date()
        fecha_min_whatif = pd.Timestamp("2026-01-01").date()
        fecha_max_whatif = pd.Timestamp("2026-06-30").date()

    with col_fecha_whatif:
        fecha_whatif = st.date_input(
            "📅 Fecha",
            value=fecha_inicial_whatif,
            min_value=fecha_min_whatif,
            max_value=fecha_max_whatif,
            key="fecha_whatif"
        )

    datos_fecha_whatif = datos[
        datos["fecha_hora"].dt.date == fecha_whatif
    ]

    horas_disponibles_whatif = sorted(
        datos_fecha_whatif["fecha_hora"].dt.hour.unique()
    )

    with col_hora_whatif:
        hora_whatif = st.selectbox(
            "🕐 Hora",
            horas_disponibles_whatif,
            format_func=lambda x: f"{x:02d}:00",
            key="hora_whatif"
        )

    observacion_whatif = datos_fecha_whatif[
        datos_fecha_whatif["fecha_hora"].dt.hour == hora_whatif
    ].iloc[0]

    st.caption(
        f"Escenario de referencia: "
        f"{fecha_whatif.strftime('%d/%m/%Y')} "
        f"a las {hora_whatif:02d}:00"
    )

    # Escenario base


    st.subheader("2. Escenario base")

    st.write(
        "Predicciones de RenewAI para las condiciones originales "
        "de la observación seleccionada."
    )

    solar_base = observacion_whatif["solar_pred_mw"]
    eolica_base = observacion_whatif["eolica_pred_mw"]
    renovable_base = solar_base + eolica_base
    demanda_base = observacion_whatif["demanda_mw"]

    cobertura_base = (
        renovable_base / demanda_base
    ) * 100


    col1_base, col2_base, col3_base, col4_base = st.columns(4)

    with col1_base:
        st.metric(
            "☀️ Solar",
            f"{solar_base:,.0f} MW"
        )

    with col2_base:
        st.metric(
            "🌬️ Eólica",
            f"{eolica_base:,.0f} MW"
        )

    with col3_base:
        st.metric(
            "⚡ Solar + eólica",
            f"{renovable_base:,.0f} MW"
        )

    with col4_base:
        st.metric(
            "🔋 Cobertura",
            f"{cobertura_base:.1f} %"
        )

     
    # Comprobación de la predicción base con los modelos

    # Recuperar las variables de entrada de la observación seleccionada
    X_solar_base = observacion_whatif[
        [f"solar_input__{var}" for var in variables_solar]
    ].copy()

    X_eolica_base = observacion_whatif[
        [f"eolica_input__{var}" for var in variables_eolica]
    ].copy()

    # Restaurar los nombres originales esperados por los modelos
    X_solar_base.index = variables_solar
    X_eolica_base.index = variables_eolica

    # Convertir a DataFrame de una sola fila
    X_solar_base = X_solar_base.to_frame().T
    X_eolica_base = X_eolica_base.to_frame().T

    # Ejecutar los modelos
    solar_modelo_base = modelo_solar.predict(X_solar_base)[0]
    eolica_modelo_base = modelo_eolica.predict(X_eolica_base)[0]


    # Modificación de condiciones


    st.divider()

    st.subheader("3. Modifica las condiciones")

    st.write(
        "Ajusta las condiciones del escenario respecto a la observación histórica "
        "seleccionada."
    )

  
    # Condiciones meteorológicas
 

    st.markdown("#### 🌤️ Condiciones meteorológicas")

    col_solar_sim, col_eolica_sim = st.columns(2)

    with col_solar_sim:

        cambio_radiacion = st.slider(
            "☀️ Irradiancia solar",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            format="%d %%",
            key="cambio_radiacion"
        )

    with col_eolica_sim:

        cambio_viento = st.slider(
            "🌬️ Velocidad del viento a 100 m",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            format="%d %%",
            key="cambio_viento"
        )



    # Condiciones del sistema


    st.markdown("#### ⚡ Condiciones del sistema")

    cambio_demanda = st.slider(
        "🔋 Demanda eléctrica",
        min_value=-30,
        max_value=30,
        value=0,
        step=5,
        format="%d %%",
        key="cambio_demanda"
    )

    st.caption(
        "La demanda no es una variable predictora de los modelos de generación. "
        "Su modificación se utiliza únicamente para recalcular la cobertura "
        "solar-eólica del escenario simulado."
    )

    
    # Escenario solar simulado


    X_solar_sim = X_solar_base.copy()

    factor_radiacion = 1 + cambio_radiacion / 100

    # Modificar las variables de irradiancia/radiación
    for var in variables_solar:

        if "cams_ghi" in var or "radiacion_solar" in var:
            X_solar_sim[var] = X_solar_sim[var] * factor_radiacion

    # Recalcular las variables de solar efectiva
    for var in variables_solar:

        if "solar_efectiva" in var:

            localizacion = var.replace("_solar_efectiva", "")

            var_radiacion = f"{localizacion}_radiacion_solar"
            var_nubes = f"{localizacion}_nubes_pct"

            if (
                var_radiacion in X_solar_sim.columns
                and var_nubes in X_solar_sim.columns
            ):
                X_solar_sim[var] = (
                    X_solar_sim[var_radiacion]
                    * (1 - X_solar_sim[var_nubes] / 100)
                )

            else:
                # Si las variables originales no están entre las variables finales,
                # aplicar el mismo cambio proporcional a solar_efectiva
                X_solar_sim[var] = (
                    X_solar_sim[var] * factor_radiacion
                )

    # Nueva predicción solar
    solar_simulada = modelo_solar.predict(X_solar_sim)[0]

  
    # Escenario eólico simulado
 

    X_eolica_sim = X_eolica_base.copy()

    factor_viento = 1 + cambio_viento / 100

    # Modificar las velocidades de viento a 100 m
    for var in variables_eolica:

        if "viento_100m" in var and "cubo" not in var:
            X_eolica_sim[var] = X_eolica_sim[var] * factor_viento

    # Recalcular las transformaciones cúbicas
    for var in variables_eolica:

        if "viento_100m_cubo" in var:

            var_base = var.replace("_cubo", "")

            if var_base in X_eolica_sim.columns:
                X_eolica_sim[var] = X_eolica_sim[var_base] ** 3

    # Nueva predicción eólica
    eolica_simulada = modelo_eolica.predict(X_eolica_sim)[0]

   
    # Resultados del escenario simulado
  

    # Demanda del escenario
    factor_demanda = 1 + cambio_demanda / 100
    demanda_simulada = demanda_base * factor_demanda

    # Generación renovable total simulada
    renovable_simulada = solar_simulada + eolica_simulada

    # Cobertura simulada
    cobertura_simulada = (
        renovable_simulada / demanda_simulada
    ) * 100

    
    # Resultados del escenario
    

    st.divider()

    st.subheader("4. Resultados del escenario")

    st.write(
        "Comparación entre el escenario histórico de referencia "
        "y el escenario hipotético definido mediante los controles anteriores."
    )

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)

    with col_res1:
        st.metric(
            "☀️ Solar",
            f"{solar_simulada:,.0f} MW",
            delta=f"{solar_simulada - solar_base:,.0f} MW"
        )

    with col_res2:
        st.metric(
            "🌬️ Eólica",
            f"{eolica_simulada:,.0f} MW",
            delta=f"{eolica_simulada - eolica_base:,.0f} MW"
        )

    with col_res3:
        st.metric(
            "⚡ Solar + eólica",
            f"{renovable_simulada:,.0f} MW",
            delta=f"{renovable_simulada - renovable_base:,.0f} MW"
        )

    with col_res4:
        st.metric(
            "🔋 Cobertura",
            f"{cobertura_simulada:.1f} %",
            delta=f"{cobertura_simulada - cobertura_base:+.1f} p.p."
        )

       
    # Comparación base vs. simulado


    st.markdown("#### Comparación de generación")

    comparacion_generacion = pd.DataFrame({
        "Fuente": [
            "Solar",
            "Eólica",
            "Solar + eólica"
        ],
        "Base": [
            solar_base,
            eolica_base,
            renovable_base
        ],
        "Simulado": [
            solar_simulada,
            eolica_simulada,
            renovable_simulada
        ]
    })

    comparacion_generacion = comparacion_generacion.melt(
        id_vars="Fuente",
        var_name="Escenario",
        value_name="Generación (MW)"
    )

    fig_comparacion = px.bar(
        comparacion_generacion,
        x="Fuente",
        y="Generación (MW)",
        color="Escenario",
        text_auto=",.0f",
        barmode="group",
        color_discrete_map={
            "Base": "#2E7D32",
            "Simulado": "#81C784"
        }
    )

    fig_comparacion.update_layout(
        xaxis_title="",
        yaxis_title="Generación (MW)",
        legend_title="",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(
        fig_comparacion,
        use_container_width=True
    )


    # Interpretación del escenario


    st.markdown("#### Interpretación del escenario")

    cambio_solar_mw = solar_simulada - solar_base
    cambio_eolica_mw = eolica_simulada - eolica_base
    cambio_renovable_mw = renovable_simulada - renovable_base
    cambio_cobertura_pp = cobertura_simulada - cobertura_base

    if cambio_renovable_mw > 0:
        tendencia_generacion = "aumentaría"
    elif cambio_renovable_mw < 0:
        tendencia_generacion = "disminuiría"
    else:
        tendencia_generacion = "se mantendría sin cambios"

    if cambio_cobertura_pp > 0:
        tendencia_cobertura = "aumentaría"
    elif cambio_cobertura_pp < 0:
        tendencia_cobertura = "disminuiría"
    else:
        tendencia_cobertura = "se mantendría estable"

    st.html(
        f"""
        <div style="
            background-color: #F4FAF5;
            border: 1px solid #DDEEDF;
            border-radius: 12px;
            padding: 16px 18px;
            margin-top: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.7;
            color: #4F5B52;
        ">
            Bajo las condiciones simuladas, la generación conjunta solar y eólica
            <b>{tendencia_generacion}</b> en
            <b>{abs(cambio_renovable_mw):,.0f} MW</b> respecto al escenario base.
            Como resultado de los cambios en generación y demanda, la cobertura
            solar-eólica <b>{tendencia_cobertura}</b> en
            <b>{abs(cambio_cobertura_pp):.1f} puntos porcentuales</b>.
        </div>
        """
    )



# MODELOS Y RESULTADOS


elif pagina == "📈 Modelos y resultados":

    st.divider()

    st.header("📈 Modelos y resultados")

    st.write(
        "Evaluación de los modelos de Machine Learning seleccionados para "
        "la predicción de generación solar y eólica en España."
    )

    st.html(
    """
    <div style="
        background-color: #F4FAF5;
        border: 1px solid #DDEEDF;
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 25px;
        color: #374151;
    ">
        El rendimiento mostrado corresponde al conjunto de test temporal
        comprendido entre julio de 2025 y junio de 2026. Estas métricas
        permiten evaluar el comportamiento global de los modelos y no
        únicamente su desempeño en una fecha concreta.
    </div>
    """
)

 
    # Modelos finales


    st.subheader("Modelos finales")

    col_solar, col_eolica = st.columns(2)

    with col_solar:
        st.html(
            """
            <div style="
                background-color: #FAFAFA;
                border: 1px solid #C8E6C9;
                border-radius: 16px;
                padding: 22px;
            ">
                <h3 style="margin-top: 0;">☀️ Modelo solar</h3>

                <p style="margin-bottom: 5px;">
                    <strong>Gradient Boosting Regressor</strong>
                </p>

                <p style="
                    color: #52745A;
                    margin-top: 0;
                    margin-bottom: 22px;
                ">
                    13 variables · 200 estimadores
                </p>

                <table style="
                    width: 100%;
                    border-collapse: collapse;
                    text-align: left;
                ">
                    <tr style="color: #52745A;">
                        <td>MAE</td>
                        <td>RMSE</td>
                        <td>R²</td>
                    </tr>

                    <tr style="
                        font-size: 22px;
                        font-weight: 700;
                    ">
                        <td>1,015.88</td>
                        <td>1,786.94</td>
                        <td>0.948</td>
                    </tr>

                    <tr style="
                        color: #52745A;
                        font-size: 13px;
                    ">
                        <td>MW</td>
                        <td>MW</td>
                        <td>Test</td>
                    </tr>
                </table>
            </div>
            """
        )

    with col_eolica:
        st.html(
            """
            <div style="
                background-color: #FAFAFA;
                border: 1px solid #C8E6C9;
                border-radius: 16px;
                padding: 22px;
            ">
                <h3 style="margin-top: 0;">🌬️ Modelo eólico</h3>

                <p style="margin-bottom: 5px;">
                    <strong>Gradient Boosting Regressor</strong>
                </p>

                <p style="
                    color: #52745A;
                    margin-top: 0;
                    margin-bottom: 22px;
                ">
                    42 variables · 300 estimadores
                </p>

                <table style="
                    width: 100%;
                    border-collapse: collapse;
                    text-align: left;
                ">
                    <tr style="color: #52745A;">
                        <td>MAE</td>
                        <td>RMSE</td>
                        <td>R²</td>
                    </tr>

                    <tr style="
                        font-size: 22px;
                        font-weight: 700;
                    ">
                        <td>1,201.27</td>
                        <td>1,649.55</td>
                        <td>0.831</td>
                    </tr>

                    <tr style="
                        color: #52745A;
                        font-size: 13px;
                    ">
                        <td>MW</td>
                        <td>MW</td>
                        <td>Test</td>
                    </tr>
                </table>
            </div>
            """
        )

  
    # Explicabilidad SHAP


    st.divider()

    st.subheader("Explicabilidad de los modelos")

    st.write(
        "La metodología SHAP permite analizar qué variables tienen mayor "
        "influencia sobre las predicciones realizadas por cada modelo."
    )

    col_shap_solar, col_shap_eolica = st.columns(2)

    with col_shap_solar:
        st.markdown("#### ☀️ Modelo solar")

        st.image(
            "assets/shap_importancia_solar.png",
            use_container_width=True
        )

        st.markdown(
            """
            **Hallazgo —** La generación solar del día anterior (`solar_lag_24h`)
            presenta el mayor impacto medio sobre las predicciones, seguida por
            las variables de irradiancia solar CAMS de distintas localizaciones.
            """
        )

    with col_shap_eolica:
        st.markdown("#### 🌬️ Modelo eólico")

        st.image(
            "assets/shap_importancia_eolica.png",
            use_container_width=True
        )

        st.markdown(
            """
            **Hallazgo —** La velocidad del viento a 100 m y sus transformaciones
            no lineales concentran gran parte de la influencia sobre las
            predicciones, especialmente en Valladolid y Zaragoza. También
            intervienen variables temporales y la generación eólica del día anterior.
            """
        )

   
    # Red metereológica


    st.divider()

    st.subheader("🗺️ Red meteorológica")

    st.write(
        "Los modelos de RenewAI estiman la generación solar y eólica agregada "
        "de España utilizando información meteorológica procedente de 10 "
        "localizaciones distribuidas por el territorio."
    )

  
  
    # Localizaciones
 

    localizaciones_mapa = {
        "Valencia": {
            "codigo": "valencia",
            "lat": 39.4699,
            "lon": -0.3763
        },
        "Cádiz": {
            "codigo": "cadiz",
            "lat": 36.5271,
            "lon": -6.2886
        },
        "Madrid": {
            "codigo": "madrid",
            "lat": 40.4168,
            "lon": -3.7038
        },
        "Bilbao": {
            "codigo": "bilbao",
            "lat": 43.2630,
            "lon": -2.9350
        },
        "Cáceres": {
            "codigo": "caceres",
            "lat": 39.4753,
            "lon": -6.3724
        },
        "Tenerife": {
            "codigo": "tenerife",
            "lat": 28.2916,
            "lon": -16.6291
        },
        "Zaragoza": {
            "codigo": "zaragoza",
            "lat": 41.6488,
            "lon": -0.8891
        },
        "A Coruña": {
            "codigo": "a_coruna",
            "lat": 43.3623,
            "lon": -8.4115
        },
        "Pamplona": {
            "codigo": "pamplona",
            "lat": 42.8125,
            "lon": -1.6458
        },
        "Valladolid": {
            "codigo": "valladolid",
            "lat": 41.6523,
            "lon": -4.7245
        }
    }

  
    # Selección del modelo

    tecnologia_red = st.radio(
        "Modelo",
        ["☀️ Solar", "🌬️ Eólica"],
        horizontal=True,
        key="tecnologia_red"
    )

    if tecnologia_red == "☀️ Solar":
        variables_red = list(variables_solar)
        nombre_modelo_red = "solar"
    else:
        variables_red = list(variables_eolica)
        nombre_modelo_red = "eólico"

  
    # Variables retenidas por localización
  

    variables_por_localizacion = {}

    for nombre_loc, info_loc in localizaciones_mapa.items():

        codigo_loc = info_loc["codigo"]

        variables_loc = [
            variable
            for variable in variables_red
            if variable.startswith(f"{codigo_loc}_")
        ]

        variables_por_localizacion[
            nombre_loc
        ] = variables_loc

    localizaciones_retenidas = [
        nombre_loc
        for nombre_loc, variables_loc
        in variables_por_localizacion.items()
        if len(variables_loc) > 0
    ]

   
    # Crear mapa
   
    fig_red = go.Figure()

    for nombre_loc, info_loc in localizaciones_mapa.items():

        retenida = (
            len(
                variables_por_localizacion[
                    nombre_loc
                ]
            ) > 0
        )

        color = (
            "#2E7D32"
            if retenida
            else "#C7C7C7"
        )

        tamaño = (
            11
            if retenida
            else 8
        )

        if retenida:

            estado = (
                f"{len(variables_por_localizacion[nombre_loc])} "
                f"variable(s) en el modelo {nombre_modelo_red}"
            )

        else:

            estado = (
                f"Sin variables en el modelo {nombre_modelo_red}"
            )

        fig_red.add_trace(
            go.Scattergeo(
                lon=[
                    info_loc["lon"]
                ],
                lat=[
                    info_loc["lat"]
                ],
                text=[
                    nombre_loc
                ],
                customdata=[
                    f"<b>{nombre_loc}</b><br>{estado}"
                ],
                hovertemplate=(
                    "%{customdata}"
                    "<extra></extra>"
                ),
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=tamaño,
                    color=color,
                    line=dict(
                        width=1,
                        color="white"
                    )
                ),
                textfont=dict(
                    size=11,
                    color="#6B7280"
                ),
                showlegend=False
            )
        )

    fig_red.update_geos(
        projection_type="mercator",

        showland=True,
        landcolor="#F4F6F4",

        showocean=True,
        oceancolor="#FFFFFF",

        showcountries=True,
        countrycolor="#D5DDD6",

        showcoastlines=True,
        coastlinecolor="#BAC5BB",

        showlakes=False,

        bgcolor="rgba(0,0,0,0)",

        lonaxis=dict(
            range=[
                -18.8,
                4.0
            ]
        ),

        lataxis=dict(
            range=[
                27.0,
                44.5
            ]
        )
    )

    fig_red.update_layout(
        height=430,

        margin=dict(
            l=0,
            r=0,
            t=5,
            b=0
        ),

        showlegend=False,

        paper_bgcolor="rgba(0,0,0,0)",

        geo_bgcolor="rgba(0,0,0,0)"
    )


    # Mapa + interpretación

    col_mapa_red, col_interpretacion_red = st.columns(
        [1.45, 1],
        gap="large"
    )

  
    # Mapa

    with col_mapa_red:

        st.markdown(
            "#### Localizaciones utilizadas"
        )

        st.plotly_chart(
            fig_red,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "staticPlot": True
            }
        )

        st.caption(
            "🟢 Variables retenidas · "
            "⚪ Sin variables retenidas"
        )
        

    # Interpretación

    with col_interpretacion_red:

        st.markdown(
            "#### Lectura del modelo"
        )

        st.metric(
            "Localizaciones representadas",
            f"{len(localizaciones_retenidas)}/10"
        )

        st.metric(
            "Variables del modelo final",
            len(variables_red)
        )

        if tecnologia_red == "☀️ Solar":

            st.html(
                """
                <div style="
                    background-color:#F4FAF5;
                    border-left:4px solid #2E7D32;
                    border-radius:10px;
                    padding:15px 17px;
                    margin-top:8px;
                    line-height:1.65;
                    font-size:14px;
                    color:#4F5B52;
                ">
                    <b>☀️ Modelo solar</b><br><br>

                    El modelo final combina información meteorológica
                    de distintas zonas de España con la generación
                    solar del día anterior.<br><br>

                    Entre las variables meteorológicas con mayor
                    influencia destacan las relacionadas con
                    <b>CAMS GHI</b>, especialmente en
                    <b>Cáceres, Cádiz y Madrid</b>.<br><br>

                    La variable <b>solar_lag_24h</b> también tiene
                    una importancia elevada en la predicción.
                </div>
                """
            )

        else:

            st.html(
                """
                <div style="
                    background-color:#F4FAF5;
                    border-left:4px solid #2E7D32;
                    border-radius:10px;
                    padding:15px 17px;
                    margin-top:8px;
                    line-height:1.65;
                    font-size:14px;
                    color:#4F5B52;
                ">
                    <b>🌬️ Modelo eólico</b><br><br>

                    El modelo final utiliza información meteorológica
                    procedente de las <b>10 localizaciones</b>
                    analizadas.<br><br>

                    Predominan las variables de
                    <b>velocidad del viento a 100 m</b> y sus
                    transformaciones no lineales, con especial
                    relevancia de <b>Valladolid y Zaragoza</b>.<br><br>

                    También intervienen variables temporales y la
                    <b>generación eólica del día anterior</b>.
                </div>
                """
            )

    st.caption(
        "La presencia de una localización en el modelo no representa su "
        "contribución física directa a la generación eléctrica nacional; "
        "sus condiciones meteorológicas actúan como predictores de la "
        "generación agregada de España."
    )



# ALCANCE E IMPACTO DE NEGOCIO

elif pagina == "💼 Impacto y alcance":

    st.divider()

    
    # Cabecera
   

    st.caption("EL RETO")

    st.markdown(
        """
        ## La generación renovable es variable.  
        ## Anticiparla mejora la toma de decisiones.
        """
    )

    st.write(
        "RenewAI utiliza modelos de Machine Learning para estimar la generación "
        "solar y eólica en España, contextualizar su contribución a la demanda "
        "eléctrica y explorar escenarios alternativos."
    )

    st.write("")

 
    # Impacto y utilidad
   

    st.caption("IMPACTO Y UTILIDAD")

    col_imp1, col_imp2, col_imp3, col_imp4 = st.columns(4)

    with col_imp1:
        st.html(
            """
            <div style="
                border-top: 3px solid #2E7D32;
                padding-top: 14px;
            ">
                <div style="
                    font-size: 42px;
                    font-weight: 700;
                    color: #2E7D32;
                    line-height: 1.1;
                ">
                    94.8%
                </div>

                <div style="
                    font-size: 13px;
                    color: #7A7A7A;
                    margin-top: 4px;
                ">
                    R² solar
                </div>

                <div style="
                    font-weight: 600;
                    margin-top: 14px;
                ">
                    Alta capacidad predictiva
                </div>

                <div style="
                    font-size: 13px;
                    color: #6B6B6B;
                    margin-top: 5px;
                    line-height: 1.5;
                ">
                    El modelo explica gran parte de la variabilidad observada
                    en la generación solar.
                </div>
            </div>
            """
        )

    with col_imp2:
        st.html(
            """
            <div style="
                border-top: 3px solid #2E7D32;
                padding-top: 14px;
            ">
                <div style="
                    font-size: 42px;
                    font-weight: 700;
                    color: #2E7D32;
                    line-height: 1.1;
                ">
                    83.1%
                </div>

                <div style="
                    font-size: 13px;
                    color: #7A7A7A;
                    margin-top: 4px;
                ">
                    R² eólica
                </div>

                <div style="
                    font-weight: 600;
                    margin-top: 14px;
                ">
                    Predicción eólica robusta
                </div>

                <div style="
                    font-size: 13px;
                    color: #6B6B6B;
                    margin-top: 5px;
                    line-height: 1.5;
                ">
                    El modelo captura una proporción elevada de la variabilidad
                    de la generación eólica.
                </div>
            </div>
            """
        )

    with col_imp3:
        st.html(
            """
            <div style="
                border-top: 3px solid #2E7D32;
                padding-top: 14px;
            ">
                <div style="
                    font-size: 42px;
                    font-weight: 700;
                    color: #2E7D32;
                    line-height: 1.1;
                ">
                    6.55 p.p.
                </div>

                <div style="
                    font-size: 13px;
                    color: #7A7A7A;
                    margin-top: 4px;
                ">
                    MAE cobertura
                </div>

                <div style="
                    font-weight: 600;
                    margin-top: 14px;
                ">
                    Estimación de cobertura
                </div>

                <div style="
                    font-size: 13px;
                    color: #6B6B6B;
                    margin-top: 5px;
                    line-height: 1.5;
                ">
                    La cobertura estimada se desvía de la cobertura real en
                    6.55 puntos porcentuales de media.
                </div>
            </div>
            """
        )

    
    st.write("")


    # Valor para la toma de decisiones


    st.caption("VALOR PARA LA TOMA DE DECISIONES")

    col_val1, col_val2, col_val3 = st.columns(3)

    with col_val1:
        st.markdown("### ⚡ Anticipación energética")
        st.write(
            "Estimación horaria de generación solar y eólica a partir de "
            "variables meteorológicas y generación reciente."
        )

    with col_val2:
        st.markdown("### 🔬 Análisis de escenarios")
        st.write(
            "Evaluación de cómo podrían variar las predicciones ante cambios "
            "hipotéticos en irradiancia, viento y demanda."
        )

    with col_val3:
        st.markdown("### 🔋 Contextualización del sistema")
        st.write(
            "Análisis conjunto de generación renovable, demanda eléctrica, "
            "cobertura y precio mayorista observado."
        )

    st.divider()

   
    # Alcance y desarrollo futuro

    st.caption("ALCANCE Y DESARROLLO FUTURO")

    col_fut1, col_fut2, col_fut3 = st.columns(3)

    with col_fut1:
        st.markdown("### 🔄 Operación futura")
        st.write(
            "Una implementación en tiempo real requeriría automatizar la obtención "
            "de datos actualizados y previsiones meteorológicas mediante las APIs "
            "utilizadas en el proyecto."
        )

    with col_fut2:
        st.markdown("### 🔋 Previsión de demanda")
        st.write(
            "Para estimar la cobertura renovable a futuro sería necesario incorporar "
            "también una previsión de demanda eléctrica."
        )

    with col_fut3:
        st.markdown("### 💶 Mercado eléctrico")
        st.write(
            "El precio se utiliza actualmente como contexto histórico y no como "
            "predicción. Una extensión futura podría incorporar un modelo específico "
            "de precio eléctrico."
        )


        


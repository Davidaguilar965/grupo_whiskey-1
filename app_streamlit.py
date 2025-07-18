import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
import io

# Configuración de la página
st.set_page_config(
    page_title="Análisis Geotécnico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    h1 {
        color: #1a5276;
        border-bottom: 2px solid #1a5276;
        padding-bottom: 10px;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stDownloadButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stFileUploader>div>div>div>div {
        border: 2px dashed #2980b9;
        border-radius: 5px;
    }
    .css-1aumxhk {
        background-color: #d4e6f1;
        border-radius: 5px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar con información
with st.sidebar:
    st.title("📊 Configuración")
    st.markdown("""
    *Instrucciones:*
    1. Sube tu archivo CSV con datos de desplazamiento y precipitación
    2. Los datos deben incluir columnas para fecha, desplazamiento y precipitación
    3. La aplicación generará gráficos y análisis automáticos
    """)
    
    st.markdown("---")
    st.markdown("*Acerca de esta aplicación:*")
    st.markdown("""
    Esta herramienta analiza la relación entre desplazamiento del terreno 
    y precipitaciones, generando visualizaciones interactivas y reportes estadísticos.
    """)
    
    st.markdown("---")
    st.markdown("Desarrollado por [Tu Nombre]")
    st.markdown("Versión 1.0 | Julio 2023")

# Contenido principal
st.title("📈 Análisis de Desplazamiento y Precipitación")
st.markdown("Visualización dual de datos geotécnicos con análisis estadístico integrado")

# División en pestañas
tab1, tab2, tab3 = st.tabs(["📤 Carga de datos", "📊 Visualización", "📝 Reporte"])

with tab1:
    st.header("Carga de datos")
    st.markdown("""
    Sube tu archivo CSV con los siguientes requisitos:
    - Columna 'fecha' en formato día/mes/año o mes/día/año
    - Columna 'rainfall(mm)' con valores de precipitación
    - Columnas numéricas para desplazamiento (nombres como 1, 2, 3, etc.)
    """)
    
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo CSV",
        type=["csv"],
        key="file_uploader",
        help="El archivo debe contener datos de desplazamiento y precipitación"
    )

with tab2:
    st.header("Visualización de datos")
    
    if not uploaded_file:
        st.warning("Por favor sube un archivo CSV en la pestaña 'Carga de datos'")
    else:
        # Procesamiento de datos
        @st.cache_data
        def load_data(uploaded_file):
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=',',
                    encoding='utf-8-sig',
                    engine='python'
                )
                
                # Limpieza de datos
                df.columns = df.columns.str.strip()
                df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
                
                # Convertir fechas
                df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['fecha'])
                df.sort_values('fecha', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                return df
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
                return None
        
        df = load_data(uploaded_file)
        
        if df is not None:
            # Verificación de columnas
            precip_col = 'rainfall(mm)'
            if precip_col not in df.columns:
                st.error(f"No se encontró la columna '{precip_col}' en el CSV.")
            else:
                disp_cols = [c for c in df.columns if c not in ['fecha', precip_col]]
                disp_cols = sorted(disp_cols, key=lambda x: int(x))
                
                if not disp_cols:
                    st.error("No se encontraron columnas de desplazamiento.")
                else:
                    # Mostrar vista previa de datos
                    with st.expander("📋 Vista previa de los datos (primeras 10 filas)"):
                        st.dataframe(df.head(10))
                    
                    # Configuración del gráfico
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Configuración del gráfico")
                        marker_size = st.slider("Tamaño de marcadores", 10, 100, 40)
                        line_width = st.slider("Ancho de línea", 1, 10, 2)
                        date_format = st.selectbox(
                            "Formato de fecha",
                            ["%b %Y", "%d/%m/%Y", "%m/%d/%Y"],
                            index=0
                        )
                    
                    with col2:
                        st.subheader("Selección de datos")
                        show_annotations = st.checkbox("Mostrar anotaciones de valores", True)
                        selected_cols = st.multiselect(
                            "Seleccionar puntos de desplazamiento a mostrar",
                            disp_cols,
                            default=disp_cols
                        )
                    
                    # Crear gráfico
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax2 = ax.twinx()
                    
                    # Precipitación
                    y_precip = df[precip_col]
                    line, = ax2.plot(
                        df['fecha'], 
                        y_precip, 
                        label='Precipitación (mm)', 
                        linewidth=line_width, 
                        marker='o',
                        color='#3498db',
                        markersize=8
                    )
                    
                    if show_annotations:
                        for xi, yi in zip(df['fecha'], y_precip):
                            if pd.notna(yi):
                                ax2.annotate(
                                    f"{yi:.1f}", 
                                    (xi, yi), 
                                    textcoords='offset points', 
                                    xytext=(0,5), 
                                    ha='center', 
                                    fontsize=8,
                                    color='#3498db'
                                )
                    
                    # Desplazamientos
                    colors = plt.cm.viridis_r([i/len(selected_cols) for i in range(len(selected_cols))])
                    for col, color in zip(selected_cols, colors):
                        ax.scatter(
                            df['fecha'], 
                            df[col], 
                            s=marker_size, 
                            label=f"Punto {col}",
                            color=color,
                            edgecolors='white',
                            linewidth=0.5
                        )
                    
                    # Estilo del gráfico
                    ax.set_xlabel('Fecha', fontsize=12)
                    ax.set_ylabel('Desplazamiento (cm)', fontsize=12)
                    ax2.set_ylabel('Precipitación (mm)', fontsize=12)
                    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
                    fig.autofmt_xdate(rotation=45)
                    
                    # Leyenda unificada
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax.legend(
                        h1+h2, 
                        l1+l2, 
                        loc='upper left', 
                        fontsize=10, 
                        ncol=2,
                        framealpha=1
                    )
                    
                    # Título y mostrar gráfico
                    plt.title("Relación entre Desplazamiento y Precipitación", pad=20, fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Botón para descargar el gráfico
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300)
                    st.download_button(
                        label="Descargar gráfico",
                        data=buf.getvalue(),
                        file_name="grafico_desplazamiento_precipitacion.png",
                        mime="image/png"
                    )

with tab3:
    st.header("Reporte de análisis")
    
    if not uploaded_file:
        st.warning("Por favor sube un archivo CSV en la pestaña 'Carga de datos'")
    elif df is not None and precip_col in df.columns and disp_cols:
        # Sección 1: Fecha con mayor tasa de desplazamiento
        st.subheader("📈 Tasa de desplazamiento")
        
        deltas = df[disp_cols].diff()
        delta_t = df['fecha'].diff().dt.total_seconds() / (24*3600)
        tasa = deltas.div(delta_t, axis=0)
        tasa_prom = tasa.mean(axis=1)
        idx_max = tasa_prom.idxmax()
        fecha_tasa = df.loc[idx_max, 'fecha']
        valor_tasa = tasa_prom.max()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Fecha con mayor tasa de desplazamiento",
                value=fecha_tasa.strftime('%d/%m/%Y'),
                help="Fecha con la mayor tasa media diaria de desplazamiento"
            )
        with col2:
            st.metric(
                label="Tasa máxima registrada",
                value=f"{valor_tasa:.3f} cm/día",
                help="Tasa media diaria de desplazamiento máxima"
            )
        
        # Gráfico de tasa de desplazamiento
        fig_tasa, ax_tasa = plt.subplots(figsize=(10, 4))
        ax_tasa.plot(df['fecha'], tasa_prom, label='Tasa de desplazamiento', color='#e74c3c')
        ax_tasa.scatter(fecha_tasa, valor_tasa, color='red', s=100, zorder=5, 
                       label='Máxima tasa')
        ax_tasa.annotate(f'Máximo: {valor_tasa:.3f} cm/día', 
                        (fecha_tasa, valor_tasa),
                        textcoords='offset points', 
                        xytext=(10,10), 
                        ha='left',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        ax_tasa.set_title("Evolución de la tasa media de desplazamiento")
        ax_tasa.set_xlabel("Fecha")
        ax_tasa.set_ylabel("Tasa (cm/día)")
        ax_tasa.grid(True, linestyle='--', alpha=0.7)
        ax_tasa.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_tasa.legend()
        fig_tasa.autofmt_xdate(rotation=45)
        st.pyplot(fig_tasa)
        
        # Sección 2: Estadísticas descriptivas
        st.subheader("📋 Estadísticas descriptivas")
        
        # Estadísticas para desplazamiento
        st.markdown("*Estadísticas de desplazamiento por punto de medición*")
        st.dataframe(df[disp_cols].describe().T.style.format("{:.2f}"))
        
        # Estadísticas para precipitación
        st.markdown("*Estadísticas de precipitación*")
        st.dataframe(df[[precip_col]].describe().T.style.format("{:.2f}"))
        
        # Sección 3: Correlación entre variables
        st.subheader("🔍 Correlación entre desplazamiento y precipitación")
        
        # Calcular correlación para cada punto
        correlaciones = []
        for col in disp_cols:
            corr = df[col].corr(df[precip_col])
            correlaciones.append(corr)
        
        # Mostrar resultados en columnas
        cols = st.columns(3)
        for i, (col, corr) in enumerate(zip(disp_cols, correlaciones)):
            with cols[i % 3]:
                color = "green" if abs(corr) > 0.5 else "orange" if abs(corr) > 0.3 else "red"
                st.metric(
                    label=f"Punto {col}",
                    value=f"{corr:.2f}",
                    help=f"Correlación entre desplazamiento en punto {col} y precipitación"
                )
                st.markdown(f"<div style='height:5px; background-color:{color}; width:{abs(corr)*100}%;'></div>", 
                          unsafe_allow_html=True)
        
        st.caption("Nota: Correlación varía de -1 (inversa perfecta) a 1 (directa perfecta). Valores cercanos a 0 indican no correlación.")
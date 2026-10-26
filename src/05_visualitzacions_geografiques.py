# %% ============================================================================
# PART 2: VISUALITZACIONS GEOGRÀFIQUES
# ==============================================================================

import pandas as pd
import folium
from folium import plugins
from folium.features import DivIcon
import geopandas as gpd
from branca.colormap import LinearColormap
import os

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

# Carregar dades
df = pd.read_csv('./data/supply_chain_cleaned.csv')
date_cols = ['order_date_dateorders', 'shipping_date_dateorders']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['order_year'] = df['order_date_dateorders'].dt.year
df['delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']

# ------------------------------------------------------------------------------
# CORRECCIÓ DE NOMS DE PAÏSOS (CASTELLÀ -> ANGLÈS STANDARD)
# ------------------------------------------------------------------------------
# Diccionari de traducció per coincidir amb el GeoJSON (Natural Earth)
translation_map = {
    'Estados Unidos': 'United States of America',
    'Francia': 'France',
    'Alemania': 'Germany',
    'Reino Unido': 'United Kingdom',
    'España': 'Spain',
    'Italia': 'Italy',
    'Japón': 'Japan',
    'China': 'China',  # Igual
    'India': 'India',  # Igual
    'Brasil': 'Brazil',
    'México': 'Mexico',
    'Canadá': 'Canada',
    'Australia': 'Australia', # Igual
    'Turquía': 'Turkey',
    'Países Bajos': 'Netherlands',
    'Suecia': 'Sweden',
    'Bélgica': 'Belgium',
    'Suiza': 'Switzerland',
    'Austria': 'Austria', # Igual
    'Polonia': 'Poland',
    'Noruega': 'Norway',
    'Dinamarca': 'Denmark',
    'Finlandia': 'Finland',
    'Rusia': 'Russia',
    'Corea del Sur': 'South Korea',
    'Singapur': 'Singapore',
    'Irlanda': 'Ireland',
    'Portugal': 'Portugal', # Igual
    'Grecia': 'Greece',
    'República Checa': 'Czech Republic',
    'Rumania': 'Romania',
    'Vietnam': 'Vietnam', # Igual
    'Filipinas': 'Philippines',
    'Tailandia': 'Thailand',
    'Egipto': 'Egypt',
    'SudAfrica': 'South Africa',
    'Arabia Saudí': 'Saudi Arabia',
    'Colombia': 'Colombia', # Igual
    'Argentina': 'Argentina', # Igual
    'Perú': 'Peru',
    'Marruecos': 'Morocco',
    'Ucrania': 'Ukraine',
    'Hungría': 'Hungary',
    'Emiratos Árabes Unidos': 'United Arab Emirates',
    'Nueva Zelanda': 'New Zealand',
    'Israel': 'Israel', # Igual
    'Irán': 'Iran',
    'Pakistán': 'Pakistan',
    'Kazajistán': 'Kazakhstan',
    'Argelia': 'Algeria',
    'Nigeria': 'Nigeria', # Igual
    'Kenia': 'Kenya',
    'Etiopía': 'Ethiopia',
    'Ghana': 'Ghana', # Igual
    'Tanzania': 'Tanzania', # Igual
    'Angola': 'Angola', # Igual
    'República Dominicana': 'Dominican Republic',
    'Guatemala': 'Guatemala', # Igual
    'El Salvador': 'El Salvador', # Igual
    'Panamá': 'Panama',
    'Honduras': 'Honduras', # Igual
    'Venezuela': 'Venezuela', # Igual
    'Ecuador': 'Ecuador', # Igual
    'Bolivia': 'Bolivia', # Igual
    'Chile': 'Chile', # Igual
    'Uruguay': 'Uruguay', # Igual
    'Paraguay': 'Paraguay' # Igual
}

# Aplicar la traducció. Si no troba la traducció, deixa el nom original.
print("Traduint noms de països al format GeoJSON...")
df['order_country_en'] = df['order_country'].map(translation_map).fillna(df['order_country'])

# Comprovació ràpida
print("   Exemple: 'Estados Unidos' ara és ->", 
      df[df['order_country'] == 'Estados Unidos']['order_country_en'].iloc[0])

print_section("PART 2: VISUALITZACIONS GEOGRÀFIQUES")

MAPS_DIR = 'figures/maps'
os.makedirs(MAPS_DIR, exist_ok=True)

# %% PREPARACIÓ DE DADES
print("\nPREPARANT DADES GEOGRÀFIQUES...")

try:
    countries_geo = gpd.read_file('./data/countries.geojson')
    print(f"GeoJSON països carregat: {len(countries_geo)} països")
except:
    print("countries.geojson no trobat")
    countries_geo = None

country_stats = df.groupby('order_country_en').agg({
    'order_id': 'count',
    'sales': 'sum',
    'benefit_per_order': 'mean',
    'late_delivery_risk': 'mean',
    'delay_days': 'mean'
}).round(2)
country_stats.columns = ['total_orders', 'total_sales', 'avg_benefit', 
                         'late_delivery_rate', 'avg_delay']
country_stats['late_delivery_rate'] = country_stats['late_delivery_rate'] * 100

seller_locations = df.groupby(['customer_city', 'latitude_src', 'longitude_src']).agg({
    'order_id': 'count',
    'sales': 'sum',
    'benefit_per_order': 'mean'
}).reset_index()
seller_locations.columns = ['city', 'lat', 'lon', 'orders', 'sales', 'benefit']
seller_locations = seller_locations[
    (seller_locations['lat'].notna()) & 
    (seller_locations['lon'].notna()) &
    (seller_locations['lat'] != 0) &
    (seller_locations['lon'] != 0) &
    (seller_locations['lat'].abs() <= 90) &
    (seller_locations['lon'].abs() <= 180)
]

print(f"{len(seller_locations)} ubicacions de venedors identificades")
# %% ============================================================================
# VISUALITZACIÓ 0: VISIÓ ESTRATÈGICA PER CONTINENT (EXECITIU)
# ==============================================================================
"""
ESTUDI 0: Quadre de Comandament Continental
Objectiu: Visió "Big Picture". Analitzar volum de vendes i marge de benefici
          agregat per grans blocs continentals abans d'entrar al detall.
"""

print("\n" + "="*80)
print("VISUALITZACIÓ 0: ESTRATÈGIA CONTINENTAL")
print("="*80)

try:
    # 1. Carregar GeoJSON
    continents_geo = gpd.read_file('./data/continents.geojson')
    
    # 2. Mapejar els nostres mercats ('Market') als noms del GeoJSON
    # Nota: Això és una aproximació de negoci per agrupar ràpidament
    market_map = {
        'Europe': 'europe',          
        'Africa': 'africa',          
        'USCA': 'northAmerica',     
        'LATAM': 'southAmerica',     
        'Pacific Asia': 'asia'       
    }
    
    df['continent_mapped'] = df['market'].map(market_map)
    
    # 3. Calcular KPIs financers per Continent
    cont_stats = df.groupby('continent_mapped').agg({
        'sales': 'sum',
        'benefit_per_order': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    # Calcular Marge (%) i Tiquet Mitjà
    cont_stats['margin_pct'] = (cont_stats['benefit_per_order'] / cont_stats['sales'] * 100).round(1)
    cont_stats['avg_ticket'] = (cont_stats['sales'] / cont_stats['order_id']).round(0)
    cont_stats['sales_millions'] = (cont_stats['sales'] / 1000000).round(2)
    
    # 4. Unir dades amb el GeoJSON
    # Assegurem que el GeoJSON tingui la columna 'continent' (o similar)
    geo_col = 'continent' 
    
    continents_merged = continents_geo.merge(
        cont_stats,
        left_on=geo_col,
        right_on='continent_mapped',
        how='left'
    )
    values_to_fill = ['sales', 'sales_millions', 'margin_pct', 'avg_ticket', 'benefit_per_order', 'order_id']
    for col in values_to_fill:
        if col in continents_merged.columns:
            continents_merged[col] = continents_merged[col].fillna(0)
    # 5. Crear Mapa (Estil Executiu Net)
    m0 = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')
    
    # Choropleth: Color segons Vendes Totals
    folium.Choropleth(
        geo_data=continents_merged,
        data=continents_merged,
        columns=[geo_col, 'sales'],
        key_on=f'feature.properties.{geo_col}',
        fill_color='GnBu',  # Verd-Blau (Professional)
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Vendes Totals ($)',
        highlight=True
    ).add_to(m0)
    
    # 6. Tooltip informatiu (La clau de la utilitat)
    # Afegim una capa transparent a sobre només per als popups/tooltips
    style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1}
    
    folium.GeoJson(
        continents_merged,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[geo_col, 'sales_millions', 'margin_pct', 'avg_ticket'],
            aliases=['Continent:', 'Vendes (M$):', 'Marge Real (%):', 'Tiquet Mitjà ($):'],
            style="font-family: Arial; font-size: 14px; padding: 10px; background-color: white; border: 2px solid black;",
            localize=True
        )
    ).add_to(m0)

    # Títol flotant "Executive"
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 100px; 
                background-color: white; border-left:5px solid #2980b9; z-index:9999; 
                font-family: Arial; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
    <h4 style="margin:0; color:#2c3e50;">Visió Global de Negoci</h4>
    <small style="color:#7f8c8d;">Resum executiu per continent</small><br>
    <hr style="margin:5px 0;">
    <i style="font-size:11px;">Passa el ratolí per veure Marge i Vendes</i>
    </div>
    '''
    m0.get_root().html.add_child(folium.Element(title_html))

    m0.save(f'{MAPS_DIR}/map0_strategy_continents.html')
    print("Mapa 0 guardat: map0_strategy_continents.html (Dashboard Executiu)")

except Exception as e:
    print(f"No s'ha pogut generar el Mapa 0: {e}")


# %% MAPA 1: CHOROPLETH - RISC LOGÍSTIC
print("\n" + "="*80)
print("MAPA 1: RISC LOGÍSTIC (CHOROPLETH)")
print("="*80)

try:
    m1 = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
    
    if countries_geo is not None:
        countries_geo_merged = countries_geo.merge(
            country_stats, left_on='ADMIN', right_index=True, how='left'
        )
        
        colormap = LinearColormap(
            colors=['#2ECC71', '#F39C12', '#E74C3C'],
            vmin=0, vmax=100, caption='% Comandes amb Retard'
        )
        
        def style_function(feature):
            rate = feature['properties'].get('late_delivery_rate', None)
            if rate is None:
                return {'fillColor': '#d3d3d3', 'color': 'black', 
                       'weight': 0.5, 'fillOpacity': 0.3}
            return {'fillColor': colormap(rate), 'color': 'black', 
                   'weight': 1, 'fillOpacity': 0.7}
        
        folium.GeoJson(
            countries_geo_merged,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['ADMIN', 'late_delivery_rate', 'total_orders'],
                aliases=['País:', '% Retards:', 'Comandes:']
            )
        ).add_to(m1)
        
        colormap.add_to(m1)
        m1.save(f'{MAPS_DIR}/map1_choropleth_risk.html')
        print("Mapa 1 guardat")
    else:
        print("Mapa 1 no generat (GeoJSON no disponible)")
except Exception as e:
    print(f"Error Mapa 1: {e}")     
# %% MAPA 2: BUBBLE MAP - HUBS COMERCIALS 
print("\n" + "="*80)
print("MAPA 2: HUBS COMERCIALS (BUBBLE MAP)")
print("="*80)

try:
    # Determinar centre del mapa basant-se en les dades reals
    center_lat = seller_locations['lat'].median()
    center_lon = seller_locations['lon'].median()
    
    m2 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='CartoDB positron'
    )
    
    
    
    # Grup 1: Markers individuals (per veure tots)
    individual_group = folium.FeatureGroup(name='🔵 Ubicacions Individuals', show=True)
    
    # Grup 2: Clusters automàtics
    cluster_group = plugins.MarkerCluster(name='🔴 Vista Agrupada (Clusters)')
    
    def get_color(benefit):
        if benefit < 0: return 'red'
        elif benefit < 20: return 'orange'
        elif benefit < 40: return 'lightgreen'
        else: return 'darkgreen'
    
    sales_min, sales_max = seller_locations['sales'].min(), seller_locations['sales'].max()
    def get_radius(sales):
        return 5 + ((sales - sales_min) / (sales_max - sales_min) * 20)
    
    for idx, row in seller_locations.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4>{row['city']}</h4>
            <b>Comandes:</b> {row['orders']:,}<br>
            <b>Vendes:</b> ${row['sales']:,.0f}<br>
            <b>Benefici:</b> ${row['benefit']:.2f}
        </div>
        """
        
        # Per al grup individual: CircleMarker
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=get_radius(row['sales']),
            popup=folium.Popup(popup_html, max_width=250),
            color='black',
            fillColor=get_color(row['benefit']),
            fillOpacity=0.7,
            weight=1
        ).add_to(individual_group)
        
        # Per al cluster: Marker normal amb icon personalitzat
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=get_color(row['benefit']).replace('lightgreen', 'green')
                            .replace('darkgreen', 'green'), icon='info-sign')
        ).add_to(cluster_group)
    
    individual_group.add_to(m2)
    cluster_group.add_to(m2)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                padding: 10px;">
    <b>Llegenda</b><br>
    <i style="color:darkgreen">●</i> Benefici > $40<br>
    <i style="color:lightgreen">●</i> Benefici $20-40<br>
    <i style="color:orange">●</i> Benefici $0-20<br>
    <i style="color:red">●</i> Pèrdues<br>
    <hr>
    <small>Usa el control de capes per canviar vista</small>
    </div>
    '''
    m2.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position='topright').add_to(m2)
    
    m2.save(f'{MAPS_DIR}/map2_bubble_sellers.html')
    print("Mapa 2 guardat")
    print(f"   • Ubicacions: {len(seller_locations)}")
    
except Exception as e:
    print(f"Error Mapa 2: {e}")
    import traceback
    traceback.print_exc()

# %% MAPA 3: HEATMAP TEMPORAL
print("\n" + "="*80)
print("MAPA 3: MAPA DE CALOR TEMPORAL")
print("="*80)

try:
    dest_points = df[['latitude_dest', 'longitude_dest', 'order_date_dateorders', 'order_country_en']].copy()
    dest_points = dest_points[
        (dest_points['latitude_dest'].notna()) & 
        (dest_points['longitude_dest'].notna()) &
        (dest_points['latitude_dest'] != 0) &
        (dest_points['longitude_dest'] != 0)
    ]
    static_sample = dest_points
    if len(dest_points) > 30000:
        dest_points = dest_points.sample(30000, random_state=42)
    
    m3 = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB dark_matter')
    
    heat_data = [[row['latitude_dest'], row['longitude_dest']] 
                 for idx, row in static_sample.iterrows()]
    
    plugins.HeatMap(
        heat_data,
        name='Mapa de Calor Estàtic',
        min_opacity=0.2,
        radius=12,
        blur=15,
        gradient={0.0: 'blue', 0.5: 'lime', 0.7: 'yellow', 1.0: 'red'}
    ).add_to(m3)

    dest_points['YYYY_MM'] = dest_points['order_date_dateorders'].dt.to_period('M')
    periods = sorted(dest_points['YYYY_MM'].unique().astype(str))

    heat_data_time = []
    time_index = []

    for period_srt in periods:
        period_data = dest_points[dest_points['YYYY_MM'] == period_srt]
        if len(period_data) > 2000:
            period_data = period_data.sample(2000, random_state=42)
        period_coords = [[row['latitude_dest'], row['longitude_dest']] 
                         for idx, row in period_data.iterrows()]
        heat_data_time.append(period_coords)
        time_index.append(period_srt)
    
    hm_time = plugins.HeatMapWithTime(
        heat_data_time,
        index=time_index,
        name='Mapa de Calor Temporal',
        auto_play=True,
        max_opacity=0.8,
        radius=15,
        display_index=True
    )
    hm_time.add_to(m3)

    top_destinations = df['order_country_en'].value_counts().head(10)
    feature_group_top = folium.FeatureGroup(name='Destinacions Top 10')

    for country in top_destinations.index:
        country_data = df[df['order_country_en'] == country]
        avg_lat = country_data['latitude_dest'].mean()
        avg_lon = country_data['longitude_dest'].mean()
        count = top_destinations[country]

        if pd.notna(avg_lat):
            folium.Marker(
                location=[avg_lat, avg_lon],
                icon=DivIcon(html=f"""
                    <div style="font-size: 10px; color: white; 
                               text-shadow: 1px 1px 2px black; font-weight: bold;
                               white-space: nowrap;">
                         {country} ({count:,})
                    </div>
                """)
            ).add_to(feature_group_top)
    feature_group_top.add_to(m3)
    folium.LayerControl(position='topright').add_to(m3)

    m3.save(f'{MAPS_DIR}/map3_heatmap_destinations.html')
    print("Mapa 3 guardat")
    print(f"   • Punts: {len(dest_points):,}")
    
except Exception as e:
    print(f"Error Mapa 3: {e}")
# %% MAPA 4: FLOW MAP - RUTES
print("\n" + "="*80)
print("MAPA 4: RUTES COMERCIALS (FLOW MAP)")
print("="*80)

try:
    routes_data = df[['latitude_src', 'longitude_src', 
                      'latitude_dest', 'longitude_dest',
                      'benefit_per_order', 'sales', 
                      'customer_city', 'order_city']].copy()
    
    routes_data = routes_data[
        (routes_data['latitude_src'].notna()) & 
        (routes_data['longitude_src'].notna()) &
        (routes_data['latitude_dest'].notna()) & 
        (routes_data['longitude_dest'].notna()) &
        (routes_data['latitude_src'] != 0) &
        (routes_data['latitude_dest'] != 0)
    ]
    
    routes_agg = routes_data.groupby(['customer_city', 'order_city']).agg({
        'latitude_src': 'first',
        'longitude_src': 'first',
        'latitude_dest': 'first',
        'longitude_dest': 'first',
        'benefit_per_order': 'mean',
        'sales': 'sum'
    }).reset_index()
    
    routes_agg = routes_agg.nlargest(50, 'sales')  # Reduït a 50 per rendiment
    
    m4 = folium.Map(location=[39, -95], zoom_start=4, tiles='CartoDB positron')
    
    for idx, row in routes_agg.iterrows():
        if row['benefit_per_order'] < 0:
            color = 'red'
        elif row['benefit_per_order'] < 20:
            color = 'orange'
        else:
            color = 'green'
        
        weight = 1 + (row['sales'] / routes_agg['sales'].max() * 4)
        
        coords = [
            [row['latitude_src'], row['longitude_src']],
            [row['latitude_dest'], row['longitude_dest']]
        ]
        
        
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.6,
            popup=f"{row['customer_city']} → {row['order_city']}<br>${row['sales']:,.0f}"
        ).add_to(m4)
        
        # Marcador origen
        folium.CircleMarker(
            [row['latitude_src'], row['longitude_src']],
            radius=4, color='blue', fillColor='lightblue',
            fillOpacity=0.8, weight=1
        ).add_to(m4)
        
        # Marcador destí
        folium.CircleMarker(
            [row['latitude_dest'], row['longitude_dest']],
            radius=3, color='purple', fillColor='pink',
            fillOpacity=0.6, weight=1
        ).add_to(m4)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                padding: 10px;">
    <b>Llegenda de Rutes</b><br>
    <hr style="margin: 5px 0;">
    <i style="color:green">━━━</i> Rendibles (>$20)<br>
    <i style="color:orange">━━━</i> Baix Marge ($0-20)<br>
    <i style="color:red">━━━</i> Pèrdues<br>
    <hr style="margin: 5px 0;">
    <i style="color:blue">●</i> Origen | <i style="color:purple">●</i> Destí
    </div>
    '''
    m4.get_root().html.add_child(folium.Element(legend_html))
    
    m4.save(f'{MAPS_DIR}/map4_flow_routes.html')
    print("Mapa 4 guardat")
    print(f"   • Rutes: {len(routes_agg)}")
    
except Exception as e:
    print(f"Error Mapa 4: {e}")
# %% MAPA 5: CATEGORIES PER REGIÓ 
print("\n" + "="*80)
print("MAPA 5: PREFERÈNCIES PER REGIÓ")
print("="*80)

try:
    category_by_region = df.groupby(['order_region', 'category_name']).agg({
        'sales': 'sum'
    }).reset_index()
    
    top_cats_per_region = (category_by_region
                          .sort_values('sales', ascending=False)
                          .groupby('order_region')
                          .head(5))
    
    region_totals = category_by_region.groupby('order_region')['sales'].sum()
    top_cats_per_region['pct'] = top_cats_per_region.apply(
        lambda x: x['sales'] / region_totals[x['order_region']] * 100, axis=1
    )
    
    region_coords = df.groupby('order_region').agg({
        'latitude_dest': 'mean',
        'longitude_dest': 'mean',
        'order_id': 'count'
    }).reset_index()
    region_coords.columns = ['order_region', 'lat', 'lon', 'total_orders']
    
    # Filtrar regions amb coordenades vàlides
    region_coords = region_coords[
        (region_coords['lat'].notna()) & 
        (region_coords['lon'].notna()) &
        (region_coords['lat'] != 0)
    ]
    
    m5 = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
    
    top_categories = df['category_name'].value_counts().head(10).index
    category_colors = {
        cat: color for cat, color in zip(
            top_categories,
            ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
             '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
        )
    }
    
    for idx, region_row in region_coords.iterrows():
        region = region_row['order_region']
        lat = region_row['lat']
        lon = region_row['lon']
        
        region_cats = top_cats_per_region[top_cats_per_region['order_region'] == region]
        
        if len(region_cats) == 0:
            continue
        
        # MILLORA: Popup més compacte i funcional
        popup_html = f"""
        <div style="font-family: Arial; width: 260px; padding: 8px;">
            <h3 style="margin: 3px 0; color: #333;">{region}</h3>
            <hr style="margin: 5px 0;">
            <b>Top Categories:</b><br>
        """
        
        for _, cat_row in region_cats.iterrows():
            cat = cat_row['category_name']
            pct = cat_row['pct']
            color = category_colors.get(cat, '#888')
            
            popup_html += f"""
            <div style="margin: 3px 0;">
                <span style="color: {color}; font-size: 16px;">■</span> 
                <small>{cat[:25]}: {pct:.1f}%</small>
            </div>
            """
        
        total = region_totals[region]
        popup_html += f"<hr><b>Total:</b> ${total:,.0f}</div>"
        
        # Mida proporcional al volum
        radius = 8 + (region_row['total_orders'] / region_coords['total_orders'].max() * 15)
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{region} (click per detalls)",
            color='black',
            fillColor='#3388ff',
            fillOpacity=0.6,
            weight=2
        ).add_to(m5)
    
    legend_items = '<br>'.join([
        f'<span style="color:{category_colors.get(cat, "#ccc")}; font-size:16px;">■</span> {cat}'
        for cat in list(top_categories)[:6]
    ])
    
    legend_html = f'''
    <div style="position: fixed; top: 80px; right: 20px; width: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                padding: 8px; font-size:11px;">
    <b>Top Categories</b><br>
    <hr style="margin: 5px 0;">
    {legend_items}
    <hr style="margin: 5px 0;">
    <small>Click sobre regions per veure mix de productes</small>
    </div>
    '''
    m5.get_root().html.add_child(folium.Element(legend_html))
    
    m5.save(f'{MAPS_DIR}/map5_categories_regions.html')
    print("Mapa 5 guardat")
    print(f"   • Regions: {len(region_coords)}")
    
except Exception as e:
    print(f"Error Mapa 5: {e}")
    import traceback
    traceback.print_exc()

# %% RESUM
print("\n" + "="*80)
print("VISUALITZACIONS COMPLETADES")
print("="*80)
print(f"""
 5 mapes interactius generats a {MAPS_DIR}/

1 Choropleth: Risc logístic per país
2 Bubble Map: Concentració d'activitat comercial (2 vistes)
3 Heatmap: Zones d'alta demanda
4 Flow Map: Rutes principals origen-destí
5 Regional: Preferències de producte per regió
   
Per veure els mapes: Obre els fitxers .html amb el navegador
""")
import streamlit as st
import requests
import re
from datetime import datetime
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from geopy.distance import geodesic

# ─── Seitenkonfiguration ──────────────────────────────────────────────────────
st.set_page_config(page_title="🫀 AED-Finder Schweiz", page_icon="🫀", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:700; color:#e63946; margin-bottom:0.2rem; }
    .sub-header  { color:#555; font-size:1rem; margin-bottom:1.5rem; }
    .defi-card   { background:#fff; border-left:5px solid #2a9d8f; border-radius:8px;
                   padding:1rem 1.2rem; margin-bottom:0.8rem; box-shadow:0 2px 6px rgba(0,0,0,0.08); }
    .defi-card.nearest { border-left-color:#e63946; background:#fff5f5; }
    .defi-card.grey    { border-left-color:#ccc; background:#fafafa; }
    .badge-24h     { background:#2a9d8f; color:white; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-limited { background:#f4a261; color:white; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .distance-tag  { font-size:1.1rem; font-weight:700; color:#e63946; }
    .loc-box-ok    { background:#eafaf1; border:1px solid #2a9d8f; border-radius:8px; padding:0.7rem 1rem; font-size:0.88rem; color:#1a5276; margin-bottom:0.5rem; }
    .loc-box-warn  { background:#fef9e7; border:1px solid #f4a261; border-radius:8px; padding:0.7rem 1rem; font-size:0.88rem; color:#7d6608; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Klasse: AED ─────────────────────────────────────────────────────────────
# Eigene Klasse für einen AED-Eintrag.
# Der Vorteil gegenüber einem einfachen Dictionary: wir haben Methoden direkt
# dabei (z.B. get_distanz_text), und der Code wird viel lesbarer.

class AED:
    """Repräsentiert einen einzelnen AED (Defibrillator) mit allen seinen Eigenschaften."""

    def __init__(self, daten_dict):
        # Alle relevanten Felder aus dem übergebenen Dictionary auslesen.
        # .get() gibt None zurück wenn ein Key fehlt – so gibt es keinen KeyError.
        self.lat = daten_dict.get("lat")
        self.lon = daten_dict.get("lon")
        self.name = daten_dict.get("name", "")
        self.oeffnungszeiten = daten_dict.get("oeffnungszeiten", "")
        self.betreiber = daten_dict.get("betreiber", "")
        self.strasse = daten_dict.get("strasse", "")
        self.hausnummer = daten_dict.get("hausnummer", "")
        self.postleitzahl = daten_dict.get("postleitzahl", "")
        self.ort = daten_dict.get("ort", "")
        self.stockwerk = daten_dict.get("stockwerk", "")
        self.telefon = daten_dict.get("telefon", "")
        self.standort_beschreibung = daten_dict.get("standort_beschreibung", "")

        # Diese drei Felder kennen wir erst nach dem Filtern –
        # deshalb starten sie als None und werden später gesetzt.
        self.distanz_m = None
        self.ist_offen = None
        self.bestaetigt = None

    def get_anzeigename(self):
        """Gibt den besten verfügbaren Namen für diesen AED zurück."""
        if self.name:
            return self.name
        elif self.standort_beschreibung:
            return self.standort_beschreibung
        else:
            return "AED"

    def ist_24h(self):
        """Prüft ob dieser AED 24/7 zugänglich ist."""
        oh = self.oeffnungszeiten.strip().lower()
        return oh == "24/7" or oh == "yes" or oh == "always"

    def get_distanz_text(self):
        """Formatiert die Distanz als lesbaren Text (z.B. '350 m' oder '1.2 km')."""
        dm = int(self.distanz_m)
        if dm < 1000:
            return f"{dm} m"
        else:
            return f"{dm / 1000:.1f} km"

    def get_gmaps_link(self, benutzer_lat=None, benutzer_lon=None):
        """Erstellt den Google Maps Navigationslink für diesen AED."""
        if benutzer_lat and benutzer_lon:
            return (f"https://www.google.com/maps/dir/?api=1"
                    f"&origin={benutzer_lat},{benutzer_lon}"
                    f"&destination={self.lat},{self.lon}&travelmode=walking")
        return f"https://www.google.com/maps/dir/?api=1&destination={self.lat},{self.lon}&travelmode=walking"



# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

# Berechnet die Luftlinie zwischen zwei GPS-Punkten in Metern.
# geopy.geodesic nutzt das WGS-84-Ellipsoid – präziser als eine einfache Kugelformel.
def berechne_distanz(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters


# Prüft ob ein AED gerade zugänglich ist, und ob das auch wirklich bestätigt ist.
# Rückgabe: Tupel (ist_offen, ist_bestaetigt)
#   ist_offen     → True wenn der AED vermutlich zugänglich ist (auch wenn Zeiten fehlen)
#   ist_bestaetigt → True nur wenn wir es anhand der Öffnungszeiten sicher wissen
def pruefe_oeffnungszeiten(oeffnungszeiten):
    # Keine Angabe → wir gehen davon aus, dass er zugänglich ist, wissen es aber nicht sicher
    if not oeffnungszeiten or not oeffnungszeiten.strip():
        return True, False

    oh = oeffnungszeiten.strip().lower()

    # Klare 24/7-Angaben → offen und bestätigt
    if oh == "24/7" or oh == "yes" or oh == "always":
        return True, True

    # Aktuelle Zeit und Wochentag ermitteln
    jetzt = datetime.now()
    wochentag = jetzt.weekday()  # 0 = Montag, 6 = Sonntag
    aktuelle_minuten = jetzt.hour * 60 + jetzt.minute

    # Mapping von Tages-Kürzeln auf Wochentag-Nummern
    tag_nummern = {
        "mo": 0,
        "tu": 1,
        "we": 2,
        "th": 3,
        "fr": 4,
        "sa": 5,
        "su": 6
    }

    # Mit Regex nach Zeitblöcken wie "Mo-Fr 08:00-18:00" suchen
    muster = re.compile(
        r'(mo|tu|we|th|fr|sa|su)(?:-(mo|tu|we|th|fr|sa|su))?\s+(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})',
        re.IGNORECASE
    )

    for treffer in muster.finditer(oh):
        tag_von = tag_nummern.get(treffer.group(1).lower(), -1)

        # Entweder ein Tagesbereich (z.B. Mo-Fr) oder ein einzelner Tag
        if treffer.group(2):
            tag_bis = tag_nummern.get(treffer.group(2).lower(), tag_von)
        else:
            tag_bis = tag_von

        # Öffnungs- und Schliesszeit in Minuten seit Mitternacht umrechnen
        oeffnung_min = int(treffer.group(3)) * 60 + int(treffer.group(4))
        schliessung_min = int(treffer.group(5)) * 60 + int(treffer.group(6))

        # Stimmt der heutige Tag und liegt die Uhrzeit im Zeitfenster?
        if tag_von <= wochentag <= tag_bis and oeffnung_min <= aktuelle_minuten <= schliessung_min:
            return True, True  # Zeitfenster passt → bestätigt offen

    # Zeitangabe vorhanden, kein passendes Fenster → bestätigt geschlossen
    return False, True


# Liest AED-Einträge aus einer GeoJSON-Antwort aus und gibt eine Liste von AED-Objekten zurück.
# GeoJSON ist ein weit verbreitetes Format für geografische Daten – jeder Eintrag
# heisst "Feature" und enthält Koordinaten + Properties (die eigentlichen Infos).
def geojson_einlesen(geojson_daten):
    aeds = []

    for eintrag in geojson_daten.get("features", []):
        geometrie = eintrag.get("geometry", {}) or {}
        eigenschaften = eintrag.get("properties", {}) or {}
        koordinaten = geometrie.get("coordinates", [])

        # Einträge ohne Koordinaten können wir nicht auf der Karte anzeigen → überspringen
        if not koordinaten or len(koordinaten) < 2:
            continue

        # Achtung: GeoJSON speichert [Längengrad, Breitengrad] – also umgekehrt als man denkt!
        try:
            laengengrad = float(koordinaten[0])
            breitengrad = float(koordinaten[1])
        except (TypeError, ValueError):
            continue

        # Alle relevanten Felder ins Dictionary packen und daraus ein AED-Objekt bauen
        daten = {
            "id":                     str(eigenschaften.get("@id", eigenschaften.get("id", ""))),
            "lat":                    breitengrad,
            "lon":                    laengengrad,
            "name":                   eigenschaften.get("name", ""),
            "oeffnungszeiten":        eigenschaften.get("opening_hours", ""),
            "betreiber":              eigenschaften.get("operator", ""),
            "strasse":                eigenschaften.get("addr:street", ""),
            "hausnummer":             eigenschaften.get("addr:housenumber", ""),
            "postleitzahl":           eigenschaften.get("addr:postcode", ""),
            "ort":                    eigenschaften.get("addr:city", ""),
            "stockwerk":              eigenschaften.get("level", ""),
            "telefon":                eigenschaften.get("phone", ""),
            "standort_beschreibung":  eigenschaften.get("defibrillator:location", ""),
        }
        aeds.append(AED(daten))

    return aeds


# Lädt alle AED-Daten aus dem Internet.
# @st.cache_data speichert das Ergebnis für 1 Stunde (ttl = time to live in Sekunden),
# damit die App nicht bei jedem Klick neu lädt – das wäre zu langsam.
@st.cache_data(ttl=3600)
def aeds_laden():
    url = "https://raw.githubusercontent.com/OpenBracketsCH/defi_data/main/data/json/defis_switzerland.geojson"
    kopfzeilen = {
        "User-Agent": "AEDFinderApp/1.0",
        "Accept": "application/json"
    }

    try:
        antwort = requests.get(url, headers=kopfzeilen, timeout=30)
        antwort.raise_for_status()  # Wirft eine Exception wenn der HTTP-Status nicht 2xx ist
        return geojson_einlesen(antwort.json())
    except requests.RequestException:
        # Bei Netzwerkfehler leere Liste zurückgeben – die App zeigt dann eine Fehlermeldung
        return []


# ─── ML: KNN-Empfehlung ───────────────────────────────────────────────────────

# Hilfsfunktion: gibt 1 zurück wenn ein Feld einen Wert hat, sonst 0.
# Wird für die Feature-Extraktion gebraucht – KNN arbeitet mit Zahlen, nicht mit Text.
def _text_vorhanden(value):
    return 1 if value is not None and str(value).strip() else 0


def _aed_zu_features(aed):
    """Wandelt ein AED-Objekt in ein Feature-Dictionary für den KNN um."""
    return {
        "distanz_m":        aed.distanz_m or 0,
        "ist_24h":          1 if aed.ist_24h() else 0,
        "ist_offen":        1 if aed.ist_offen else 0,
        "zugänglichkeit_bekannt":  1 if aed.bestaetigt else 0,
        "hat_adresse":      1 if (_text_vorhanden(aed.strasse) or _text_vorhanden(aed.ort)) else 0,
        "hat_standorttext": _text_vorhanden(aed.standort_beschreibung),
        "hat_telefon":      _text_vorhanden(aed.telefon),
        "hat_stockwerk":    _text_vorhanden(aed.stockwerk),
    }


def _empfehlungs_label(row):
    """Weist jedem AED ein Label zu, das später als Trainings-Zielwert dient.

    Kriterien (Kombination aus Distanz + Zugänglichkeit):
      Sehr empfohlen : Zugänglichkeit bekannt UND offen UND ≤ 500 m
      Empfohlen      : offen UND ≤ 1000 m
      Nur wenn nötig : weit weg (> 1000 m) ODER Zugänglichkeit unklar/geschlossen
    """
    bekannt = row["zugänglichkeit_bekannt"] == 1
    offen   = row["ist_offen"] == 1
    ist_24h = row["ist_24h"] == 1
    distanz = row["distanz_m"]

    # 24/7 zählt als bestätigt offen
    if (ist_24h or (bekannt and offen)) and distanz <= 500:
        return "Sehr empfohlen"
    elif (ist_24h or offen) and distanz <= 1000:
        return "Empfohlen"
    else:
        return "Nur wenn nötig"


FEATURE_SPALTEN = [
    "distanz_m", "ist_24h", "ist_offen", "zugänglichkeit_bekannt",
    "hat_adresse", "hat_standorttext", "hat_telefon", "hat_stockwerk",
]


def trainiere_knn_empfehlung(aeds):
    """Trainiert ein KNN-Modell auf den gegebenen (bereits gefilterten) AEDs."""
    rows = [_aed_zu_features(d) for d in aeds]
    tabelle = pd.DataFrame(rows)
    tabelle["empfehlungsklasse"] = tabelle.apply(_empfehlungs_label, axis=1)

    X = tabelle[FEATURE_SPALTEN]
    y = tabelle["empfehlungsklasse"]

    modell = Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn",    KNeighborsClassifier(n_neighbors=min(5, len(aeds)))),
    ])
    modell.fit(X, y)
    return modell


def empfehlung_vorhersagen(aeds, modell):
    """
    Wendet das trainierte KNN-Modell auf alle AEDs an und hängt jedem
    das vorhergesagte Label als Attribut an. Gibt die sortierte Liste zurück.
    """
    rows = [_aed_zu_features(d) for d in aeds]
    tabelle = pd.DataFrame(rows)
    vorhersagen = modell.predict(tabelle[FEATURE_SPALTEN])

    reihenfolge = {"Sehr empfohlen": 0, "Empfohlen": 1, "Nur wenn nötig": 2}

    for aed, label in zip(aeds, vorhersagen):
        aed.knn_empfehlung = label
        aed.knn_sortierung = reihenfolge[label]

    # Sortiert nach: bestätigt offen zuerst, dann KNN-Klasse, dann Distanz.
    # "not d.bestaetigt" ergibt False (=0) wenn bestätigt → kommt weiter vorne.
    return sorted(aeds, key=lambda d: (not d.bestaetigt, d.knn_sortierung, d.distanz_m))


# Filtert alle AEDs nach Radius und Öffnungsstatus,
# berechnet die Distanz und sortiert danach mit KNN (falls genug Daten vorhanden).
def filtern_und_sortieren(alle_aeds, benutzer_lat, benutzer_lon, radius_km, alle_anzeigen):
    gefilterte_aeds = []

    for aed in alle_aeds:
        distanz = berechne_distanz(benutzer_lat, benutzer_lon, aed.lat, aed.lon)

        # Ausserhalb des Suchradius → nicht relevant
        if distanz > radius_km * 1000:
            continue

        offen, bestaetigt = pruefe_oeffnungszeiten(aed.oeffnungszeiten)

        # Wenn der Filter "nur offene" aktiv ist und der AED gerade zu ist → überspringen
        if not alle_anzeigen and not offen:
            continue

        # Distanz und Status direkt am Objekt speichern – so kann die Anzeige direkt darauf zugreifen
        aed.distanz_m = distanz
        aed.ist_offen = offen
        aed.bestaetigt = bestaetigt

        gefilterte_aeds.append(aed)

    # KNN-Sortierung nur wenn mindestens 3 AEDs vorhanden (sonst zu wenig Trainingsdaten)
    if len(gefilterte_aeds) >= 3:
        try:
            modell = trainiere_knn_empfehlung(gefilterte_aeds)
            return empfehlung_vorhersagen(gefilterte_aeds, modell)
        except ValueError as e:
            # Das kann passieren wenn z.B. alle AEDs die gleiche Klasse haben.
            # Sichtbare Warnung statt stiller Fehler – so fällt es beim Testen auf.
            st.warning(f"KNN-Sortierung nicht verfügbar ({e}) – Sortierung nach Distanz.")

    # Einfacher Fallback: bestätigte zuerst, dann nach Distanz
    for aed in gefilterte_aeds:
        aed.knn_empfehlung = None
    return sorted(gefilterte_aeds, key=lambda d: (not d.bestaetigt, d.distanz_m))


# Erstellt das Marker-Icon für einen AED auf der Karte.
# Der nächste AED bekommt ein grösseres, rotes Icon – die anderen sind kleiner und
# entweder grün (24/7 oder bestätigt offen) oder orange (eingeschränkt).
def aed_icon_erstellen(ist_naechster, ist_24h, ist_bestaetigt):
    if ist_naechster:
        farbe = "#e63946"
        groesse = 38
        rahmen = "3px solid white"
        schatten = "0 4px 14px rgba(230,57,70,0.55)"
    elif ist_24h or ist_bestaetigt:
        farbe = "#2a9d8f"  # grün = aktuell zugänglich
        groesse = 30
        rahmen = "2px solid white"
        schatten = "0 2px 8px rgba(42,157,143,0.4)"
    else:
        farbe = "#f4a261"  # orange = eingeschränkte Öffnungszeiten
        groesse = 30
        rahmen = "2px solid white"
        schatten = "0 2px 8px rgba(244,162,97,0.4)"

    # Das Icon ist ein gedrehtes Quadrat mit abgerundetem Ende – klassische Kartennadel-Form
    html = f"""
    <div style="width:{groesse}px;height:{groesse}px;border-radius:50% 50% 50% 0;
                transform:rotate(-45deg);background:{farbe};border:{rahmen};
                box-shadow:{schatten};display:flex;align-items:center;justify-content:center;">
    </div>"""

    return folium.DivIcon(
        html=html,
        icon_size=(groesse, groesse),
        icon_anchor=(0, groesse),
        popup_anchor=(groesse // 2, -groesse // 2),
    )


# Baut die interaktive Folium-Karte mit allen AED-Markern.
def karte_erstellen(aeds_anzeigen, benutzer_lat, benutzer_lon):
    # Karte auf den aktuellen Standort zentrieren, mit schönem Carto-Kartenstil
    karte = folium.Map(
        location=[benutzer_lat, benutzer_lon],
        zoom_start=15,
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
    )


    # Eigenen Standort als blauer Punkt einzeichnen
    folium.CircleMarker(
        location=[benutzer_lat, benutzer_lon],
        radius=10,
        color="#3b82f6",
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.9,
        weight=3,
        popup=folium.Popup("<b>GPS-Standort</b>", max_width=150),
        tooltip="GPS-Standort",
    ).add_to(karte)

    # Grösserer, halbtransparenter Kreis als Standortring
    folium.CircleMarker(
        location=[benutzer_lat, benutzer_lon],
        radius=20,
        color="#3b82f6",
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.15,
        weight=1,
    ).add_to(karte)

    # Marker für jeden AED hinzufügen – maximal 80 damit die Karte flüssig bleibt
    anzahl = 0
    for aed in aeds_anzeigen:
        if anzahl >= 80:
            break

        name = aed.get_anzeigename()
        adresse = " ".join(filter(None, [aed.strasse, aed.hausnummer]))
        ort = " ".join(filter(None, [aed.postleitzahl, aed.ort]))
        distanz_text = aed.get_distanz_text()
        gmaps_link = aed.get_gmaps_link(benutzer_lat, benutzer_lon)
        oh_anzeige = aed.oeffnungszeiten or "unbekannt"

        if anzahl == 0:
            label = "Nächster AED"
        else:
            label = f"#{anzahl + 1}"

        if aed.ist_24h():
            badge_farbe = "#2a9d8f"
            badge_text = "24/7"
        elif aed.bestaetigt:
            badge_farbe = "#2a9d8f"
            badge_text = aed.oeffnungszeiten or "?"
        elif aed.oeffnungszeiten:
            badge_farbe = "#f4a261"
            badge_text = aed.oeffnungszeiten
        else:
            badge_farbe = "#aaaaaa"
            badge_text = "?"

        # Adresszeile nur anzeigen wenn eine Adresse vorhanden ist
        adresse_zeile = ""
        if adresse:
            adresse_zeile = f"<div>{adresse}"
            if ort:
                adresse_zeile += f", {ort}"
            adresse_zeile += "</div>"

        popup_html = f"""
        <div style='font-family:sans-serif;font-size:13px;min-width:190px;line-height:1.5'>
          <div style='font-weight:700;margin-bottom:4px'>{label}</div>
          <div style='color:#333;margin-bottom:6px'><i>{name}</i></div>
          <div style='margin-bottom:6px'>
            <span style='background:{badge_farbe};color:white;padding:1px 8px;
                         border-radius:10px;font-size:11px'>{badge_text}</span>
          </div>
          {adresse_zeile}
          <div>{oh_anzeige}</div>
          <div><b>{distanz_text}</b></div>
          <div style='margin-top:8px'>
            <a href='{gmaps_link}' target='_blank'
               style='background:#e63946;color:white;padding:4px 12px;
                      border-radius:10px;text-decoration:none;font-size:12px'>
              🚶 Navigation starten
            </a>
          </div>
        </div>"""

        # Marker zur Karte hinzufügen – Popup erscheint beim Klick, Tooltip beim Hover
        folium.Marker(
            location=[aed.lat, aed.lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{distanz_text} – {name}",
            icon=aed_icon_erstellen(anzahl == 0, aed.ist_24h(), aed.bestaetigt),
        ).add_to(karte)

        anzahl += 1

    return karte


# ─── App Layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🫀 AED-Finder Schweiz</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.subheader("GPS-Standort")

    gps_daten = streamlit_geolocation()

    benutzer_lat = None
    benutzer_lon = None

    # GPS-Koordinaten auslesen falls der Browser sie geliefert hat
    if gps_daten and isinstance(gps_daten, dict):
        if gps_daten.get("latitude") is not None and gps_daten.get("longitude") is not None:
            benutzer_lat = float(gps_daten["latitude"])
            benutzer_lon = float(gps_daten["longitude"])

            # Genauigkeit in Metern anzeigen wenn vorhanden
            genauigkeit = gps_daten.get("accuracy")
            if genauigkeit:
                genauigkeit_text = f" (±{int(genauigkeit)} m)"
            else:
                genauigkeit_text = ""

            st.markdown(
                f'<div class="loc-box-ok">GPS-Standort{genauigkeit_text}<br>'
                f'<small>{benutzer_lat:.5f}, {benutzer_lon:.5f}</small></div>',
                unsafe_allow_html=True
            )

    # Kein GPS → Zürich HB als Standardposition (damit die App trotzdem nutzbar ist)
    if benutzer_lat is None:
        st.markdown(
            '<div class="loc-box-warn">Noch kein Standort – GPS verwenden.</div>',
            unsafe_allow_html=True
        )
        benutzer_lat = 47.3769
        benutzer_lon = 8.5417  # Zürich HB

    st.divider()
    st.subheader("Filter")
    radius_km = st.slider("Suchradius (km)", 0.2, 10.0, 1.5, 0.1)
    max_resultate = st.slider("Max. Resultate", 3, 30, 10)
    alle_anzeigen = st.checkbox("Auch geschlossene AEDs anzeigen", value=False)

    st.divider()
    st.info(f"{datetime.now().strftime('%a %d.%m.%Y %H:%M')}")

    # Cache leeren und Seite neu laden – nützlich wenn die Daten veraltet wirken
    if st.button("Daten neu laden", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── Daten laden ──────────────────────────────────────────────────────────────
alle_aeds = aeds_laden()

if not alle_aeds:
    st.error("Daten nicht verfügbar – bitte 'Daten neu laden' versuchen.")
    st.stop()

# Filtern, Distanz berechnen, sortieren
aeds_gefiltert = filtern_und_sortieren(alle_aeds, benutzer_lat, benutzer_lon, radius_km, alle_anzeigen)

# Nur so viele anzeigen wie im Slider eingestellt
aeds_anzeigen = aeds_gefiltert[:max_resultate]

# ─── Karte + Liste ────────────────────────────────────────────────────────────
# Layout mit zwei Spalten: Karte links (breiter), Liste rechts
spalte_karte, spalte_liste = st.columns([3, 2])

with spalte_karte:
    st.subheader("Karte")
    if not aeds_anzeigen:
        st.warning(f"Keine zugänglichen AEDs im Umkreis von {radius_km} km.")
    karte = karte_erstellen(aeds_anzeigen, benutzer_lat, benutzer_lon)
    st_folium(karte, width="100%", height=540, returned_objects=[])

    if aeds_gefiltert:
        with st.expander("Verteilung im Suchradius"):
            distanzen = [aed.distanz_m for aed in aeds_gefiltert]
            klassen = [getattr(aed, "knn_empfehlung", None) for aed in aeds_gefiltert]

            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8))
            fig.patch.set_facecolor("none")

            # Histogramm: Distanzverteilung
            ax1.hist(distanzen, bins=12, color="#2a9d8f", edgecolor="white")
            ax1.set_xlabel("Distanz in Metern", fontsize=9)
            ax1.set_ylabel("Anzahl AEDs", fontsize=9)
            ax1.set_title("Distanzverteilung", fontsize=10)
            ax1.tick_params(labelsize=8)

            # Balkendiagramm: Klassenverteilung
            klassen_farben = {
                "Sehr empfohlen":       "#2a9d8f",
                "Empfohlen":            "#2a9d8f",
                "Nur wenn nötig": "#f4a261",
            }
            reihenfolge = ["Sehr empfohlen", "Empfohlen", "Nur wenn nötig"]
            anzahlen = {k: klassen.count(k) for k in reihenfolge}

            balken = ax2.bar(
                reihenfolge,
                [anzahlen[k] for k in reihenfolge],
                color=[klassen_farben[k] for k in reihenfolge],
                edgecolor="white",
            )
            for bar, k in zip(balken, reihenfolge):
                h = bar.get_height()
                if h > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)),
                             ha="center", va="bottom", fontsize=8)
            ax2.set_ylabel("Anzahl AEDs", fontsize=9)
            ax2.set_title("KNN-Empfehlung", fontsize=10)
            ax2.tick_params(axis="x", labelsize=8)
            ax2.tick_params(axis="y", labelsize=8)

            # Confusion Matrix
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html?utm_source=chatgpt.com
            rows = [_aed_zu_features(aed) for aed in aeds_gefiltert]
            tabelle = pd.DataFrame(rows)

            # Actual Klassen
            y_true = tabelle.apply(_empfehlungs_label, axis=1)

            # Prediction von KNN
            y_pred = [aed.knn_empfehlung for aed in aeds_gefiltert]

            labels = ["Sehr empfohlen", "Empfohlen", "Nur wenn nötig"]

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels
            )

            anzeige = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=labels
                )
            
            anzeige.plot(
                ax=ax3,
                colorbar=False
                )
            
            ax3.set_title("Confusion Matrix", fontsize=10)
            ax3.tick_params(axis="x", rotation=15)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.caption(f"{len(aeds_gefiltert)} AEDs im Umkreis gefunden")
    st.markdown(
        '<div class="sub-header">Nächsten zugänglichen AED finden – '
        'Daten via <a href="https://defikarte.ch" target="_blank">defikarte.ch</a> / OpenStreetMap</div>',
        unsafe_allow_html=True
    )

with spalte_liste:

    if not aeds_gefiltert:
        st.info("Suchradius erhöhen oder 'Geschlossene anzeigen' aktivieren.")
    else:
        # Jeden AED als HTML-Karte ausgeben
        for i in range(len(aeds_anzeigen)):
            aed = aeds_anzeigen[i]

            # Alle Anzeigewerte direkt über die Methoden des AED-Objekts holen
            distanz_text = aed.get_distanz_text()
            oh = aed.oeffnungszeiten.strip()
            bestaetigt = aed.bestaetigt
            name = aed.get_anzeigename()
            adresse = " ".join(filter(None, [aed.strasse, aed.hausnummer]))
            ort = " ".join(filter(None, [aed.postleitzahl, aed.ort]))
            betreiber = aed.betreiber
            telefon = aed.telefon
            stockwerk = aed.stockwerk
            gmaps_link = aed.get_gmaps_link(benutzer_lat, benutzer_lon)

            # Farbiger Badge je nach Öffnungszeit-Status
            if aed.ist_24h():
                badge = '<span class="badge-24h">24/7</span>'
            elif oh and bestaetigt:
                badge = f'<span class="badge-24h">{oh}</span>'
            elif oh and not bestaetigt:
                badge = f'<span class="badge-limited">{oh}</span>'
            else:
                badge = '<span style="color:#aaa;font-size:0.8rem">Zeiten unbekannt</span>'

            # Kartenstil: der nächste AED bekommt eine rote Hervorhebung
            if i == 0:
                karten_klasse = "defi-card nearest"
                label = "Nächster AED"
            else:
                karten_klasse = "defi-card grey"
                label = f"#{i + 1}"

            # Details-Block: nur Felder anzeigen die auch wirklich einen Wert haben
            details = ""
            if adresse:
                details += f"{adresse}"
                if ort:
                    details += f", {ort}"
                details += "<br>"
            elif ort:
                details += f"{ort}<br>"
            if betreiber:
                details += f"{betreiber}<br>"
            if telefon:
                details += f"{telefon}<br>"
            if stockwerk:
                details += f"Stockwerk: {stockwerk}<br>"

            # KNN-Badge: zeigt die Empfehlung aus dem ML-Modell an (falls vorhanden)
            knn_label = getattr(aed, "knn_empfehlung", None)
            if knn_label == "Sehr empfohlen":
                knn_badge = '<span style="background:#2a9d8f;color:white;padding:2px 9px;border-radius:12px;font-size:0.75rem;font-weight:600">Sehr empfohlen</span>'
            elif knn_label == "Empfohlen":
                knn_badge = '<span style="background:#2a9d8f;color:white;padding:2px 9px;border-radius:12px;font-size:0.75rem;font-weight:600">Empfohlen</span>'
            elif knn_label == "Nur wenn nötig":
                knn_badge = '<span style="background:#f4a261;color:white;padding:2px 9px;border-radius:12px;font-size:0.75rem;font-weight:600">Nur wenn nötig</span>'
            else:
                knn_badge = ""

            st.markdown(f"""
            <div class="{karten_klasse}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:700;color:#333">{label} – {name}</span>
                    <span class="distance-tag">{distanz_text}</span>
                </div>
                <div style="margin:6px 0;display:flex;gap:6px;flex-wrap:wrap">{badge}{knn_badge}</div>
                <div style="color:#555;font-size:0.85rem;line-height:1.6">{details}</div>
                <div style="margin-top:8px">
                    <a href="{gmaps_link}" target="_blank"
                       style="background:#e63946;color:white;padding:5px 14px;
                              border-radius:20px;text-decoration:none;font-size:0.82rem">
                       🚶 Navigation starten
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)



import os
import tempfile

import streamlit as st
from PIL import Image

from cbir import rechercher, MESURES_DISTANCE, FONCTIONS_DESCRIPTEURS


st.set_page_config(
    page_title='CBIR — Recherche d\'images par le contenu',
    page_icon='🔍',
    layout='wide'
)

st.title('Recherche d\'images par le contenu (CBIR)')
st.markdown('Téléversez une image, choisissez un descripteur et une mesure de distance pour trouver les images similaires.')


st.sidebar.header('Paramètres de recherche')

descripteur_choisi = st.sidebar.selectbox(
    'Descripteur',
    options=list(FONCTIONS_DESCRIPTEURS.keys()),
    format_func=lambda nom: nom.upper()
)

distance_choisie = st.sidebar.selectbox(
    'Mesure de distance',
    options=list(MESURES_DISTANCE.keys()),
    format_func=lambda nom: nom.capitalize()
)

nb_resultats = st.sidebar.slider(
    'Nombre d\'images similaires',
    min_value=1,
    max_value=20,
    value=5
)


fichier_televerse = st.file_uploader(
    'Téléversez une image',
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if fichier_televerse is not None:
    image_requete = Image.open(fichier_televerse)

    col_requete, col_info = st.columns([1, 2])
    with col_requete:
        st.subheader('Image requête')
        st.image(image_requete, use_container_width=True)

    # Sauvegarde temporaire pour traitement, car les fonctions de recherche et les descripteurs attendent un chemin de fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as fichier_temp:
        image_requete.save(fichier_temp.name)
        chemin_temp = fichier_temp.name

    with col_info:
        st.subheader('Analyse')
        with st.spinner('Extraction des caractéristiques et recherche en cours...'):
            try:
                classe_predite, resultats = rechercher(
                    chemin_temp,
                    descripteur_choisi,
                    distance_choisie,
                    nb_resultats
                )
                st.success(f'Classe prédite : **{classe_predite}**')
                st.info(
                    f'Descripteur : **{descripteur_choisi.upper()}** | '
                    f'Distance : **{distance_choisie.capitalize()}** | '
                    f'Résultats : **{nb_resultats}**'
                )
            except FileNotFoundError as erreur:
                st.error(f'Fichier introuvable : {erreur}')
                st.stop()
            except Exception as erreur:
                st.error(f'Erreur lors de la recherche : {erreur}')
                st.stop()

    os.unlink(chemin_temp)

    # ─────────────────────────────────────────────
    #  Affichage des résultats
    # ─────────────────────────────────────────────

    st.subheader(f'Images similaires — classe « {classe_predite} »')

    nb_colonnes = min(5, nb_resultats)
    colonnes = st.columns(nb_colonnes)

    for rang, (chemin_image, distance) in enumerate(resultats):
        # Affichage dans la colonne correspondante de façon à ce que les résultats soient organisés en lignes de nb_colonnes images (grille responsive)
        with colonnes[rang % nb_colonnes]:
            image_resultat = Image.open(chemin_image)
            st.image(image_resultat, width='stretch')
            st.caption(f'#{rang + 1} | distance : {distance:.4f}')

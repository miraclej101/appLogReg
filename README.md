# appLogReg
Application qui analyse les données avec le modèle de régression logistique. 
## Condition nécessaire
Les bibliothèques de Python requise sont dans le fichier requirements.txt.

Des bibliothèques principales :
- imbalanced-learn v.0.11.0
- matplotlib v.3.7.1
- numpy v.1.23.5
- pandas v.1.5.3
- plotly v.5.14.1
- Python v.3.9
- scikit-learn v.1.3.2
- seaborn v.0.12.2
- statsmodels v.0.14.1
- streamlit v.1.29.0

## Fonctionalités
- Uploader un fichier CSV pour entrer les données
- Explorer les données en description et en graphique
- Traitement les données manquantes  et imputer des données  
- Enlever des variables selon les critères statistique tels que p-value et VIF 
- Possiblité d'effectuer le suréchatillage
- Fitting dans le modèle de regréession logistique pour prevenir les résultats d'une variable cible
- Prédire un résultat ou des résultats par soit entrer maunuellement soit uploader un fihcier de CSV

Vous pourriez tester l'application sur Streamlit Cloud : [applogreg](https://applogreg.streamlit.ap/)

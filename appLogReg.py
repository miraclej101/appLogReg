import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
from pandas.api.types import is_numeric_dtype
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
import math
import operator
# Configurer la page de l'application
st.set_page_config(layout = "wide")
#global data, val
#val = ""
#Initialise session state of une clé dataFrame pour passer travers menus
if "dataFrame" not in st.session_state :
    st.session_state.dataFrame = pd.DataFrame()

def load_data(file, delimiter, encode) :
    data = pd.read_csv(file, sep = delimiter, encoding = encode)
    return data

codecs = ['ascii','big5','big5hkscs','cp037','cp273','cp424','cp437','cp500','cp720','cp737','cp775','cp850','cp852','cp855',
          'cp856','cp857','cp858','cp860','cp861','cp862','cp863','cp864','cp865','cp866','cp869','cp874','cp875','cp932','cp949',
          'cp950','cp1006','cp1026','cp1125','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','cp1255','cp1256','cp1257','cp1258',
          'euc_jp','euc_jis_2004','euc_jisx0213','euc_kr','gb2312','gbk','gb18030','hz','iso2022_jp','iso2022_jp_1','iso2022_jp_2',
          'iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext','iso2022_kr','latin_1','iso8859_2','iso8859_3','iso8859_4','iso8859_5','iso8859_6',
          'iso8859_7','iso8859_8','iso8859_9','iso8859_10','iso8859_11','iso8859_13','iso8859_14','iso8859_15','iso8859_16','johab','koi8_r','koi8_t',
          'koi8_u','kz1048','mac_cyrillic','mac_greek','mac_iceland','mac_latin2','mac_roman','mac_turkish','ptcp154','shift_jis','shift_jis_2004',
          'shift_jisx0213','utf_32','utf_32_be','utf_32_le','utf_16','utf_16_be','utf_16_le','utf_7','utf_8','utf_8_sig']

codecsArray = np.array(codecs) 

#Function à trouver les colonnes avec  un type numéric
def columsNumeric(data) :
    colNum = []
    if ~data.empty :
        for col in data.columns :
            if is_numeric_dtype(data[col]) :
                colNum.append(col)
    return colNum

#Function à trouver les columns de type de catégorie
def catColums(data) :
    cat_col = ['object']
    df_catcols = data.select_dtypes(include=cat_col)
    return df_catcols.columns

def calculVal(method, serie) :
    if method == "mean" :
        val = serie.mean()
    elif method == "median" :
        val = serie.median()
    else :
        val = serie.mode()
    return val

def makeTextMode(serie) :
    text=""
    for val in serie :
        text += str(val)+" "
    text = text.strip()
    return text

#Fonction à montrer des correlations entre variables  
@st.cache_data
def plotCorr(df) :
    cor = df.corr(numeric_only=True)
    fig = plt.figure(figsize=(8,4))
    plt.title("Matrice de Corrélation")
    sns.heatmap(cor, annot=True,linewidth=.5, fmt=".3f", cmap="plasma")
    st.pyplot(fig)

#Fonction à dessiner des plots de correlations entre variables  
@st.cache_data 
def plotPairGraph(df, cible) :
    plt.figure(figsize = (12, 12)) 
    sns.pairplot(df,hue = cible)
    st.pyplot(plt)

# Sidebar paramétrage
with st.sidebar :
    st.header("Paramétrage ")

# Main affichage
selected = option_menu("Régression Logistique", ["Accueil", "Lecture des données", "Statistiques descriptives",
        "Représentations graphiques des variables","Corrélation","Tranformation des données","p-value et facteur d'inflation de variance (VIF)",
        "Régression","Prédiction"], 
        icons=["house-fill", "book-fill", "file-spreadsheet-fill", "bar-chart-fill", "table", "clipboard-data","graph-down",
        "graph-up","graph-up-arrow"], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={   
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#77b5fe"},
        })
selected

# Rubrique Accueil
if selected == "Accueil" :
    st.header("Régression logistique")
    st.markdown("La régression logistique est une méthode prédictive. \
    Elle vise à construire un modèle permettant de prédire les valeurs prises par une variable cible qualitative binaire.")
    st.markdown("Avec la régression logistique, il est désormais possible d'expliquer la variable cible ou d'estimer la probabilité \
    d'occurrence des catégories de la variable.")

    st.subheader("Exemple en marketing :")
    st.write("Pour un détaillant en ligne, vous devez prédire quel produit un client donné est le plus susceptible d'acheter. Pour ce faire, \
        vous disposez d'un ensemble de données concernant les visiteurs précédents et leurs achats auprès du détaillant en ligne.")

    st.subheader("Exemple en médecine :")
    st.write("Vous souhaitez déterminer si une personne est susceptible ou non de contracter une certaine maladie. Pour ce faire,\
        vous recevez un ensemble de données comprenant des personnes malades et non malades, ainsi que d'autres paramètres médicaux.")

    st.subheader("Exemple en politique :")
    st.write("Une personne voterait-elle pour le parti A s'il y avait des élections le week-end prochain ?")


# Rubrique Lecture des données
if selected == "Lecture des données" :
    col1, col2 = st.columns(2)
    with col1 :
        delimiter = st.text_input("Entrer un délimiter pour un fichier CSV", placeholder = "Par exemple ; ou , ou | ")
    with col2 :
        encode = st.selectbox("Sélectionner un codec :", codecsArray, index = 95)
    uploaded_file = st.file_uploader("Télécharger le fichier ici", type=["csv"])
    if uploaded_file is not None :
        if delimiter == "" :
            # delimiter par défault = ";"
            delimiter = ";"
        
        data = load_data(uploaded_file, delimiter, encode) 
        colNums = columsNumeric(data)
        catCols = catColums(data)
        st.session_state.colNums = colNums
        st.session_state.catCols = catCols
        # Set session avec un key dataFrame pour contenir une varaible data
        st.session_state.dataFrame = data
        st.dataframe(data.head()) 

#Rubrique Statistiques descriptives
#st.session_state.dataFrame = data
def crossTab(data, selected_colnums, selected_catcols, container) :
    for col in selected_colnums :
        container.markdown("**"+col+"**")
        container.table(data.groupby(selected_catcols)[col].describe())
    

if selected == "Statistiques descriptives" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        st.write(data.describe())
        container = st.container()
         # Side bar paramétrage
        with st.sidebar :
            selected_colnums = st.multiselect(label="Variables quantitatives", options=st.session_state.colNums, placeholder="Sélectionner une ou des variables")
            selected_catcols = st.multiselect(label="Variables qualitatives",options=st.session_state.catCols, placeholder="Sélectionner une ou des variables")
            btn_decrire = st.button("Décrire des variables")
            if btn_decrire :
                if len(selected_colnums) != 0 and len(selected_catcols) !=0 :
                    crossTab(data, selected_colnums, selected_catcols, container)
                else :
                    st.warning("Merci de sélectionner des variables qualitatives et quantitatives!")

# Rubrique Représentations graphiques des variables
if selected == "Représentations graphiques des variables" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        with st.sidebar :
            selected_colnums = st.multiselect(label="Variables quantitatives", options=st.session_state.colNums, placeholder="Sélectionner une ou des variables") 
            selected_colcats = st.multiselect(label="Variables qualitatives", options=st.session_state.catCols, placeholder="Sélectionner une ou des variables") 
        if len(selected_colnums) != 0 :
            nbcol = 2
            nbrow = len(selected_colnums)
          #  st.write("nbrow = ", nbrow)
            fig, axs = plt.subplots(nbrow, nbcol, tight_layout=True)
            axs = axs.flat
            row = 0
            for i,ax in enumerate(axs) :
                col = selected_colnums[row]
                if i%2 == 0 :   
                    ax.set_title(col)
                    ax.hist(x=data[col])
                else :
                    ax.boxplot(x=data[col])
                    row +=1
            st.pyplot(fig = plt)
        if len(selected_colcats) !=0 :
            for col in selected_colcats :
                plt.figure(figsize=(10,4))
                sns.barplot(x=data[col].value_counts().values, y=data[col].value_counts().index)
                plt.title(col)
                plt.tight_layout()
                st.pyplot(fig = plt)

# Rubrique Corrélation
if selected == "Corrélation" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        with st.sidebar :
            cible = st.selectbox("Variable cible", data.columns, placeholder="Sélectionner une variable cible")
        plotCorr(data)
        if st.button("Pair plot entre des variables numériques") :
            plotPairGraph(data, cible)

# Rubrique Tranformation des données
def convertType(data, val) :
    int_col = ['int16','int32','int64']
    float_col = ['float16','float32','float64']
    if val is None :
        return
    if data.dtypes in int_col :
        val = int(val)
    elif data.dtypes in float_col :
        val = float(val)
    elif data.dtypes == 'unit8' :
        val = np.unit8(val)
    return val

if selected == "Tranformation des données" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        st.dataframe(data)
        df_miss_data = pd.DataFrame({'Cols' : data.columns, 'Types' : np.array(data.dtypes), 'NANs' : np.array(data.isna().sum())})
        df_miss_data
        columns = data.columns
        col1, col2 = st.columns(2)
        selected_col = st.sidebar.selectbox("Sélectionneer une variable à traiter", columns, key = "column")
        with col1 :
             st.markdown("#### :blue[Transformer un type de donnée dans la colonne sélectionnée]")
        with col2 :
            selected_type = st.selectbox("Sélectionner un type de donnée :",('object', 'numérique', 'category', 'datetime', 'bool'))
            if st.button("Transformer") and selected_col is not None :
                if selected_type == "object" or selected_type == "category" or selected_type == "bool" :
                    data[selected_col] = data[selected_col].astype(selected_type)
                elif selected_type == "numérique" :
                    data[selected_col] = pd.to_numeric(data[selected_col])
                else :
                    data[selected_col] = pd.to_datetime(data[selected_col], format = '%d/%m/%Y')
        
        col3, col4 =st.columns(2)
        with col3 :
            st.markdown("#### :blue[Remplacer les données manquantes dans la colonne sélectionnée par :]")
        
        with col4 :
            if is_numeric_dtype(data[selected_col]) :
                radio_disabled = False
                selected_text_disabled = True     
            else :
                radio_disabled = True
                selected_text_disabled = False
               
            text_disabled = True
            texts = data[selected_col].dropna().unique()
            list_selected_text = list(texts)
            list_selected_text.append("Autre")
            if not(selected_text_disabled) :
                modeSerie = data[selected_col].dropna().mode()
                textMode = makeTextMode(modeSerie)
                st.markdown(f'Le mode = :blue[{textMode}]') 
            selected_text = st.selectbox("Sélectionner une valeur existante dans la colonne :", list_selected_text, disabled = selected_text_disabled)
            print(selected_text)
            if selected_text == "Autre" :
                text_disabled = False
            text = st.text_input("Entrer un mot :", disabled=text_disabled)
            method = st.radio("Choisir une façon à remplacer :",("mean","median","mode"), key = "methodNumeric", disabled = radio_disabled)
            serie = st.session_state.dataFrame[st.session_state.column]
            if is_numeric_dtype(serie) :
                val = calculVal(method, serie)
                if isinstance(val, pd.Series) :
                    textMode = makeTextMode(val)
                    st.markdown(f'Valeur = {textMode}')
                else :
                    st.markdown(f"Valeur = {val}")
            if st.button("Remplacer") :
                if text != "" :
                    data[selected_col].fillna(text, inplace = True)
                elif not(selected_text_disabled) and selected_text != "Autre" :
                    data[selected_col].fillna(selected_text, inplace = True)
                elif method != "mode" :     
                    data[selected_col].fillna(val, inplace = True)
                else :
                    # Choisir le première valeur de mode trouvée
                    data[selected_col].fillna(val[0], inplace = True)
                st.rerun()
         
        col5, col6 = st.columns(2)
        with col5 :
            st.markdown("#### :blue[ou supprimer les lignes de données manquantes en critère :]")

        with col6 :
            delete_line = st.radio("""any : Si des valeurs NA sont présentes, supprimez cette ligne. 
                                    all : Si toutes les valeurs sont NA, supprimez cette ligne.""",
                         ('any','all'), index=1)
            st.error("Voulez-vous supprimer les lignes de données manquantes?", icon="🚨")
            if st.button("Oui, supprimer ces lignes") :
                if delete_line == 'any' :
                    data.dropna(axis = 0, how = 'any', inplace = True)
                elif delete_line =='all' :
                    data.dropna(axis = 0, how = "all", inplace = True)
                st.rerun()

        col7, col8 =st.columns(2) 
        with col7 :
            st.markdown("#### :blue[ou supprimer les colonnes de données manquantes en critère :]") 
        with col8 :
            delete_column = st.radio("""any : Si des valeurs NA sont présentes, supprimez cette colonne. 
                                    all : Si toutes les valeurs sont NA, supprimez cette colonne.""",
                         ('any','all'), index=1)
            st.error("Voulez-vous supprimer les colonnes de données manquantes?", icon="🚨")
            if st.button("Oui, supprimer ces colonnes") :
                if delete_column == 'any' :
                    data.dropna(axis = 1, how = 'any', inplace = True)
                elif delete_column =='all' :
                    data.dropna(axis=1, how = "all", inplace = True)
                st.rerun()
    # imputation des données
        with st.sidebar :
            st.divider()
            st.subheader("Explore des données par crosstab")
            vindex = st.selectbox("Variable à mettre en index", data.columns, index=None)
            vcol = st.selectbox("Variable à mettre en colonne", data.columns, index=None)
       
        if vindex is not None and vcol is not None :
            st.dataframe(pd.crosstab(index=data[vindex], columns=data[vcol]))

        st.markdown("#### :blue[Imputation]")
        st.subheader(selected_col, ":")  
        col9, col10= st.columns(2)
        # Initialise des valeurs pour vérifier une condition
        replaced_val = None
        vinput= None
        with col9 :
            if selected_col in st.session_state.catCols :
                replaced_val = st.text_input("Une valeur de catégorie ", value=None)
            else:
                replaced_val = st.number_input("Une valeur numérique ", value=None)
                replaced_val = convertType(data[selected_col], replaced_val)
        with col10 :
            if selected_col in st.session_state.catCols :
                vinput = st.text_input("remplacée par ", value=None)
            else:
                vinput = st.number_input("remplacée par ", value=None)
                vinput = convertType(data[selected_col], vinput)
        
        st.write("avec la condition :")     
        # Dictionary of opérateurs logiques
        ops = {
                "==" : operator.eq,
                "<"  : operator.lt,
                "<=" : operator.le,
                ">"  : operator.gt,
                ">=" : operator.ge,
                "!=" : operator.ne
        }                 
        col11, col12, col13 = st.columns(3)
        with col11 :
            vcond = st.selectbox("Sélectionner une variable", data.columns, index=None)
        with col12 :
            if vcond is not None :
                # opérateur
                op = st.selectbox("Choisir un opérateur logique", ops.keys())
        with col13 :
            if vcond is not None :                    
                if vcond in st.session_state.catCols :
                    val = st.text_input("Saisir une valeur", value=None)
                else :
                    val = st.number_input("Saisir une valeur", value=None)
                    val = convertType(data[vcond], val)
      
        if st.button("Imputer") :
            if vcond is not None and val is not None :
                if replaced_val is not None and vinput is not None:                
                    data.loc[ops[op](data[vcond], val) & (data[selected_col] == replaced_val), selected_col] = vinput                
                    st.session_state.dataFrame = data
                    st.success("Données transformées")
                    if st.button("Rafraîchir") :
                        st.rerun()
            else :
                if replaced_val is not None and vinput is not None :
                    data[selected_col].replace(to_replace=[replaced_val], value=[vinput], inplace=True)
                    st.session_state.dataFrame = data
                    st.success("Données transformées")
                    if st.button("Rafraîchir") :
                        st.rerun()

    # Après le traitement , on peut enregistrer les données en fichiers csv
    col14, col15 = st.columns(2)
    with col14 :
        st.markdown("### :blue[Enregistre le fichier :]")
        delimiter = st.selectbox("Sélectionner un délimiteur :", (",",";","|"))
        st.download_button("Enregistre le fichier csv",
                        data.to_csv(sep = delimiter, encoding = "utf_8", index=False),
                        mime = "text/csv") 

# Rubrique p-value et facteur d'inflation de variance (VIF)
@st.cache_data
def splitData(data, target, testSize) :
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=testSize, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

@st.cache_data
def VIF(data) :
    vif= pd.DataFrame()
    vif['Features'] = data.columns
    vif['vif']=[variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif

if selected == "p-value et facteur d'inflation de variance (VIF)" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        catCols = st.session_state.catCols
        if "dummy" not in st.session_state :
            st.session_state.dummy = pd.get_dummies(data=data, columns=catCols, drop_first=True)
        dummy = st.session_state.dummy

        with st.sidebar :
            cible = st.selectbox("Variable cible", data.columns, index=None)
            proptest = st.slider("Proportion de l'échantillon de test", min_value=0.1, max_value=1.0, step=0.1, value= 0.3) 

        with st.expander("Ouvrir pour un résult de dummy encoding") :
            st.dataframe(dummy)  

        if st.button("Rafraîchir Dummy Encoding", type="primary") :
            dummy = pd.get_dummies(data=data, columns=catCols, drop_first=True)
            st.session_state.dummy = dummy

        if cible is not None :
            if cible in dummy.columns : 
                target = dummy.pop(cible)
                st.session_state.target = target
            else :
                # Récuperer une variable target
                target = st.session_state.target
            X_train, X_test, y_train, y_test = splitData(dummy, target, proptest)
            logit_model = sm.Logit(y_train, X_train)
            result = logit_model.fit()
            with st.expander("Sommaire de résultat") :
                st.write(result.summary())
            cols_to_drop = [col for col in result.pvalues.index if result.pvalues[col] > 0.05 ] 
            st.write(" ")
            st.markdown(":red[**Variables à enlever**]")
            st.table(cols_to_drop)
            rdb = st.radio(":blue[**Analyses statistiques**]", ["p-value", "VIF"], 
                captions=["Suppression des variables qui ont p>0,05","Facteur d'inflation de la variance. Suppression des variables ayant vif>5"],
                index=None)
            if rdb == "p-value" :
                col_drop = st.button("Enlever", type="primary") 
                if col_drop : 
                    dummy.drop(columns=cols_to_drop, inplace=True)
                    st.session_state.dummy = dummy
                    st.rerun()
            if rdb == "VIF" :
                vif = VIF(dummy)
                with st.expander("VIF résultat") :
                    st.dataframe(vif)
                choix_to_drop = vif[vif['vif']>5]
                col1, col2 = st.columns(2)
                with col1 :
                    st.dataframe(choix_to_drop)  
                with col2 :
                    drop_cols = st.multiselect("Sélectionner une ou des variables à enlever", choix_to_drop.Features) 
                if st.button("Enlever", type="primary") :
                    dummy.drop(columns=drop_cols, inplace=True)
                    st.session_state.dummy = dummy
                    st.rerun()
        else :
            st.warning("Merci de choisir une variable cible.")

# Rubrique "Régression"
@st.cache_data
def logisticModel(X, Y, maxIter, tol, solver) :
    model = LogisticRegression(max_iter=maxIter, tol=tol, solver=solver)
    model.fit(X, Y)
    return model

if selected == "Régression" :
    data = st.session_state.dataFrame
    dummy = st.session_state.dummy
    
    with st.sidebar :
        cible = st.selectbox("Variable cible", data.columns, index=None)
        proptest = st.slider("Proportion de l'échantillon de test", min_value=0.1, max_value=1.0, step=0.1, value= 0.3) 
        maxIter = st.slider("Itération maximale", min_value=100, max_value=4000, step=50, value=100)
        tol = float(st.text_input("Tolérance aux critères d'arrêt", value="0.0001"))
        solver = st.selectbox("Algorithme à utiliser dans le problème d'optimisation", ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                index=0)

    st.subheader("Avant le suréchantillonage")

    if  cible is not None :
        fig = px.bar(x=data[cible].value_counts().index, y=data[cible].value_counts().values, barmode="group",
                title="Class: Abonner ou non?")
        fig.update_xaxes(title=cible)
        fig.update_yaxes(title="count")
        st.plotly_chart(fig)
        target = data[cible]
    #    st.session_state.target = target
        oversampling = st.checkbox("Effectuer le suréchatillonage") 
        if oversampling :
            sm = SMOTE(random_state=2)
            X_resampled, y_resampled = sm.fit_resample(dummy, target)
            X_resampled = pd.DataFrame(X_resampled, columns=dummy.columns)
            st.write(y_resampled.value_counts())
            st.write("Dimension de dataframe ", dummy.shape)
            st.write("Dimension de X_resampled ",X_resampled.shape)
            st.write("Dimension de y_resampled ", y_resampled.shape)
            fig =px.bar(x=y_resampled.value_counts().index, y=y_resampled.value_counts().values, barmode="group", 
                        title="Class: Abonner ou non?")
            fig.update_xaxes(title=cible)
            fig.update_yaxes(title="count")
            st.plotly_chart(fig)
    
            X_train, X_test, y_train, y_test = splitData(X_resampled, y_resampled, proptest)
            model = logisticModel(X_resampled, y_resampled, maxIter, tol, solver)
            st.session_state.model = model
        else :
            X_train, X_test, y_train, y_test = splitData(dummy, target, proptest)
            model = logisticModel(dummy, target, maxIter, tol, solver)   
            st.session_state.model = model

        st.write('Training score =', model.score(X_train, y_train))
        st.write('Test score =', model.score(X_test, y_test))

        # Matrice de confusion
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax)
        plt.title("Matrice de Confusion")
        st.pyplot(fig=plt)
        st.write('Accuracy =',(cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]))
        st.write('Precision =', precision_score(y_test, y_pred, average=None))
        st.write('Recall =', recall_score(y_test, y_pred, average=None))
    else :
        st.warning("Merci de sélectionner une variable cible")    
    
  

if selected == "Prédiction" :
    if "dataFrame" not in st.session_state :
        st.warning("Merci de télécharger un fichier de données dans la rubrique 'Lecture des données'")
    elif "dummy" not in st.session_state :
        st.warning("Merci d'effecuter Dummy encoding dans la rubrique 'p-value et facteur d'inflation de variance (VIF)'")
    elif "model" not in st.session_state :
        st.warning("Merci d'effecuter la régression dans la rubrique 'Régression'")
    else :
        data = st.session_state.dataFrame
        dummy = st.session_state.dummy
        model = st.session_state.model
        
        with st.sidebar :
            cible = st.selectbox("Variable cible", data.columns, index=None)

        rdb = st.radio(":blue[**Variables expliquantes**]", ["Entrer", "Importer"]
            , captions=[":keyboard:"," un fichier CSV :file_folder:"])
        if rdb == "Entrer" :
            st.header("Entrer les variables expliquantes")
            # Array des variables expliquantes
            vals = []
            st.dataframe(dummy.dtypes)
        #    dummy.dtypes.age
            col1, col2 = st.columns(2)
            for i,col in enumerate(dummy.columns) :
                if i%2 == 0 :
                    with col1 :
                        val = st.number_input(col, value=None)
                        val = convertType(dummy[col], val)
                        vals.append(val)
                else :
                    with col2 :
                        val = st.number_input(col, value=None)
                        val = convertType(dummy[col], val)
                        vals.append(val)
        
            if st.button("Prédict", type="primary") :
        #       st.write(vals)
                valid = True
                for i,val in enumerate(vals) :
                    if val is None :
                        valid = False
                        break
                #     st.write(dummy[i], val)
                
                if valid :
                    vals = np.array(vals)
                    matrix = vals.reshape(1,len(vals))
                    y_pred = model.predict(matrix) 
                    prob_pred = model.predict_proba(matrix)
                    st.markdown(f"**{cible}**" )
                    st.write(y_pred)
                    st.markdown(f"**Probabilité**")
                    st.write(prob_pred)
                
                else :               
                    st.error("Merci de remplir tous les champs de variable.")
        elif rdb == "Importer" :
            col3, col4 = st.columns(2)
            with col3 :
                delimiter = st.text_input("Entrer un délimiter pour un fichier CSV", placeholder = "Par exemple ; ou , ou | ")
            with col4 :
                encode = st.selectbox("Sélectionner un codec :", codecsArray, index = 95)
            uploaded_file = st.file_uploader("Télécharger le fichier ici", type=["csv"])
            if uploaded_file is not None :
                if delimiter == "" :
                    # delimiter par défault = ";"
                    delimiter = ";"
                data_pred = load_data(uploaded_file, delimiter, encode) 
                st.dataframe(data_pred)
                predict = st.button("Predict", type="primary")
                if predict :
                    y_pred = model.predict(data_pred) 
                    prob_pred = model.predict_proba(data_pred)
                    st.markdown(f"**{cible}**" )
                    st.write(y_pred)
                    st.markdown(f"**Probabilité**")
                    st.write(prob_pred)
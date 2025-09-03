# ===============================
# APLICAÇÃO STREAMLIT - ANÁLISE DE SENTIMENTOS
# ===============================
# Desenvolvido para análise de sentimentos em reviews do IMDB
# Tecnologias: Streamlit + Scikit-learn + NLTK + Plotly

# ===============================
# app.py - APLICAÇÃO COMPLETA EM UM ARQUIVO
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Configuração da página
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sentiment-positive {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sentiment-negative {
        background: linear-gradient(90deg, #F44336, #FF5722);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #F44336; font-weight: bold; }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# CONFIGURAÇÃO DO GOOGLE DRIVE
# ===============================

# SUBSTITUA ESTE ID PELO SEU ARQUIVO NO GOOGLE DRIVE
GOOGLE_DRIVE_FILE_ID = "1I4HfIlSV7MaZSP0GChXCa1oKSNgrKHME"  # Substitua pelo ID do seu arquivo
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# ===============================
# FUNÇÕES AUXILIARES
# ===============================

@st.cache_data
def setup_nltk():
    """Download dos recursos necessários do NLTK"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("Baixando recursos do NLTK..."):
            nltk.download('stopwords', quiet=True)
    return True

@st.cache_data
def load_dataset_from_drive():
    """Carrega o dataset diretamente do Google Drive"""
    try:
        with st.spinner("Carregando dataset do Google Drive..."):
            df = pd.read_csv(GOOGLE_DRIVE_URL)
            return df
    except Exception as e:
        st.error(f"Erro ao carregar dataset do Google Drive: {e}")
        st.info("Verifique se o arquivo está público e o ID está correto.")
        return None

@st.cache_data
def get_contractions():
    """Retorna dicionário de contrações"""
    return {
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
        "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mustn't": "must not",
        "needn't": "need not", "daren't": "dare not", "mayn't": "may not", "shan't": "shall not",
        "might've": "might have", "could've": "could have", "would've": "would have", 
        "should've": "should have", "must've": "must have", "it's": "it is", "he's": "he is",
        "she's": "she is", "that's": "that is", "what's": "what is", "where's": "where is",
        "how's": "how is", "i'm": "i am", "you're": "you are", "we're": "we are",
        "they're": "they are", "i've": "i have", "you've": "you have", "we've": "we have",
        "they've": "they have", "i'd": "i would", "you'd": "you would", "he'd": "he would",
        "she'd": "she would", "we'd": "we would", "they'd": "they would", "i'll": "i will",
        "you'll": "you will", "he'll": "he will", "she'll": "she will", "we'll": "we will",
        "they'll": "they will"
    }

@st.cache_data
def load_stopwords():
    """Carrega stopwords do NLTK"""
    setup_nltk()
    return set(stopwords.words('english'))

# ===============================
# CLASSE PARA PRÉ-PROCESSAMENTO
# ===============================

class TextPreprocessor:
    """Classe para pré-processamento de texto que pode ser serializada"""
    
    def __init__(self):
        self.stop_words = load_stopwords()
        self.stemmer = PorterStemmer()
        self.contractions = get_contractions()
    
    def expand_contractions(self, text):
        """Expande contrações no texto"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def preprocess_text(self, text):
        """Pré-processa o texto"""
        text = text.lower()
        text = self.expand_contractions(text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        words = []
        for word in text.split():
            if word not in self.stop_words and len(word) > 2:
                words.append(self.stemmer.stem(word))
        
        return " ".join(words)

def load_model_components():
    """Carrega todos os componentes do modelo"""
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        preprocessor = TextPreprocessor()
        return model, vectorizer, preprocessor
    except FileNotFoundError as e:
        st.error(f"Arquivo do modelo não encontrado: {e}")
        return None, None, None

def predict_sentiment(text, model, vectorizer, preprocessor):
    """Prediz sentimento de um texto"""
    clean_text = preprocessor.preprocess_text(text)
    
    if not clean_text:
        return "neutral", 0.5, clean_text
    
    vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = max(probabilities)
    
    return sentiment, confidence, clean_text

def train_sentiment_model():
    """Treina o modelo de análise de sentimentos"""
    
    # Carregar dataset do Google Drive
    df = load_dataset_from_drive()
    
    if df is None:
        return None
    
    # Verificar se tem as colunas necessárias
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        st.error("Dataset deve conter colunas 'review' e 'sentiment'")
        return None
    
    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Dataset carregado
    progress_bar.progress(10)
    st.success(f"Dataset carregado do Google Drive: {df.shape[0]:,} reviews")
    
    # Pré-processamento
    status_text.text("Aplicando pré-processamento...")
    preprocessor = TextPreprocessor()
    df["clean_review"] = df["review"].apply(preprocessor.preprocess_text)
    df = df[df["clean_review"].str.len() > 0]
    progress_bar.progress(30)
    
    st.info(f"Reviews após limpeza: {df.shape[0]:,}")
    
    # Vetorização
    status_text.text("Criando vetores TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(df["clean_review"])
    y = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    progress_bar.progress(50)
    
    st.info(f"Formato da matriz: {X.shape}")
    
    # Divisão dos dados
    status_text.text("Dividindo dados...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    progress_bar.progress(60)
    
    # Treinamento
    status_text.text("Treinando modelo...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    progress_bar.progress(80)
    
    # Validação cruzada
    status_text.text("Executando validação cruzada...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    progress_bar.progress(90)
    
    # Avaliação
    status_text.text("Avaliando modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Criar pasta models se não existir
    os.makedirs("models", exist_ok=True)
    
    # Salvar modelo e preprocessor
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    progress_bar.progress(100)
    status_text.text("Concluído!")
    
    # Retornar métricas
    return {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'model': model,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'feature_names': vectorizer.get_feature_names_out(),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# ===============================
# APLICAÇÃO PRINCIPAL
# ===============================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Análise de Sentimentos IMDB</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🛠️ Navegação")
    
    # Verificar se modelo existe
    model_exists = (
        os.path.exists("models/sentiment_model.pkl") and 
        os.path.exists("models/vectorizer.pkl")
    )
    
    if not model_exists:
        st.sidebar.warning("⚠️ Modelo não encontrado!")
    else:
        st.sidebar.success("✅ Modelo pronto!")
        
    # Menu principal
    menu = st.sidebar.selectbox(
        "Selecione uma opção:",
        ["🏠 Início", "🤖 Treinar Modelo", "🔮 Predição", "📊 Dataset", "📈 Métricas"]
    )
    
    # ===============================
    # PÁGINA INICIAL
    # ===============================
    if menu == "🏠 Início":
        st.markdown("""
        ## Bem-vindo ao Analisador de Sentimentos! 🎬
        
        Esta aplicação utiliza **Machine Learning** para analisar sentimentos em reviews de filmes do IMDB.
        
        ### 🚀 Funcionalidades:
        - **Treinar Modelo**: Treine um novo modelo com o dataset IMDB do Google Drive
        - **Predição**: Analise o sentimento de qualquer texto em tempo real
        - **Dataset**: Explore os dados estatisticamente
        - **Métricas**: Visualize a performance do modelo
        
        ### 📋 Como começar:
        1. **Configure o ID do arquivo no Google Drive no código**
        2. Use o menu lateral para navegar
        3. Treine o modelo com dados da nuvem
        4. Faça predições instantâneas!
        
        ### 🔧 Tecnologias utilizadas:
        - **Streamlit**: Interface web interativa
        - **Scikit-learn**: Algoritmos de machine learning
        - **NLTK**: Processamento de linguagem natural
        - **Plotly**: Visualizações interativas
        - **Google Drive**: Armazenamento do dataset na nuvem
        - **TF-IDF + Logistic Regression**: Modelo de classificação
        """)
        
        # Status do sistema
        st.markdown("### 📊 Status do Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Testar conexão com Google Drive
            test_df = load_dataset_from_drive()
            dataset_ok = test_df is not None
            if dataset_ok:
                st.markdown('<div class="metric-card">✅<br><strong>Google Drive</strong><br>Conectado</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">❌<br><strong>Google Drive</strong><br>Erro de conexão</div>', unsafe_allow_html=True)
        
        with col2:
            if model_exists:
                st.markdown('<div class="metric-card">✅<br><strong>Modelo</strong><br>Treinado</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">❌<br><strong>Modelo</strong><br>Não treinado</div>', unsafe_allow_html=True)
        
        with col3:
            if dataset_ok and model_exists:
                st.markdown('<div class="metric-card">✅<br><strong>Sistema</strong><br>Pronto</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">⚠️<br><strong>Sistema</strong><br>Configuração necessária</div>', unsafe_allow_html=True)
        
        # Instruções de configuração
        st.markdown("### ⚙️ Configuração do Google Drive")
        
        with st.expander("📁 Como configurar o Google Drive"):
            st.markdown("""
            **Passo 1: Fazer upload do dataset**
            1. Suba seu arquivo `IMDB Dataset.csv` para o Google Drive
            2. Clique com o botão direito → **Compartilhar**
            3. Altere para "Qualquer pessoa com o link pode visualizar"
            4. Copie o link compartilhado
            
            **Passo 2: Extrair o ID do arquivo**
            Do link: `https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing`
            O ID é: `1ABC123XYZ`
            
            **Passo 3: Atualizar o código**
            Substitua a variável `GOOGLE_DRIVE_FILE_ID` no código pelo seu ID.
            """)
        
        # Status da conexão
        if not dataset_ok:
            st.error("❌ Não foi possível conectar ao Google Drive. Verifique o ID do arquivo!")
        
        if not model_exists and dataset_ok:
            st.info("🤖 Dataset conectado! Vá para 'Treinar Modelo' para treinar seu primeiro modelo!")
    
    # ===============================
    # TREINAR MODELO
    # ===============================
    elif menu == "🤖 Treinar Modelo":
        st.header("🤖 Treinamento do Modelo")
        
        st.markdown("""
        ### 📚 Sobre o Treinamento
        
        O modelo utiliza:
        - **Dataset do Google Drive**: Carregamento automático da nuvem
        - **TF-IDF Vectorizer**: Converte texto em números
        - **Logistic Regression**: Algoritmo de classificação
        - **Cross-validation**: Validação robusta
        - **Pré-processamento avançado**: Limpeza de texto
        """)
        
        # Testar conexão primeiro
        test_df = load_dataset_from_drive()
        if test_df is None:
            st.error("❌ Erro ao conectar com Google Drive! Verifique o ID do arquivo no código.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🚀 Iniciar Treinamento", type="primary", use_container_width=True):
                st.session_state.training = True
        
        with col2:
            if model_exists:
                st.success("✅ Modelo já existe! Treinar novamente irá sobrescrever.")
            else:
                st.info("ℹ️ Nenhum modelo encontrado. Treine para criar o primeiro!")
        
        if st.session_state.get('training', False):
            with st.container():
                results = train_sentiment_model()
                
                if results:
                    st.balloons()
                    st.success("🎉 Modelo treinado com sucesso!")
                    
                    # Reset training state
                    st.session_state.training = False
                    
                    # Métricas principais
                    st.markdown("### 📊 Resultados do Treinamento")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 Acurácia", f"{results['accuracy']:.1%}")
                    
                    with col2:
                        st.metric("🔄 CV Score", f"{results['cv_mean']:.1%}")
                    
                    with col3:
                        st.metric("📏 Features", f"{len(results['feature_names']):,}")
                    
                    with col4:
                        st.metric("⏱️ CV Desvio", f"{results['cv_std']:.3f}")
                    
                    # Matriz de confusão
                    st.markdown("### 🔍 Matriz de Confusão")
                    
                    cm = results['confusion_matrix']
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predito", y="Real", color="Quantidade"),
                        x=['Negativo', 'Positivo'],
                        y=['Negativo', 'Positivo'],
                        color_continuous_scale='Blues',
                        aspect="auto",
                        text_auto=True
                    )
                    
                    fig.update_layout(
                        title="Matriz de Confusão - Performance do Modelo",
                        width=500,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métricas detalhadas
                    with st.expander("📈 Relatório Detalhado"):
                        report = results['classification_report']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Classe NEGATIVA:**")
                            st.write(f"- Precision: {report['0']['precision']:.3f}")
                            st.write(f"- Recall: {report['0']['recall']:.3f}")
                            st.write(f"- F1-Score: {report['0']['f1-score']:.3f}")
                        
                        with col2:
                            st.markdown("**Classe POSITIVA:**")
                            st.write(f"- Precision: {report['1']['precision']:.3f}")
                            st.write(f"- Recall: {report['1']['recall']:.3f}")
                            st.write(f"- F1-Score: {report['1']['f1-score']:.3f}")
    
    # ===============================
    # FAZER PREDIÇÃO
    # ===============================
    elif menu == "🔮 Predição":
        st.header("🔮 Análise de Sentimento em Tempo Real")
        
        if not model_exists:
            st.error("❌ Modelo não encontrado! Treine o modelo primeiro na seção '🤖 Treinar Modelo'.")
            return
        
        try:
            # Carregar modelo
            if 'model_components' not in st.session_state:
                with st.spinner("Carregando modelo..."):
                    st.session_state.model_components = load_model_components()
            
            model, vectorizer, preprocessor = st.session_state.model_components
            
            if model is None:
                st.error("Erro ao carregar o modelo!")
                return
            
            # Interface de predição
            st.markdown("### ✍️ Digite seu texto para análise:")
            
            # Exemplos pré-definidos
            st.markdown("**💡 Ou escolha um exemplo:**")
            
            examples = [
                "This movie is absolutely amazing! Best film I've ever watched. The acting was superb and the plot was incredible.",
                "Terrible movie. Complete waste of time and money. I want my 2 hours back.",
                "The movie was okay, nothing special but not bad either. Average entertainment.",
                "Outstanding performance by the actors. Highly recommended! A masterpiece of cinema.",
                "Boring and predictable plot. Very disappointed with this overhyped film.",
                "One of the worst movies I have ever seen. Terrible acting and horrible script.",
                "Fantastic storyline with amazing visual effects. Loved every minute of it!"
            ]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_example = st.selectbox("Escolha um exemplo:", ["Escrever próprio texto..."] + examples)
            
            with col2:
                if st.button("🎲 Exemplo Aleatório"):
                    selected_example = np.random.choice(examples)
                    st.rerun()
            
            # Área de texto
            if selected_example == "Escrever próprio texto...":
                default_text = ""
            else:
                default_text = selected_example
            
            user_text = st.text_area(
                "**Texto para análise:**",
                value=default_text,
                height=120,
                placeholder="Digite aqui seu review de filme, opinião sobre produto, ou qualquer texto em inglês...",
                help="O modelo foi treinado em reviews de filmes em inglês, mas funciona com outros tipos de texto também."
            )
            
            if user_text and len(user_text.strip()) > 5:
                # Fazer predição
                sentiment, confidence, clean_text = predict_sentiment(
                    user_text, model, vectorizer, preprocessor
                )
                
                # Layout dos resultados
                st.markdown("### 🎯 Resultado da Análise:")
                
                # Resultado principal
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if sentiment == "positive":
                        st.markdown(f"""
                        <div class="sentiment-positive">
                            <h2>😊 SENTIMENTO POSITIVO</h2>
                            <p style="font-size: 1.2em; margin: 0;">Confiança: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="sentiment-negative">
                            <h2>😔 SENTIMENTO NEGATIVO</h2>
                            <p style="font-size: 1.2em; margin: 0;">Confiança: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Gauge de confiança
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confiança %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#4CAF50" if sentiment == "positive" else "#F44336"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Barra de progresso para confiança
                st.markdown("### 📊 Nível de Confiança:")
                
                confidence_class = (
                    "confidence-high" if confidence > 0.8 else
                    "confidence-medium" if confidence > 0.6 else
                    "confidence-low"
                )
                
                progress_col1, progress_col2 = st.columns([3, 1])
                
                with progress_col1:
                    st.progress(confidence)
                
                with progress_col2:
                    if confidence > 0.8:
                        st.markdown("🟢 **Alta**")
                    elif confidence > 0.6:
                        st.markdown("🟡 **Média**")
                    else:
                        st.markdown("🔴 **Baixa**")
                
                # Interpretação
                st.markdown("### 🧠 Interpretação:")
                
                if confidence > 0.8:
                    st.success(f"✅ O modelo está **muito confiante** que este texto expressa sentimento **{sentiment.upper()}**.")
                elif confidence > 0.6:
                    st.info(f"ℹ️ O modelo tem **confiança moderada** que este texto expressa sentimento **{sentiment.upper()}**.")
                else:
                    st.warning(f"⚠️ O modelo tem **baixa confiança** nesta predição. O texto pode ser **ambíguo** ou **neutro**.")
                
                # Detalhes técnicos
                with st.expander("🔍 Detalhes Técnicos"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📝 Texto Original:**")
                        st.text_area("", value=user_text, height=100, disabled=True)
                    
                    with col2:
                        st.markdown("**🧹 Texto Processado:**")
                        st.text_area("", value=clean_text if clean_text else "Texto muito curto após processamento", height=100, disabled=True)
                    
                    st.markdown(f"**📊 Probabilidades:**")
                    st.write(f"- Negativo: {(1-confidence if sentiment == 'negative' else confidence):.1%}")
                    st.write(f"- Positivo: {(confidence if sentiment == 'positive' else 1-confidence):.1%}")
            
            elif user_text and len(user_text.strip()) <= 5:
                st.warning("⚠️ Texto muito curto! Digite pelo menos algumas palavras para uma análise confiável.")
            
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
    
    # ===============================
    # ANÁLISE DO DATASET
    # ===============================
    elif menu == "📊 Dataset":
        st.header("📊 Análise Exploratória do Dataset IMDB")
        
        # Carregar dataset do Google Drive
        if 'dataset' not in st.session_state:
            st.session_state.dataset = load_dataset_from_drive()
        
        df = st.session_state.dataset
        
        if df is None:
            st.error("❌ Erro ao carregar dataset do Google Drive! Verifique a configuração.")
            return
        
        st.success(f"✅ Dataset carregado do Google Drive: {df.shape[0]:,} reviews")
        
        # Estatísticas básicas
        st.markdown("### 📈 Visão Geral")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Reviews", f"{len(df):,}")
        
        with col2:
            positive_count = len(df[df['sentiment'] == 'positive'])
            st.metric("😊 Positivos", f"{positive_count:,}")
        
        with col3:
            negative_count = len(df[df['sentiment'] == 'negative'])
            st.metric("😔 Negativos", f"{negative_count:,}")
        
        with col4:
            avg_length = df['review'].str.len().mean()
            st.metric("📏 Tamanho Médio", f"{avg_length:.0f} chars")
        
        # Distribuição de sentimentos
        st.markdown("### 🥧 Distribuição de Sentimentos")
        
        sentiment_counts = df['sentiment'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Proporção de Sentimentos",
                color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336'},
                hole=0.4
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Quantidade por Sentimento",
                color=sentiment_counts.index,
                color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336'},
                labels={'x': 'Sentimento', 'y': 'Quantidade'}
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Distribuição do tamanho dos textos
        st.markdown("### 📏 Análise do Tamanho dos Reviews")
        
        df['review_length'] = df['review'].str.len()
        df['word_count'] = df['review'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist1 = px.histogram(
                df,
                x='review_length',
                color='sentiment',
                nbins=50,
                title="Distribuição por Caracteres",
                labels={'review_length': 'Número de Caracteres', 'count': 'Quantidade'},
                color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336'},
                opacity=0.7
            )
            fig_hist1.update_layout(height=400)
            st.plotly_chart(fig_hist1, use_container_width=True)
        
        with col2:
            fig_hist2 = px.histogram(
                df,
                x='word_count',
                color='sentiment',
                nbins=50,
                title="Distribuição por Palavras",
                labels={'word_count': 'Número de Palavras', 'count': 'Quantidade'},
                color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336'},
                opacity=0.7
            )
            fig_hist2.update_layout(height=400)
            st.plotly_chart(fig_hist2, use_container_width=True)
        
        # Estatísticas de tamanho
        st.markdown("### 📊 Estatísticas Detalhadas")
        
        stats_df = df.groupby('sentiment').agg({
            'review_length': ['mean', 'median', 'std', 'min', 'max'],
            'word_count': ['mean', 'median', 'std', 'min', 'max']
        }).round(2)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Top palavras mais comuns
        st.markdown("### 🔤 Análise de Palavras")
        
        if st.button("🔍 Analisar Palavras Mais Comuns"):
            with st.spinner("Processando texto..."):
                # Processamento básico para análise de palavras
                import re
                from collections import Counter
                
                def simple_preprocess(text):
                    text = re.sub(r'<.*?>', '', text.lower())
                    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                    words = [word for word in text.split() if len(word) > 3]
                    return words
                
                # Análise por sentimento
                positive_words = []
                negative_words = []
                
                for idx, row in df.sample(1000).iterrows():  # Amostra para performance
                    words = simple_preprocess(row['review'])
                    if row['sentiment'] == 'positive':
                        positive_words.extend(words)
                    else:
                        negative_words.extend(words)
                
                # Top palavras
                pos_common = Counter(positive_words).most_common(15)
                neg_common = Counter(negative_words).most_common(15)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Top Palavras - Reviews Positivos**")
                    pos_df = pd.DataFrame(pos_common, columns=['Palavra', 'Frequência'])
                    
                    fig_pos = px.bar(
                        pos_df,
                        y='Palavra',
                        x='Frequência',
                        orientation='h',
                        title="Palavras Mais Comuns - Positivo",
                        color_discrete_sequence=['#4CAF50']
                    )
                    fig_pos.update_layout(height=500)
                    st.plotly_chart(fig_pos, use_container_width=True)
                
                with col2:
                    st.markdown("**🔴 Top Palavras - Reviews Negativos**")
                    neg_df = pd.DataFrame(neg_common, columns=['Palavra', 'Frequência'])
                    
                    fig_neg = px.bar(
                        neg_df,
                        y='Palavra',
                        x='Frequência',
                        orientation='h',
                        title="Palavras Mais Comuns - Negativo",
                        color_discrete_sequence=['#F44336']
                    )
                    fig_neg.update_layout(height=500)
                    st.plotly_chart(fig_neg, use_container_width=True)
        
        # Amostra do dataset
        st.markdown("### 👀 Amostra dos Dados")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sample_size = st.slider("Número de amostras:", 5, 50, 10)
            sentiment_filter = st.selectbox("Filtrar por:", ["Todos", "positive", "negative"])
        
        with col2:
            if sentiment_filter == "Todos":
                sample_df = df.sample(sample_size)
            else:
                sample_df = df[df['sentiment'] == sentiment_filter].sample(min(sample_size, len(df[df['sentiment'] == sentiment_filter])))
            
            # Adicionar colunas de estatísticas
            sample_df_display = sample_df.copy()
            sample_df_display['chars'] = sample_df_display['review'].str.len()
            sample_df_display['words'] = sample_df_display['review'].str.split().str.len()
            sample_df_display['preview'] = sample_df_display['review'].str[:100] + "..."
            
            # Mostrar apenas colunas relevantes
            display_cols = ['sentiment', 'chars', 'words', 'preview']
            st.dataframe(sample_df_display[display_cols], use_container_width=True, height=400)
    
    # ===============================
    # MÉTRICAS DO MODELO
    # ===============================
    elif menu == "📈 Métricas":
        st.header("📈 Performance e Análise do Modelo")
        
        if not model_exists:
            st.error("❌ Modelo não encontrado! Treine o modelo primeiro.")
            return
        
        try:
            # Carregar modelo
            if 'model_components' not in st.session_state:
                with st.spinner("Carregando modelo..."):
                    st.session_state.model_components = load_model_components()
            
            model, vectorizer, preprocessor = st.session_state.model_components
            
            if model is None:
                st.error("Erro ao carregar o modelo!")
                return
            
            st.success("✅ Modelo carregado com sucesso!")
            
            # Informações básicas do modelo
            st.markdown("### ℹ️ Informações do Modelo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🤖 Algoritmo", "Logistic Regression")
            
            with col2:
                feature_count = len(vectorizer.get_feature_names_out())
                st.metric("🔢 Features", f"{feature_count:,}")
            
            with col3:
                st.metric("📊 Vetorização", "TF-IDF")
            
            with col4:
                st.metric("🎯 Classes", "2 (Pos/Neg)")
            
            # Parâmetros do modelo
            with st.expander("⚙️ Parâmetros do Modelo"):
                st.json({
                    "Logistic Regression": {
                        "max_iter": model.max_iter,
                        "C": model.C,
                        "solver": model.solver,
                        "random_state": model.random_state
                    },
                    "TF-IDF Vectorizer": {
                        "max_features": vectorizer.max_features,
                        "min_df": vectorizer.min_df,
                        "max_df": vectorizer.max_df,
                        "ngram_range": vectorizer.ngram_range
                    }
                })
            
            # Análise das features mais importantes
            st.markdown("### 🔝 Features Mais Influentes")
            
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            
            # Calcular importâncias
            n_top = st.slider("Número de palavras para mostrar:", 10, 50, 20)
            
            # Top positivas
            positive_indices = np.argsort(coefficients)[-n_top:]
            positive_words = [(feature_names[idx], coefficients[idx]) for idx in reversed(positive_indices)]
            
            # Top negativas
            negative_indices = np.argsort(coefficients)[:n_top]
            negative_words = [(feature_names[idx], coefficients[idx]) for idx in negative_indices]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Palavras Mais POSITIVAS**")
                pos_df = pd.DataFrame(positive_words, columns=['Palavra', 'Coeficiente'])
                pos_df['Coeficiente'] = pos_df['Coeficiente'].round(4)
                st.dataframe(pos_df, use_container_width=True, height=400)
            
            with col2:
                st.markdown("**🔴 Palavras Mais NEGATIVAS**")
                neg_df = pd.DataFrame(negative_words, columns=['Palavra', 'Coeficiente'])
                neg_df['Coeficiente'] = neg_df['Coeficiente'].round(4)
                st.dataframe(neg_df, use_container_width=True, height=400)
            
            # Gráfico de coeficientes
            st.markdown("### 📊 Visualização dos Coeficientes")
            
            # Preparar dados para o gráfico
            top_features = negative_words[:15] + positive_words[:15]
            words, coefs = zip(*top_features)
            colors = ['red'] * 15 + ['green'] * 15
            
            fig = go.Figure([go.Bar(
                x=list(coefs),
                y=list(words),
                orientation='h',
                marker_color=colors,
                text=[f'{coef:.3f}' for coef in coefs],
                textposition='auto',
            )])
            
            fig.update_layout(
                title=f"Top {n_top} Features Mais Influentes",
                xaxis_title="Coeficiente",
                yaxis_title="Palavras",
                height=600,
                showlegend=False
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise de distribuição dos coeficientes
            st.markdown("### 📈 Distribuição dos Coeficientes")
            
            fig_dist = px.histogram(
                x=coefficients,
                nbins=50,
                title="Distribuição de Todos os Coeficientes",
                labels={'x': 'Valor do Coeficiente', 'count': 'Frequência'}
            )
            
            fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutro")
            fig_dist.update_layout(height=400)
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Estatísticas dos coeficientes
            st.markdown("### 📊 Estatísticas dos Coeficientes")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 Média", f"{np.mean(coefficients):.4f}")
            
            with col2:
                st.metric("📐 Desvio Padrão", f"{np.std(coefficients):.4f}")
            
            with col3:
                st.metric("⬇️ Mínimo", f"{np.min(coefficients):.4f}")
            
            with col4:
                st.metric("⬆️ Máximo", f"{np.max(coefficients):.4f}")
            
            # Teste interativo do modelo
            st.markdown("### 🧪 Teste Rápido do Modelo")
            
            test_examples = [
                "excellent movie wonderful acting",
                "terrible boring waste time",
                "good film recommend",
                "worst movie ever horrible",
                "amazing beautiful masterpiece"
            ]
            
            selected_test = st.selectbox("Escolha um teste rápido:", test_examples)
            
            if st.button("🔍 Testar"):
                sentiment, confidence, clean_text = predict_sentiment(
                    selected_test, model, vectorizer, preprocessor
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentimento", sentiment.upper())
                
                with col2:
                    st.metric("Confiança", f"{confidence:.1%}")
                
                with col3:
                    color = "🟢" if sentiment == "positive" else "🔴"
                    st.metric("Resultado", f"{color} {sentiment}")
                
                st.text(f"Texto processado: {clean_text}")
            
        except Exception as e:
            st.error(f"Erro ao analisar o modelo: {e}")
            st.exception(e)

# ===============================
# RODAPÉ E CONFIGURAÇÕES FINAIS
# ===============================

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎭 <strong>Analisador de Sentimentos IMDB</strong></p>
        <p>Desenvolvido com Streamlit • Machine Learning • Python • Google Drive</p>
        <p><em>Versão 3.0 - Sistema Cloud-First</em></p>
        <p>📊 Dataset: <a href="https://drive.google.com/uc?id=1I4HfIlSV7MaZSP0GChXCa1oKSNgrKHME" target="_blank">IMDB Reviews no Google Drive</a></p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# EXECUÇÃO PRINCIPAL
# ===============================

if __name__ == "__main__":
    # Inicializar session state
    if 'training' not in st.session_state:
        st.session_state.training = False
    
    # Executar aplicação principal
    try:
        main()
        show_footer()
    except Exception as e:
        st.error("Erro na aplicação:")
        st.exception(e)
        st.info("Tente recarregar a página ou verificar a conexão com o Google Drive.")
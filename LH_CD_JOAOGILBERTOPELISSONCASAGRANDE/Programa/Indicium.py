# Desafio Indicium - Ciencia de Dados
# Autor: Joao Gilberto Pelisson Casagrande
# =======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregando a tabela de dados do desafio
df = pd.read_csv("desafio_indicium_imdb.csv")

# Criacao da variavel Director_Competence
director_comp = df.groupby("Director").agg({
    "IMDB_Rating": "mean",
    "Meta_score": "mean",
    "Series_Title": "count"
}).reset_index()

# Normalizar Meta_score (0 a 100 para 0 a 10)
director_comp["Meta_score_norm"] = director_comp["Meta_score"] / 10

# Calculo da competencia
director_comp["Director_Competence"] = (
    director_comp["IMDB_Rating"] + director_comp["Meta_score_norm"]) / 2

# Renomear coluna de contagem
director_comp = director_comp.rename(columns={"Series_Title": "Num_Films"})

# Substituir NaN de competencia pela media geral
media_competencia = director_comp["Director_Competence"].mean()
director_comp["Director_Competence"] = director_comp["Director_Competence"].fillna(
    media_competencia)

# Juntar no dataframe principal
df = df.merge(
    director_comp[["Director", "Director_Competence"]], on="Director", how="left")

# =======================================

print(df.head())
print(df.info())
print(df.describe())

# Matriz de correlacao

# Ajustar Gross para numero (remover virgulas e converter)
df["Gross"] = df["Gross"].str.replace(",", "").astype(float)

# Criacao de matrizes de correlacao para cada classificacao indicativa
certificados = df["Certificate"].dropna().unique()

for cert in certificados:
    subset = df[df["Certificate"] == cert][["IMDB_Rating", "Gross"]].dropna()

    if subset.shape[0] > 1:
        plt.figure(figsize=(5, 4))
        corr_matrix = subset.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Matriz de Correlacao - Certificate: {cert}")
        plt.show()

        print(
            f"Certificate {cert} - Correlacao IMDB_Rating x Gross: {corr_matrix.loc['IMDB_Rating', 'Gross']:.2f}")

plt.figure(figsize=(10, 6))
sns.heatmap(df[["IMDB_Rating", "Meta_score", "No_of_Votes",
            "Director_Competence"]].corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlacao")
plt.show()

# Relacao entre Meta_score e IMDB_Rating
plt.scatter(df["Meta_score"], df["IMDB_Rating"])
plt.xlabel("Meta_score")
plt.ylabel("IMDB Rating")
plt.title("Relacao entre Meta_score e IMDB_Rating")
plt.show()

# Relacao entre competencia do diretor e IMDB_Rating
plt.scatter(df["Director_Competence"], df["IMDB_Rating"])
plt.xlabel("Director_Competence")
plt.ylabel("IMDB Rating")
plt.title("Relacao entre Director_Competence e IMDB_Rating")
plt.show()

# =======================================

# Comeco com Pre-processamento
df_clean = df[["Meta_score", "No_of_Votes",
               "Director_Competence", "IMDB_Rating"]].dropna()

X = df_clean[["Meta_score", "No_of_Votes", "Director_Competence"]]
y = df_clean["IMDB_Rating"]

# Separacao treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Criacao e treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliacao do modelo
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R²:", r2_score(y_test, y_pred))

# =======================================

# Exemplo de previsao com The Shawshank Redemption
exemplo = pd.DataFrame({
    "Meta_score": [80],
    "No_of_Votes": [2343110],
    "Director_Competence": df[df["Director"] == "Frank Darabont"]["Director_Competence"].iloc[0]
})

nota_prevista = model.predict(exemplo)[0]
print(f"Nota prevista para 'The Shawshank Redemption': {nota_prevista:.2f}")

# =======================================

# Ranking de diretores por competencia
plt.figure(figsize=(12, 8))
ranking = director_comp.sort_values(by="Director_Competence", ascending=False)

# Filtrar apenas os 20 melhores
top20 = ranking.head(20)

sns.barplot(
    data=top20,
    y="Director",
    x="Director_Competence",
    palette="viridis"
)

plt.title("Top 20 Diretores por Competencia (IMDB + Meta_score)", fontsize=14)
plt.xlabel("Director_Competence (0-10)")
plt.ylabel("Diretor")
plt.show()

# =======================================

# Recomendacao baseada em pontuacao continua (IMDB + No_of_Votes)
df["Pontuation"] = (df["No_of_Votes"] / 200000).clip(lower=1, upper=10)

df["Recommended"] = (df["IMDB_Rating"] + df["Pontuation"]) / 2

top20_recommended = df.sort_values("Recommended", ascending=False).head(20)[
    ["Series_Title", "IMDB_Rating", "No_of_Votes", "Pontuation", "Recommended"]
]

print("Top 20 filmes recomendados (continuo):")
print(top20_recommended)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=top20_recommended,
    y="Series_Title",
    x="Recommended",
    palette="Blues_d"
)

plt.title("Top 20 Filmes Recomendados", fontsize=16)
plt.xlabel("Pontuacao de Recomendacao")
plt.ylabel("Filme")
plt.show()

# =======================================

# Salvar o modelo treinado em arquivo .pkl
with open("modelo_imdb_joaogilberto.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo salvo como modelo_imdb_joaogilberto.pkl")

Desafio Indicium – Ciência de Dados
=======================================

Desenvolvido por João Gilberto Pelisson Casagrande 
=======================================

Descrição Básica:

Este projeto foi desenvolvido como parte de um desafio de ciência de dados, utilizando como base uma lista de filmes fornecida pela organização do desafio. 
O objetivo é analisar relações entre variáveis como notas, votos, bilheteria e competência dos diretores, além de treinar um modelo preditivo simples para estimar notas de filmes.

=======================================

Funcionalidades:

- Criação da variável `Director_Competence`
  - Métrica baseada na média de notas IMDB e Meta_score dos filmes de cada diretor.

- Análise Exploratória (EDA)
  - Estatísticas descritivas dos filmes.
  - Gráficos de correlação entre notas, votos, competência de diretores e Meta_score.
  - Estudo da relação IMDB x bilheteria (Gross) por classificação indicativa (Certificate).

- Modelagem
  - Regressão Linear para prever IMDB_Rating.
  - Avaliação do modelo com MAE, RMSE e R².
  - Exemplo de previsão com o filme 'The Shawshank Redemption'.
  - Modelo salvo em formato .pkl.

- Recomendações
  - Sistema simples que gera os 20 melhores filmes recomendados, combinando nota IMDB e um numero de pontuação feito a partir do número de votos.

=======================================

Resultados Obtidos:

- Correlação IMDB x Gross → fraca ou negativa, mostrando que notas não são bom preditor de bilheteria.
- R² ≈ 0.50 → modelo explica ~50% da variabilidade das notas.
- Erro médio (MAE ≈ 0.17) → previsões próximas das notas reais.
- Nota do filme exemplo 'The Shawshank Redemption' prevista como 8.83, próximo da nota real 9.3.
- Ranking de diretores → nomes como Orson Welles, Akira Kurosawa e Charles Chaplin apareceram no topo.
- Top 20 filmes recomendados → The Dark Knight, Inception, Fight Club, Pulp Fiction, Forrest Gump, The Godfather, entre outros.

=======================================

1. Primeiramente tenha um ambiente configurado para rodar códigos em Python 3.13.7

2. Depois baixe ou clone o repositório do link: https://github.com/Supremax/LH_CD_JOAOGILBERTOPELISSONCASAGRANDE.git

3. Criar e ativar ambiente virtual (opcional)
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows (cmd)

4. Instalar dependências
   pip install -r requirements.txt
   
   ou instale manualmente através do cmd:
   
   pip install pandas==2.3.2
   pip install numpy==2.3.2
   pip install matplotlib==3.10.6
   pip install seaborn==0.13.2
   pip install scikit-learn==1.7.1

6. Executar o programa
   python Programa/Indicium.py

=======================================

Estrutura do Projeto:

📦 LH_CD_JOAOGILBERTOPELISSONCASAGRANDE
 ┣ 📂 Programa
 ┃ ┗ Indicium.py               	    # Código principal
 ┃ ┗ desafio_indicium_imdb.csv   	  # Base de dados
 ┃ ┗ modelo_imdb_joaogilberto.pkl   # Modelo salvo
 ┃ ┗ requirements.txt          	    # Dependências do projeto
 ┗ Documentação do desafio Indicium de Ciência de Dados - João Gilberto Pelisson Casagrande           # Documentação do programa e respostas do desafio
 ┗ README.txt                       # Este arquivo

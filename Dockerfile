# Use uma imagem base oficial do Python
FROM python:3.8-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie os arquivos do projeto para o contêiner
COPY . .

# Instale as bibliotecas necessárias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que o Flask vai rodar
EXPOSE 5000

# Comando para iniciar o servidor Flask quando o contêiner é iniciado
CMD ["python", "app.py"]

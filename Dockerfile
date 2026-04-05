# ── Build .NET app ────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /repo
COPY . .
RUN dotnet publish backend/455Chapter17.API/455chapter17.API.csproj -c Release -o /app/out

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app

# Install Python + pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create venv and install scoring dependencies
RUN python3 -m venv /app/venv
RUN /app/venv/bin/pip install --no-cache-dir \
    "psycopg[binary]==3.2.12" \
    pandas==2.2.3 \
    SQLAlchemy==2.0.37 \
    scikit-learn==1.6.1 \
    joblib==1.4.2 \
    numpy==2.2.2

# Copy published .NET app
COPY --from=build /app/out .

# Copy Python scoring script and model
COPY scripts/run_fraud_scoring.py scripts/run_fraud_scoring.py
COPY crispdm-pipeline-model/fraud_model.sav crispdm-pipeline-model/fraud_model.sav

EXPOSE 8080
ENV ASPNETCORE_URLS=http://0.0.0.0:8080
ENTRYPOINT ["dotnet", "455chapter17.API.dll"]

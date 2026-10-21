# Previsibilidade do fim da estação chuvosa

Aplicação interativa em **Streamlit** para visualizar a previsão do fim
da estação chuvosa para a região de Monção Sul-Amerianala, compará-la com a data observada e examinar o erro
espacialmente.

## Funcionalidades

-   Seleção do ano de análise.
-   Busca e destaque de municípios nos três mapas.
-   Mapas de **Previsto**, **Observado** e **Erro**.
-   Seleção de uma célula da grade em qualquer mapa para consultar os
    valores previstos, observados e o erro.

## Dados utilizados

A aplicação lê os seguintes arquivos do repositório:

``` text
data/
└── processed/
    └── forecast/
        └── demise_forecast_1982-2026.nc

shapefiles/
├── BR_UF_2022_rct.shp
├── BR_UF_2022_rct.shx
├── BR_UF_2022_rct.dbf
├── BR_UF_2022_rct.prj
├── BR_UF_2022_rct.cpg
├── cidades_rct.shp
├── cidades_rct.shx
├── cidades_rct.dbf
├── cidades_rct.prj
└── cidades_rct.cpg
```

Os arquivos auxiliares de cada shapefile (`.shx`, `.dbf`, `.prj` e
`.cpg`) devem permanecer junto ao respectivo `.shp`.

O arquivo NetCDF contém as variáveis `demise_predicted`,
`demise_observed` e `error`, além das coordenadas espaciais e temporais
esperadas pelo aplicativo. O erro é definido como:

``` text
Erro = Previsto − Observado
```

Assim, erro positivo indica que a previsão aponta um fim da estação
chuvosa posterior ao observado; erro negativo indica uma previsão
anterior.

Os caminhos são definidos relativamente à raiz do projeto, permitindo
executar a aplicação localmente e hospedá-la sem depender de caminhos
absolutos da máquina do desenvolvedor.

## Executar localmente

Requisitos: Python e `pip`.

1.  Clone o repositório e entre na pasta do projeto:

    ``` bash
    git clone https://github.com/gmairink/dupla-safra.git
    cd dupla-safra
    ```

2.  Crie e ative um ambiente virtual (opcional, mas recomendado):

    ``` bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Instale as dependências:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Inicie o Streamlit:

    ``` bash
    streamlit run src/app.py
    ```

O Streamlit mostrará no terminal o endereço local para abrir a aplicação
no navegador.

## Estrutura do projeto

``` text
dupla-safra/
├── data/
│   └── processed/
│       └── forecast/
│           └── demise_forecast_1982-2026.nc
├── shapefiles/
├── src/
│   └── app.py
├── .gitignore
└── requirements.txt
```

## Repositório

[github.com/gmairink/dupla-safra](https://github.com/gmairink/dupla-safra)
